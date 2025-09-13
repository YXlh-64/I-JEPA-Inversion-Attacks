"""Train a simple embedding->image decoder on full CIFAR-10 leaving last 2000 images for test.

Features:
 - Uses entire CIFAR-10 (50k images) with first 48k for training, last 2k for test.
 - No random subsetting. Deterministic split.
 - Computes average PSNR/SSIM/LPIPS on test set each epoch.
 - Robust checkpoint saving: atomic write + best metric (PSNR) tracking.
 - Resume support via --resume flag (auto loads optimizer + scaler + epoch).
 - Minimal simple embedding model (linear projector + small conv decoder).
 - Embeddings are produced on-the-fly by frozen I-JEPA (memory efficient stream).
 - Avoids saving with torch.save default risks by using temporary file + rename.
"""

from __future__ import annotations
import os
import argparse
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from transformers import AutoProcessor, AutoModel

from utils.metrics import ImageMetrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--embedding-batch-size', type=int, default=128, help='batch size for forward IJepa embedding extraction')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--resize', type=int, default=224, help='resize shorter side to this size (square)')
    p.add_argument('--mixed-precision', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-dir', type=str, default='checkpoints_simple_decoder')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CIFAR10Full(Dataset):
    """CIFAR10 with deterministic split (train part) returning PIL image + index."""
    def __init__(self, root: str, train_split: bool, resize: int):
        tfm = transforms.Compose([
            transforms.Resize((resize, resize)),
            # Return PIL for processor; we convert to tensor after embedding for reconstruction target
        ])
        base = datasets.CIFAR10(root=root, train=True, download=True)
        # Deterministic ordering as given (base.data order). Use last 2000 as test.
        if train_split:
            self.indices = list(range(0, 48000))
        else:
            self.indices = list(range(48000, 50000))
        self.data = base.data
        self.tfm = tfm
        self.resize = resize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # data is numpy array HWC uint8
        from PIL import Image
        img = Image.fromarray(self.data[real_idx])
        img = self.tfm(img)
        return img


def image_to_tensor(img):
    t = TF.to_tensor(img) * 2 - 1  # [-1,1]
    return t


class SmallDecoder(nn.Module):
    def __init__(self, in_ch=128, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class InversionModel(nn.Module):
    def __init__(self, embedding_dim=1280, spatial=16, channels=128):
        super().__init__()
        self.channels = channels
        self.spatial = spatial
        self.projector = nn.Linear(embedding_dim, channels * spatial * spatial)
        self.decoder = SmallDecoder(in_ch=channels, out_ch=3)

    def forward(self, emb):
        x = self.projector(emb)
        x = x.view(-1, self.channels, self.spatial, self.spatial)
        return self.decoder(x)


def atomic_save(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    # Data
    train_ds = CIFAR10Full('./data', True, args.resize)
    test_ds = CIFAR10Full('./data', False, args.resize)

    # Models for embeddings
    processor = AutoProcessor.from_pretrained('facebook/ijepa_vith14_1k')
    backbone = AutoModel.from_pretrained('facebook/ijepa_vith14_1k').to(device)
    backbone.eval()

    # Inversion network
    inv = InversionModel().to(device)

    opt = optim.Adam(inv.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.startswith('cuda'))

    start_epoch = 0
    best_psnr = -math.inf

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        inv.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', best_psnr)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    metrics_helper = ImageMetrics(device=device)

    def iter_embed(ds, batch_size):
        for i in range(0, len(ds), batch_size):
            batch_imgs = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            # Processor expects list of PIL images
            inputs = processor(batch_imgs, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = backbone(**inputs)
                emb = out.last_hidden_state.mean(dim=1)  # [B, D]
            targets = torch.stack([image_to_tensor(img) for img in batch_imgs]).to(device)
            yield emb, targets

    for epoch in range(start_epoch, args.epochs):
        inv.train()
        total_loss = 0.0
        n_batches = 0
        for emb, target in iter_embed(train_ds, args.embedding_batch_size):
            with torch.cuda.amp.autocast(enabled=args.mixed_precision and device.startswith('cuda')):
                pred = inv(emb)
                loss_recon = torch.nn.functional.l1_loss(pred, target)
                # mild tv
                tv = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean() + (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
                loss = loss_recon + 0.05 * tv
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item())
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={avg_loss:.4f}")

        if (epoch + 1) % args.eval_every == 0:
            inv.eval()
            psnr_sum = ssid_sum = lpips_sum = 0.0
            mcount = 0
            with torch.no_grad():
                for emb, target in iter_embed(test_ds, args.embedding_batch_size):
                    pred = inv(emb)
                    batch_metrics = metrics_helper.compute(pred, target)
                    for k, v in batch_metrics.items():
                        if k == 'psnr':
                            psnr_sum += v
                        elif k == 'ssim':
                            ssid_sum += v
                        elif k == 'lpips':
                            lpips_sum += v
                    mcount += 1
            psnr_avg = psnr_sum / max(1, mcount)
            ssim_avg = ssid_sum / max(1, mcount) if ssid_sum else float('nan')
            lpips_avg = lpips_sum / max(1, mcount) if lpips_sum else float('nan')
            print(f"Eval epoch {epoch+1}: PSNR={psnr_avg:.3f} SSIM={ssim_avg:.3f} LPIPS={lpips_avg:.3f}")

            # Save checkpoint
            ckpt_path = os.path.join(args.save_dir, 'last.pt')
            atomic_save({'epoch': epoch, 'model': inv.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'best_psnr': best_psnr}, ckpt_path)
            if psnr_avg > best_psnr:
                best_psnr = psnr_avg
                best_path = os.path.join(args.save_dir, 'best.pt')
                atomic_save({'epoch': epoch, 'model': inv.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'best_psnr': best_psnr}, best_path)
                print(f"  New best PSNR {best_psnr:.3f} saved -> {best_path}")

    print('Training complete.')


if __name__ == '__main__':
    main()
