"""Train a simple embedding->image decoder on full CIFAR-10 leaving last 2000 images for test.

Features:
 - Full CIFAR-10 (first 48k train, last 2k test) deterministic split.
 - On-the-fly IJepa embeddings (backbone frozen).
 - UNet-like tiny decoder (linear projector -> conv transpose stack).
 - Dynamic output size equals --resize (default 224).
 - Mixed precision (optional) with torch.amp APIs.
 - Atomic checkpointing (last + best by PSNR) and resume.
 - Average PSNR/SSIM/LPIPS over test batches.
"""

from __future__ import annotations
import os, argparse, math
from typing import Dict

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, AutoModel
from utils.metrics import ImageMetrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--embedding-batch-size', type=int, default=None, help='(deprecated) separate embedding extraction batch size; overrides --batch-size if set')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--resize', type=int, default=224, help='resize shorter side to this size (square)')
    p.add_argument('--mixed-precision', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-dir', type=str, default='checkpoints_simple_decoder')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--print-freq', type=int, default=100, help='Print every N batches when tqdm disabled')
    p.add_argument('--no-tqdm', action='store_true', help='Disable tqdm progress bars')
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
    def __init__(self, root: str, train_split: bool, resize: int):
        self.tfm = transforms.Compose([transforms.Resize((resize, resize))])
        base = datasets.CIFAR10(root=root, train=True, download=True)
        self.indices = list(range(0, 48000)) if train_split else list(range(48000, 50000))
        self.data = base.data
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        from PIL import Image
        return self.tfm(Image.fromarray(self.data[self.indices[idx]]))


def image_to_tensor(img):
    return TF.to_tensor(img) * 2 - 1


class SmallDecoder(nn.Module):
    def __init__(self, in_ch=128, out_ch=3, target_size=224):
        super().__init__()
        self.target_size = target_size
        self.body = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 256, 4, 2, 1),  # 16->32
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # 32->64
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # 64->128
            nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Upsample(size=(target_size, target_size), mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


class InversionModel(nn.Module):
    def __init__(self, embedding_dim=1280, spatial=16, channels=128, target_size=224):
        super().__init__()
        self.channels = channels
        self.spatial = spatial
        self.projector = nn.Linear(embedding_dim, channels * spatial * spatial)
        self.decoder = SmallDecoder(in_ch=channels, out_ch=3, target_size=target_size)

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

    # NOTE: cuFFT/cuDNN/cuBLAS "already registered" warnings are benign (multiple framework initializations).
    # To suppress some backend verbosity you may set: os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' before imports.

    # Data
    train_ds = CIFAR10Full('./data', True, args.resize)
    test_ds = CIFAR10Full('./data', False, args.resize)

    # Models for embeddings
    processor = AutoProcessor.from_pretrained('facebook/ijepa_vith14_1k')
    backbone = AutoModel.from_pretrained('facebook/ijepa_vith14_1k').to(device)
    backbone.eval()

    # Inversion network
    inv = InversionModel(target_size=args.resize).to(device)

    opt = optim.Adam(inv.parameters(), lr=args.lr)
    scaler = (torch.amp.GradScaler('cuda') if (args.mixed_precision and device.startswith('cuda')) else None)

    start_epoch = 0
    best_psnr = -math.inf

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        inv.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        if scaler and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', best_psnr)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    metrics_helper = ImageMetrics(device=device)

    embed_bs = args.embedding_batch_size if args.embedding_batch_size else args.batch_size

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

    use_tqdm = not args.no_tqdm
    if use_tqdm:
        try:
            from tqdm import tqdm
        except ImportError:
            use_tqdm = False

    from contextlib import nullcontext

    for epoch in range(start_epoch, args.epochs):
        inv.train()
        total_loss = 0.0
        n_batches = 0
        printed_debug = False
        train_iter = iter_embed(train_ds, embed_bs)
        if use_tqdm:
            train_iter = tqdm(train_iter, total=math.ceil(len(train_ds)/embed_bs), desc=f"Train {epoch+1}/{args.epochs}", leave=False)
        for emb, target in train_iter:
            if args.mixed_precision and device.startswith('cuda'):
                autocast_ctx = torch.amp.autocast('cuda')
            else:
                autocast_ctx = nullcontext()
            with autocast_ctx:
                pred = inv(emb)
                if not printed_debug:
                    print(f"Debug shapes -> pred: {list(pred.shape)} target: {list(target.shape)}")
                    printed_debug = True
                if pred.shape[-2:] != target.shape[-2:]:
                    target = torch.nn.functional.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
                loss_recon = torch.nn.functional.l1_loss(pred, target)
                tv = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean() + (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
                loss = loss_recon + 0.05 * tv
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            total_loss += float(loss.item())
            n_batches += 1
            if (not use_tqdm) and args.print_freq > 0 and (n_batches % args.print_freq == 0):
                print(f"Epoch {epoch+1} batch {n_batches} loss={loss.item():.4f}")
        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={avg_loss:.4f}")

        if (epoch + 1) % args.eval_every == 0:
            inv.eval()
            psnr_sum = ssid_sum = lpips_sum = 0.0
            mcount = 0
            with torch.no_grad():
                eval_iter = iter_embed(test_ds, embed_bs)
                if use_tqdm:
                    eval_iter = tqdm(eval_iter, total=math.ceil(len(test_ds)/embed_bs), desc=f"Eval {epoch+1}", leave=False)
                for emb, target in eval_iter:
                    pred = inv(emb)
                    if pred.shape[-2:] != target.shape[-2:]:
                        target = torch.nn.functional.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
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
            atomic_save({'epoch': epoch, 'model': inv.state_dict(), 'opt': opt.state_dict(), 'scaler': (scaler.state_dict() if scaler else None), 'best_psnr': best_psnr}, ckpt_path)
            if psnr_avg > best_psnr:
                best_psnr = psnr_avg
                best_path = os.path.join(args.save_dir, 'best.pt')
                atomic_save({'epoch': epoch, 'model': inv.state_dict(), 'opt': opt.state_dict(), 'scaler': (scaler.state_dict() if scaler else None), 'best_psnr': best_psnr}, best_path)
                print(f"  New best PSNR {best_psnr:.3f} saved -> {best_path}")

    print('Training complete.')


if __name__ == '__main__':
    main()
