import argparse, os, json, math, sys, time, random
from contextlib import nullcontext
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from diffusers import AutoencoderKL

# Ensure project root on path for models import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models.unet import UNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MemmapLatentDataset(Dataset):
    """Loads embeddings + latents from prepared_data for decoder-based training.

    (embedding -> latent -> VAE decode -> image loss)
    """
    def __init__(self, root: str, split: str):
        with open(os.path.join(root, 'meta.json')) as f:
            meta = json.load(f)
        size = meta['train_size'] if split == 'train' else meta['test_size']
        self.emb = np.memmap(os.path.join(root, f'{split}_embeddings.mmap'), mode='r', dtype='float32', shape=(size, meta['embedding_dim']))
        self.lat = np.memmap(os.path.join(root, f'{split}_latents.mmap'), mode='r', dtype='float32', shape=(size, *meta['latent_shape']))
        self.meta = meta

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        e = torch.from_numpy(self.emb[idx].copy())
        lat = torch.from_numpy(self.lat[idx].copy())
        return e, lat


class LatentInversionModel(torch.nn.Module):
    def __init__(self, embedding_dim=1280, proj_C=128, spatial=16, out_channels=4):
        super().__init__()
        self.proj_C = proj_C
        self.spatial = spatial
        self.projector = torch.nn.Linear(embedding_dim, proj_C * spatial * spatial)
        self.unet = UNet(input_channels=proj_C, output_channels=out_channels, feature_size=spatial)

    def forward(self, emb):
        x = self.projector(emb)
        x = x.view(-1, self.proj_C, self.spatial, self.spatial)
        return self.unet(x)


def psnr(recon, target):
    mse = F.mse_loss(recon, target)
    if mse.item() == 0:
        return 99.0
    return 10 * torch.log10(4.0 / mse)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-root', type=str, default='prepared_data')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--proj-channels', type=int, default=128)
    ap.add_argument('--spatial', type=int, default=0, help='(Optional) internal spatial size; 0=auto from latent grid.')
    ap.add_argument('--mixed-precision', action='store_true')
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-dir', type=str, default='checkpoints_db')
    ap.add_argument('--print-freq', type=int, default=100)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--save-samples', type=int, default=4, help='Save this many reconstructed image grids each eval (0=disable).')
    ap.add_argument('--sample-dir', type=str, default='samples_db')
    ap.add_argument('--l2-weight', type=float, default=1.0, help='Weight for image-domain MSE loss.')
    ap.add_argument('--latent-loss-weight', type=float, default=0.0, help='Optional auxiliary latent MSE weight.')
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    print(f'Device: {device}')

    train_ds = MemmapLatentDataset(args.prepared_root, 'train')
    test_ds = MemmapLatentDataset(args.prepared_root, 'test')
    print(f'Train size: {len(train_ds)}  Test size: {len(test_ds)}')

    emb_dim = train_ds.emb.shape[1]
    out_ch = train_ds.lat.shape[1]
    latent_h, latent_w = train_ds.lat.shape[-2], train_ds.lat.shape[-1]
    internal_spatial = args.spatial if args.spatial > 0 else latent_h
    if internal_spatial != latent_h:
        print(f"[INFO] Internal spatial {internal_spatial} differs from latent grid {latent_h}; will upsample predicted latents.")
    model = LatentInversionModel(embedding_dim=emb_dim, proj_C=args.proj_channels, spatial=internal_spatial, out_channels=out_ch).to(device)

    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='vae').to(device)
    vae.eval()
    # Freeze VAE params (we still need autograd ops for predicted latents, but not parameter grads)
    for p in vae.parameters():
        p.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') if (args.mixed_precision and device.startswith('cuda')) else None

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=2, pin_memory=device.startswith('cuda'))

    train_loader = make_loader(train_ds, True)
    test_loader = make_loader(test_ds, False)

    start_epoch = 0
    best_psnr = -math.inf
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if scaler and ck.get('scaler'):
            scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1
        best_psnr = ck.get('best_psnr', best_psnr)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    def save_ckpt(name, epoch, pscore):
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scaler': (scaler.state_dict() if scaler else None),
            'best_psnr': pscore,
        }, os.path.join(args.save_dir, name))

    # Latent ground-truth spatial size already captured

    def decode_pred(lat_batch):
        # Allow gradient to flow from image space back to predicted latent.
        imgs = vae.decode(lat_batch / vae.config.scaling_factor).sample  # [-1,1]
        return imgs

    def decode_target(lat_batch):
        # No gradient needed for target image.
        with torch.no_grad():
            imgs = vae.decode(lat_batch / vae.config.scaling_factor).sample
        return imgs

    def train_epoch(epoch):
        model.train()
        total = 0.0
        nb = 0
        start = time.time()
        opt.zero_grad(set_to_none=True)
        for bi, (emb, lat) in enumerate(train_loader):
            emb, lat = emb.to(device, non_blocking=True), lat.to(device, non_blocking=True)
            autocast_enabled = bool(scaler and device.startswith('cuda'))
            ctx = torch.amp.autocast(device_type='cuda', enabled=autocast_enabled) if device.startswith('cuda') else torch.amp.autocast(device_type='cpu', enabled=False)
            with ctx:
                pred_lat = model(emb)
                if pred_lat.shape[-1] != latent_w or pred_lat.shape[-2] != latent_h:
                    pred_lat = torch.nn.functional.interpolate(pred_lat, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
                recon = decode_pred(pred_lat)
                target_img = decode_target(lat)
                img_loss = F.mse_loss(recon, target_img)
                aux_lat = F.mse_loss(pred_lat, lat) if args.latent_loss_weight > 0 else 0.0
                loss = (args.l2_weight * img_loss + args.latent_loss_weight * aux_lat) / args.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (bi + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            total += float(loss.item()) * args.grad_accum
            nb += 1
            if (bi + 1) % args.print_freq == 0:
                elapsed = time.time() - start
                print(f"  Batch {bi+1}/{len(train_loader)} loss={total/nb:.4f} lr={opt.param_groups[0]['lr']:.2e} ({elapsed:.1f}s)")
        return total / max(1, nb)

    def eval_epoch(epoch):
        model.eval()
        with torch.no_grad():
            psum = 0.0
            nb = 0
            for emb, lat in test_loader:
                emb, lat = emb.to(device), lat.to(device)
                pred_lat = model(emb)
                if pred_lat.shape[-1] != latent_w or pred_lat.shape[-2] != latent_h:
                    pred_lat = torch.nn.functional.interpolate(pred_lat, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
                recon = decode_pred(pred_lat)
                target_img = decode_target(lat)
                p = psnr(recon, target_img)
                if args.save_samples and nb < args.save_samples:
                    # Save recon grid progress
                    os.makedirs(args.sample_dir, exist_ok=True)
                    try:
                        import torchvision.utils as vutils
                        pair = torch.cat([target_img[:2], recon[:2]], dim=0)
                        pair = (pair * 0.5 + 0.5).clamp(0,1)
                        vutils.save_image(pair, os.path.join(args.sample_dir, f'epoch{epoch:03d}_sample{nb}.png'), nrow=2)
                    except Exception as e:
                        if nb == 0:
                            print(f"[WARN] sample save failed: {e}")
                psum += p.item(); nb += 1
        return psum / max(1, nb)

    for epoch in range(start_epoch, args.epochs):
        tr_loss = train_epoch(epoch)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={tr_loss:.4f}")
        if (epoch + 1) % args.eval_every == 0:
            pscore = eval_epoch(epoch)
            print(f"  Eval PSNR={pscore:.3f}")
            save_ckpt('last.pt', epoch, best_psnr)
            if pscore > best_psnr:
                best_psnr = pscore
                save_ckpt('best.pt', epoch, best_psnr)
                print(f"  New best PSNR {best_psnr:.3f}")

    print('Done.')


if __name__ == '__main__':
    main()