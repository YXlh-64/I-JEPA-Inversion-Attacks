import argparse, os, math, json, sys, random, time
import numpy as np
import argparse, os, math, json, sys, random, time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
# Ensure project root (parent of scripts/) is on sys.path for "models" package import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
# Ensure project root (parent of scripts/) is on sys.path for "models" package import
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


class MemmapPreparedDataset(Dataset):
    """Loads embeddings + either images or latents from prepared_data memmaps.

    When use_images=True it returns (embedding, image_tensor[-1,1] scaled).
    Otherwise returns (embedding, latent_tensor) for latent prediction variant.
    """
    def __init__(self, root: str, split: str, use_images: bool = True):
        with open(os.path.join(root, 'meta.json')) as f:
            meta = json.load(f)
        self.split = split
        self.use_images = use_images
        size = meta['train_size'] if split == 'train' else meta['test_size']
        self.emb = np.memmap(os.path.join(root, f'{split}_embeddings.mmap'), mode='r', dtype='float32', shape=(size, meta['embedding_dim']))
        self.img = np.memmap(os.path.join(root, f'{split}_images.mmap'), mode='r', dtype='uint8', shape=(size, *meta['image_shape']))
        self.lat = np.memmap(os.path.join(root, f'{split}_latents.mmap'), mode='r', dtype='float32', shape=(size, *meta['latent_shape']))
        self.meta = meta

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        # Copy to avoid non-writable memmap warning and ensure tensor is writable for potential in-place ops
        e = torch.from_numpy(self.emb[idx].copy())  # float32
        if self.use_images:
            # uint8 -> float in [-1,1]
            img = torch.from_numpy(self.img[idx].copy()).float() / 255.0
            img = img * 2 - 1
            return e, img
        else:
            lat = torch.from_numpy(self.lat[idx].copy())  # float32 latents already
            return e, lat


class DirectOptimizationModel(torch.nn.Module):
    """Embedding -> image with small UNet backbone.

    For image mode (default) output channels=3, for latent mode output channels=latent C.
    spatial and channels define internal projected tensor shape (C, H, W) before UNet.
    """
    def __init__(self, embedding_dim=1280, proj_channels=32, spatial=64, out_channels=3):
        super().__init__()
        self.proj_channels = proj_channels
        self.spatial = spatial
        # Project embedding to spatial feature map then refine with UNet
        self.projector = torch.nn.Linear(embedding_dim, proj_channels * spatial * spatial)
        self.unet = UNet(proj_channels, out_channels, feature_size=spatial)

    def forward(self, emb):
        x = self.projector(emb)
        x = x.view(-1, self.proj_channels, self.spatial, self.spatial)
        return self.unet(x)


def psnr(recon, target):
    mse = F.mse_loss(recon, target)
    if mse.item() == 0:
        return 99.0
    return 10 * torch.log10(4.0 / mse)  # range [-1,1] => peak-to-peak 2 => P^2=4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-root', type=str, default='prepared_data', help='Directory produced by prepare_dataset.py')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--use-images', action='store_true', help='Predict images (default).')
    ap.add_argument('--use-latents', action='store_true', help='Predict latents instead of images (overrides use-images).')
    ap.add_argument('--mixed-precision', action='store_true')
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-dir', type=str, default='checkpoints_do')
    ap.add_argument('--proj-channels', type=int, default=32)
    ap.add_argument('--spatial', type=int, default=64, help='Spatial size for projected tensor fed to UNet')
    ap.add_argument('--latent-channels', type=int, default=4, help='Latent channel count when predicting latents')
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--print-freq', type=int, default=100, help='Batches between progress prints (if no tqdm).')
    ap.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps.')
    ap.add_argument('--cosine', action='store_true', help='Use cosine LR schedule.')
    ap.add_argument('--save-samples', type=int, default=0, help='If >0, save this many recon samples each eval.')
    ap.add_argument('--sample-dir', type=str, default='samples_do')
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    set_seed(args.seed)

    use_latents = args.use_latents
    if args.use_latents:
        mode_desc = 'latent prediction'
    else:
        mode_desc = 'image prediction'
    print(f"Mode: {mode_desc}")

    # Datasets
    train_ds = MemmapPreparedDataset(args.prepared_root, 'train', use_images=not use_latents)
    test_ds = MemmapPreparedDataset(args.prepared_root, 'test', use_images=not use_latents)
    print(f"Train size: {len(train_ds)}  Test size: {len(test_ds)}")

    out_ch = train_ds.lat.shape[1] if use_latents else train_ds.img.shape[1]
    model = DirectOptimizationModel(
        embedding_dim=train_ds.emb.shape[1],
        proj_channels=args.proj_channels,
        spatial=args.spatial,
        out_channels=(train_ds.lat.shape[1] if use_latents else train_ds.img.shape[1])
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
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
            'use_latents': use_latents,
        }, os.path.join(args.save_dir, name))

    def train_epoch(epoch):
        model.train()
        total = 0.0
        nb = 0
        start_time = time.time()
        opt.zero_grad(set_to_none=True)
        for batch_idx, (emb, tgt) in enumerate(train_loader):
            emb, tgt = emb.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            if not use_latents and (tgt.shape[-1] != args.spatial or tgt.shape[-2] != args.spatial):
                tgt = torch.nn.functional.interpolate(tgt, size=(args.spatial, args.spatial), mode='bilinear', align_corners=False)
            autocast_enabled = bool(scaler and device.startswith('cuda'))
            ctx = torch.amp.autocast(device_type='cuda', enabled=autocast_enabled) if device.startswith('cuda') else torch.amp.autocast(device_type='cpu', enabled=False)
            with ctx:
                pred = model(emb)
                if not use_latents:
                    l1 = torch.nn.functional.l1_loss(pred, tgt)
                    tv = (pred[:,:,1:,:]-pred[:,:,:-1,:]).abs().mean() + (pred[:,:,:,1:]-pred[:,:,:,:-1]).abs().mean()
                    loss = l1 + 0.05 * tv
                else:
                    loss = torch.nn.functional.mse_loss(pred, tgt)
                loss = loss / args.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            total += float(loss.item()) * args.grad_accum
            nb += 1
            if (batch_idx + 1) % args.print_freq == 0:
                elapsed = time.time() - start_time
                lr_cur = opt.param_groups[0]['lr']
                print(f"  Batch {batch_idx+1}/{len(train_loader)} loss={total/nb:.4f} lr={lr_cur:.2e} ({elapsed:.1f}s)")
        return total / max(1, nb)

    def eval_epoch(epoch):
        model.eval()
        with torch.no_grad():
            psum = 0.0
            nb = 0
            saved = 0
            for emb, tgt in test_loader:
                emb, tgt = emb.to(device), tgt.to(device)
                if not use_latents and (tgt.shape[-1] != args.spatial or tgt.shape[-2] != args.spatial):
                    tgt = torch.nn.functional.interpolate(tgt, size=(args.spatial, args.spatial), mode='bilinear', align_corners=False)
                pred = model(emb)
                p = psnr(pred, tgt)
                psum += p.item()
                nb += 1
                if args.save_samples > 0 and saved < args.save_samples and not use_latents:
                    # Save reconstruction grid
                    os.makedirs(args.sample_dir, exist_ok=True)
                    import torchvision.utils as vutils
                    # Denormalize [-1,1] -> [0,1]
                    grid = torch.cat([tgt[:4], pred[:4]], dim=0)
                    grid = (grid * 0.5 + 0.5).clamp(0,1)
                    vutils.save_image(grid, os.path.join(args.sample_dir, f'epoch{epoch:03d}_sample{saved}.png'), nrow=4)
                    saved += 1
        return psum / max(1, nb)

    for epoch in range(start_epoch, args.epochs):
        tr_loss = train_epoch(epoch)
        if scheduler:
            scheduler.step()
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
        self.proj_channels = proj_channels
        self.spatial = spatial
        # Project embedding to spatial feature map then refine with UNet
        self.projector = torch.nn.Linear(embedding_dim, proj_channels * spatial * spatial)
        self.unet = UNet(proj_channels, out_channels, feature_size=spatial)

    def forward(self, emb):
        x = self.projector(emb)
        x = x.view(-1, self.proj_channels, self.spatial, self.spatial)
        return self.unet(x)


def psnr(recon, target):
    mse = F.mse_loss(recon, target)
    if mse.item() == 0:
        return 99.0
    return 10 * torch.log10(4.0 / mse)  # range [-1,1] => peak-to-peak 2 => P^2=4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-root', type=str, default='prepared_data', help='Directory produced by prepare_dataset.py')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--use-images', action='store_true', help='Predict images (default).')
    ap.add_argument('--use-latents', action='store_true', help='Predict latents instead of images (overrides use-images).')
    ap.add_argument('--mixed-precision', action='store_true')
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-dir', type=str, default='checkpoints_do')
    ap.add_argument('--proj-channels', type=int, default=32)
    ap.add_argument('--spatial', type=int, default=64, help='Spatial size for projected tensor fed to UNet')
    ap.add_argument('--latent-channels', type=int, default=4, help='Latent channel count when predicting latents')
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--print-freq', type=int, default=100, help='Batches between progress prints (if no tqdm).')
    ap.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps.')
    ap.add_argument('--cosine', action='store_true', help='Use cosine LR schedule.')
    ap.add_argument('--save-samples', type=int, default=0, help='If >0, save this many recon samples each eval.')
    ap.add_argument('--sample-dir', type=str, default='samples_do')
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    set_seed(args.seed)

    use_latents = args.use_latents
    if args.use_latents:
        mode_desc = 'latent prediction'
    else:
        mode_desc = 'image prediction'
    print(f"Mode: {mode_desc}")

    # Datasets
    train_ds = MemmapPreparedDataset(args.prepared_root, 'train', use_images=not use_latents)
    test_ds = MemmapPreparedDataset(args.prepared_root, 'test', use_images=not use_latents)
    print(f"Train size: {len(train_ds)}  Test size: {len(test_ds)}")

    out_ch = train_ds.lat.shape[1] if use_latents else train_ds.img.shape[1]
    model = DirectOptimizationModel(
        embedding_dim=train_ds.emb.shape[1],
        proj_channels=args.proj_channels,
        spatial=args.spatial,
        out_channels=(train_ds.lat.shape[1] if use_latents else train_ds.img.shape[1])
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
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
            'use_latents': use_latents,
        }, os.path.join(args.save_dir, name))

    def train_epoch(epoch):
        model.train()
        total = 0.0
        nb = 0
        start_time = time.time()
        opt.zero_grad(set_to_none=True)
        for batch_idx, (emb, tgt) in enumerate(train_loader):
            emb, tgt = emb.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            if not use_latents and (tgt.shape[-1] != args.spatial or tgt.shape[-2] != args.spatial):
                tgt = torch.nn.functional.interpolate(tgt, size=(args.spatial, args.spatial), mode='bilinear', align_corners=False)
            autocast_enabled = bool(scaler and device.startswith('cuda'))
            ctx = torch.amp.autocast(device_type='cuda', enabled=autocast_enabled) if device.startswith('cuda') else torch.amp.autocast(device_type='cpu', enabled=False)
            with ctx:
                pred = model(emb)
                if not use_latents:
                    l1 = torch.nn.functional.l1_loss(pred, tgt)
                    tv = (pred[:,:,1:,:]-pred[:,:,:-1,:]).abs().mean() + (pred[:,:,:,1:]-pred[:,:,:,:-1]).abs().mean()
                    loss = l1 + 0.05 * tv
                else:
                    loss = torch.nn.functional.mse_loss(pred, tgt)
                loss = loss / args.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            total += float(loss.item()) * args.grad_accum
            nb += 1
            if (batch_idx + 1) % args.print_freq == 0:
                elapsed = time.time() - start_time
                lr_cur = opt.param_groups[0]['lr']
                print(f"  Batch {batch_idx+1}/{len(train_loader)} loss={total/nb:.4f} lr={lr_cur:.2e} ({elapsed:.1f}s)")
        return total / max(1, nb)

    def eval_epoch(epoch):
        model.eval()
        with torch.no_grad():
            psum = 0.0
            nb = 0
            saved = 0
            for emb, tgt in test_loader:
                emb, tgt = emb.to(device), tgt.to(device)
                if not use_latents and (tgt.shape[-1] != args.spatial or tgt.shape[-2] != args.spatial):
                    tgt = torch.nn.functional.interpolate(tgt, size=(args.spatial, args.spatial), mode='bilinear', align_corners=False)
                pred = model(emb)
                p = psnr(pred, tgt)
                psum += p.item()
                nb += 1
                if args.save_samples > 0 and saved < args.save_samples and not use_latents:
                    # Save reconstruction grid
                    os.makedirs(args.sample_dir, exist_ok=True)
                    import torchvision.utils as vutils
                    # Denormalize [-1,1] -> [0,1]
                    grid = torch.cat([tgt[:4], pred[:4]], dim=0)
                    grid = (grid * 0.5 + 0.5).clamp(0,1)
                    vutils.save_image(grid, os.path.join(args.sample_dir, f'epoch{epoch:03d}_sample{saved}.png'), nrow=4)
                    saved += 1
        return psum / max(1, nb)

    for epoch in range(start_epoch, args.epochs):
        tr_loss = train_epoch(epoch)
        if scheduler:
            scheduler.step()
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