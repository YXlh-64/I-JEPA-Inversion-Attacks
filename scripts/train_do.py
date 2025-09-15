import argparse, os, math, json, sys, random, time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torchvision.utils as vutils

# Single clean import section
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


class MemmapImageDataset(Dataset):
    """Embedding -> image pairs from prepared_data (for direct inversion)."""
    def __init__(self, root: str, split: str):
        with open(os.path.join(root, 'meta.json')) as f:
            meta = json.load(f)
        size = meta['train_size'] if split == 'train' else meta['test_size']
        self.emb = np.memmap(os.path.join(root, f'{split}_embeddings.mmap'), mode='r', dtype='float32', shape=(size, meta['embedding_dim']))
        # images stored uint8 0..255
        self.img = np.memmap(os.path.join(root, f'{split}_images.mmap'), mode='r', dtype='uint8', shape=(size, *meta['image_shape']))
        self.meta = meta

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        e = torch.from_numpy(self.emb[idx].copy())
        img = torch.from_numpy(self.img[idx].copy()).float() / 255.0
        img = img * 2 - 1  # scale to [-1,1]
        return e, img


class InversionUNet(torch.nn.Module):
    """Embedding -> (project to spatial) -> U-Net -> upsample -> image."""
    def __init__(self, embedding_dim=1280, proj_channels=64, spatial=32, out_channels=3, output_size=512, up_mode='bilinear'):
        super().__init__()
        self.proj_channels = proj_channels
        self.spatial = spatial
        self.output_size = output_size
        self.up_mode = up_mode
        self.projector = torch.nn.Linear(embedding_dim, proj_channels * spatial * spatial)
        self.unet = UNet(proj_channels, proj_channels, feature_size=spatial)
        self.head = torch.nn.Conv2d(proj_channels, out_channels, 1)

    def forward(self, emb):
        x = self.projector(emb)
        x = x.view(-1, self.proj_channels, self.spatial, self.spatial)
        x = self.unet(x)
        x = self.head(x)
        if x.shape[-1] != self.output_size:
            x = torch.nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode=self.up_mode, align_corners=False if self.up_mode=='bilinear' else None)
        return x


def psnr(recon, target):
    mse = F.mse_loss(recon, target)
    if mse.item() == 0:
        return 99.0
    return 10 * torch.log10(4.0 / mse)  # range [-1,1] => peak-to-peak 2 => P^2=4


def parse_args():
    ap = argparse.ArgumentParser(description='Direct embedding->image inversion attack (UNet).')
    ap.add_argument('--prepared-root', type=str, default='prepared_data')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--proj-channels', type=int, default=64)
    ap.add_argument('--spatial', type=int, default=32, help='Internal projected spatial size.')
    ap.add_argument('--output-size', type=int, default=512, help='Final reconstructed image size.')
    ap.add_argument('--mixed-precision', action='store_true')
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-dir', type=str, default='checkpoints_do')
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--print-freq', type=int, default=100)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--save-samples', type=int, default=4)
    ap.add_argument('--sample-dir', type=str, default='samples_do')
    ap.add_argument('--l1-weight', type=float, default=1.0)
    ap.add_argument('--tv-weight', type=float, default=0.05)
    ap.add_argument('--mse-weight', type=float, default=0.0, help='Optional extra MSE weight in loss.')
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")
    set_seed(args.seed)
    # Dataset
    train_ds = MemmapImageDataset(args.prepared_root, 'train')
    test_ds = MemmapImageDataset(args.prepared_root, 'test')
    print(f"Train size: {len(train_ds)}  Test size: {len(test_ds)}")
    emb_dim = train_ds.emb.shape[1]
    model = InversionUNet(embedding_dim=emb_dim, proj_channels=args.proj_channels, spatial=args.spatial, out_channels=3, output_size=args.output_size).to(device)

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
        }, os.path.join(args.save_dir, name))

    def tv_loss(x):
        return (x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean() + (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean()

    def train_epoch(epoch):
        model.train()
        total = 0.0
        nb = 0
        start_time = time.time()
        opt.zero_grad(set_to_none=True)
        for batch_idx, (emb, tgt) in enumerate(train_loader):
            emb, tgt = emb.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            # Ensure target at desired output size for fair loss
            if tgt.shape[-1] != args.output_size:
                tgt = torch.nn.functional.interpolate(tgt, size=(args.output_size, args.output_size), mode='bilinear', align_corners=False)
            autocast_enabled = bool(scaler and device.startswith('cuda'))
            ctx = torch.amp.autocast(device_type='cuda', enabled=autocast_enabled) if device.startswith('cuda') else torch.amp.autocast(device_type='cpu', enabled=False)
            with ctx:
                pred = model(emb)
                l1 = F.l1_loss(pred, tgt)*args.l1_weight
                tv = tv_loss(pred)*args.tv_weight
                mse_term = F.mse_loss(pred, tgt)*args.mse_weight if args.mse_weight>0 else 0.0
                loss = (l1 + tv + mse_term) / args.grad_accum
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
                if tgt.shape[-1] != args.output_size:
                    tgt = torch.nn.functional.interpolate(tgt, size=(args.output_size, args.output_size), mode='bilinear', align_corners=False)
                pred = model(emb)
                p = psnr(pred, tgt)
                psum += p.item(); nb += 1
                if args.save_samples > 0 and saved < args.save_samples:
                    os.makedirs(args.sample_dir, exist_ok=True)
                    import torchvision.utils as vutils
                    grid = torch.cat([tgt[:4], pred[:4]], dim=0)
                    grid = (grid*0.5 + 0.5).clamp(0,1)
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