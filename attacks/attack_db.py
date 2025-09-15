import argparse, os, sys, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from diffusers import AutoencoderKL
from pytorch_msssim import ssim
from torchvision.utils import save_image
import lpips

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models.unet import UNet


class MemmapLatentDataset(Dataset):
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


class LatentModel(torch.nn.Module):
    def __init__(self, embedding_dim, proj_channels=128, spatial=16, out_channels=4):
        super().__init__()
        self.proj_channels = proj_channels
        self.spatial = spatial
        self.projector = torch.nn.Linear(embedding_dim, proj_channels * spatial * spatial)
        self.unet = UNet(proj_channels, out_channels, feature_size=spatial)

    def forward(self, emb):
        x = self.projector(emb)
        x = x.view(-1, self.proj_channels, self.spatial, self.spatial)
        return self.unet(x)


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return 99.0
    return 10 * torch.log10(4.0 / mse)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-root', type=str, default='prepared_data')
    ap.add_argument('--split', type=str, default='test', choices=['train','test'])
    ap.add_argument('--checkpoint', type=str, default='checkpoints_db/best.pt')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--proj-channels', type=int, default=128)
    ap.add_argument('--spatial', type=int, default=16)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output-dir', type=str, default='results')
    ap.add_argument('--save-grid', type=int, default=16)
    ap.add_argument('--no-lpips', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Device: {device}')

    ds = MemmapLatentDataset(args.prepared_root, args.split)
    if args.limit > 0:
        class Subset(Dataset):
            def __init__(self, base, k): self.base, self.k = base, k
            def __len__(self): return min(self.k, len(self.base))
            def __getitem__(self, i): return self.base[i]
        ds = Subset(ds, args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.startswith('cuda'))

    emb_dim = ds.emb.shape[1]
    model = LatentModel(embedding_dim=emb_dim, proj_channels=args.proj_channels, spatial=args.spatial).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck['model'] if 'model' in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()

    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='vae').to(device)
    vae.eval()

    lpips_fn = None
    if not args.no_lpips:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)

    mse_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
    n = 0
    saved_pred = []
    saved_gt = []
    with torch.no_grad():
        for emb, lat in loader:
            emb, lat = emb.to(device), lat.to(device)
            pred_lat = model(emb)
            recon = vae.decode(pred_lat / vae.config.scaling_factor).sample  # [-1,1]
            target_img = vae.decode(lat / vae.config.scaling_factor).sample  # [-1,1]
            batch_mse = F.mse_loss(recon, target_img, reduction='mean')
            batch_psnr = psnr(recon, target_img)
            r01 = (recon * 0.5 + 0.5).clamp(0,1)
            t01 = (target_img * 0.5 + 0.5).clamp(0,1)
            batch_ssim = ssim(r01, t01, data_range=1.0)
            if lpips_fn:
                batch_lp = lpips_fn(recon, target_img).mean()
            else:
                batch_lp = torch.tensor(0.0, device=device)
            bs = recon.shape[0]
            mse_sum += batch_mse.item() * bs
            psnr_sum += batch_psnr.item() * bs
            ssim_sum += batch_ssim.item() * bs
            lpips_sum += batch_lp.item() * bs
            n += bs
            if len(saved_gt) < args.save_grid:
                saved_gt.append(t01.cpu())
                saved_pred.append(r01.cpu())

    print(f"Results over {n} samples: MSE={mse_sum/n:.6f} PSNR={psnr_sum/n:.3f} SSIM={ssim_sum/n:.4f} LPIPS={lpips_sum/n:.4f}")

    if saved_gt:
        import torch
        grid_gt = torch.cat(saved_gt, dim=0)[:args.save_grid]
        grid_pred = torch.cat(saved_pred, dim=0)[:args.save_grid]
        comp = torch.cat([grid_gt, grid_pred], dim=0)
        save_image(comp, os.path.join(args.output_dir, 'db_comparison_grid.png'), nrow=args.save_grid//2)
        print(f'Saved comparison grid to {os.path.join(args.output_dir, "db_comparison_grid.png")}')


if __name__ == '__main__':
    main()