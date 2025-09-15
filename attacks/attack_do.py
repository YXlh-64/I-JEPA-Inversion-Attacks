import argparse, os, sys, json, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from pytorch_msssim import ssim
from torchvision.utils import save_image
import lpips

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models.unet import UNet


class MemmapImageDataset(Dataset):
    def __init__(self, root: str, split: str):
        with open(os.path.join(root, 'meta.json')) as f:
            meta = json.load(f)
        size = meta['train_size'] if split == 'train' else meta['test_size']
        self.emb = np.memmap(os.path.join(root, f'{split}_embeddings.mmap'), mode='r', dtype='float32', shape=(size, meta['embedding_dim']))
        self.img = np.memmap(os.path.join(root, f'{split}_images.mmap'), mode='r', dtype='uint8', shape=(size, *meta['image_shape']))
        self.meta = meta

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        e = torch.from_numpy(self.emb[idx].copy())
        img = torch.from_numpy(self.img[idx].copy()).float() / 255.0  # [0,1]
        img = img * 2 - 1  # [-1,1]
        return e, img


class DirectOptimizationModel(torch.nn.Module):
    def __init__(self, embedding_dim, proj_channels=32, spatial=64, out_channels=3):
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
    ap.add_argument('--checkpoint', type=str, default='checkpoints_do/best.pt')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--proj-channels', type=int, default=32)
    ap.add_argument('--spatial', type=int, default=64)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of samples (0=all)')
    ap.add_argument('--output-dir', type=str, default='results')
    ap.add_argument('--save-grid', type=int, default=16, help='Number of examples to save in comparison grid (even number)')
    ap.add_argument('--no-lpips', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f'Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    ds = MemmapImageDataset(args.prepared_root, args.split)
    if args.limit > 0:
        class Subset(Dataset):
            def __init__(self, base, k): self.base, self.k = base, k
            def __len__(self): return min(self.k, len(self.base))
            def __getitem__(self, i): return self.base[i]
        ds = Subset(ds, args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.startswith('cuda'))

    emb_dim = ds.emb.shape[1]
    model = DirectOptimizationModel(embedding_dim=emb_dim, proj_channels=args.proj_channels, spatial=args.spatial, out_channels=3).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck['model'] if 'model' in ck else ck
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print('Warning: missing keys', missing.missing_keys)
    model.eval()

    lpips_fn = None
    if not args.no_lpips:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)

    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    n = 0
    saved_imgs_pred = []
    saved_imgs_gt = []
    with torch.no_grad():
        for emb, img in loader:
            emb, img = emb.to(device), img.to(device)
            if img.shape[-1] != args.spatial or img.shape[-2] != args.spatial:
                img_rs = torch.nn.functional.interpolate(img, size=(args.spatial, args.spatial), mode='bilinear', align_corners=False)
            else:
                img_rs = img
            pred = model(emb)
            # Metrics
            batch_mse = F.mse_loss(pred, img_rs, reduction='mean')
            batch_psnr = psnr(pred, img_rs)
            pred_01 = (pred * 0.5 + 0.5).clamp(0,1)
            gt_01 = (img_rs * 0.5 + 0.5).clamp(0,1)
            batch_ssim = ssim(pred_01, gt_01, data_range=1.0)
            if lpips_fn:
                batch_lp = lpips_fn(pred, img_rs).mean()
            else:
                batch_lp = torch.tensor(0.0, device=device)
            bs = pred.shape[0]
            mse_sum += batch_mse.item() * bs
            psnr_sum += batch_psnr.item() * bs
            ssim_sum += batch_ssim.item() * bs
            lpips_sum += batch_lp.item() * bs
            n += bs
            if len(saved_imgs_gt) * 1 < args.save_grid and saved_imgs_gt is not None:
                saved_imgs_gt.append(gt_01.cpu())
                saved_imgs_pred.append(pred_01.cpu())

    print(f"Results over {n} samples: MSE={mse_sum/n:.6f} PSNR={psnr_sum/n:.3f} SSIM={ssim_sum/n:.4f} LPIPS={lpips_sum/n:.4f}")

    if saved_imgs_gt:
        import torch
        grid_gt = torch.cat(saved_imgs_gt, dim=0)[:args.save_grid]
        grid_pred = torch.cat(saved_imgs_pred, dim=0)[:args.save_grid]
        comp = torch.cat([grid_gt, grid_pred], dim=0)
        save_image(comp, os.path.join(args.output_dir, 'do_comparison_grid.png'), nrow=args.save_grid//2)
        print(f'Saved comparison grid to {os.path.join(args.output_dir, "do_comparison_grid.png")}')


if __name__ == '__main__':
    main()
