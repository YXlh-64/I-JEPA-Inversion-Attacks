import argparse, os, sys, json, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
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
        img = torch.from_numpy(self.img[idx].copy()).float() / 255.0
        img = img * 2 - 1
        return e, img


class InversionModel(torch.nn.Module):
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
    ap = argparse.ArgumentParser(description='Evaluate DMB inversion: embedding -> UNet -> z_T -> DM decode')
    ap.add_argument('--prepared-root', type=str, default='prepared_data')
    ap.add_argument('--split', type=str, default='test', choices=['train','test'])
    ap.add_argument('--checkpoint', type=str, default='checkpoints_dmb/best.pt')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--proj-channels', type=int, default=128)
    ap.add_argument('--spatial', type=int, default=0, help='Internal UNet spatial size; 0=auto (latent grid).')
    ap.add_argument('--num-inference-steps', type=int, default=30)
    ap.add_argument('--predict-latents', action='store_true', help='Match --predict-latents training mode (skip full diffusion).')
    ap.add_argument('--guidance-scale', type=float, default=1.0, help='Classifier-free guidance scale for DM (1.0 = off).')
    ap.add_argument('--normalize-latents', action='store_true', help='Normalize predicted latents to scheduler init sigma per-sample.')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output-dir', type=str, default='results')
    ap.add_argument('--save-grid', type=int, default=8)
    ap.add_argument('--no-lpips', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Device: {device}')

    ds = MemmapImageDataset(args.prepared_root, args.split)
    if args.limit > 0:
        class Subset(Dataset):
            def __init__(self, base, k): self.base, self.k = base, k
            def __len__(self): return min(self.k, len(self.base))
            def __getitem__(self, i): return self.base[i]
        ds = Subset(ds, args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.startswith('cuda'))

    emb_dim = ds.emb.shape[1]
    # Auto latent grid from meta
    lat_h, lat_w = ds.meta['latent_shape'][-2], ds.meta['latent_shape'][-1]
    spatial = args.spatial if args.spatial > 0 else lat_h
    if spatial != lat_h:
        print(f"[INFO] Using internal spatial={spatial}, latent grid={lat_h}; will upsample predicted z_T to latent grid.")
    model = InversionModel(embedding_dim=emb_dim, proj_channels=args.proj_channels, spatial=spatial).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck['model'] if 'model' in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()

    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='vae').to(device)
    vae.eval()
    pipe = None
    if not args.predict_latents:
        pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1')
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        except Exception:
            pass
        pipe = pipe.to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

    lpips_fn = None
    if not args.no_lpips:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)

    mse_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
    n = 0
    saved_gt = []
    saved_pred = []
    with torch.no_grad():
        for emb, img in loader:
            emb, img = emb.to(device), img.to(device)
            pred_lat = model(emb)
            # Ensure latent grid size
            if pred_lat.shape[-2] != lat_h or pred_lat.shape[-1] != lat_w:
                pred_lat = F.interpolate(pred_lat, size=(lat_h, lat_w), mode='bilinear', align_corners=False)
            if args.predict_latents:
                recon = vae.decode(pred_lat / vae.config.scaling_factor).sample
            else:
                if args.normalize_latents:
                    sigma = pipe.scheduler.init_noise_sigma
                    std = pred_lat.flatten(1).std(dim=1, keepdim=True).clamp(min=1e-6)
                    pred_lat = pred_lat / std.view(-1,1,1,1) * sigma
                out = pipe(prompt=[""] * pred_lat.shape[0], latents=pred_lat, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, output_type='pt')
                recon = out.images
            if recon.shape[-1] != img.shape[-1]:
                img_rs = torch.nn.functional.interpolate(img, size=recon.shape[-2:], mode='bilinear', align_corners=False)
            else:
                img_rs = img
            batch_mse = F.mse_loss(recon, img_rs, reduction='mean')
            batch_psnr = psnr(recon, img_rs)
            r01 = (recon * 0.5 + 0.5).clamp(0,1)
            t01 = (img_rs * 0.5 + 0.5).clamp(0,1)
            batch_ssim = ssim(r01, t01, data_range=1.0)
            if lpips_fn:
                batch_lp = lpips_fn(recon, img_rs).mean()
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
        grid_gt = torch.cat(saved_gt, dim=0)[:args.save_grid]
        grid_pred = torch.cat(saved_pred, dim=0)[:args.save_grid]
        comp = torch.cat([grid_gt, grid_pred], dim=0)
        save_image(comp, os.path.join(args.output_dir, 'dmb_comparison_grid.png'), nrow=args.save_grid//2)
        print(f'Saved comparison grid to {os.path.join(args.output_dir, "dmb_comparison_grid.png")}')


if __name__ == '__main__':
    main()