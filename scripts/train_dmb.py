
import argparse, os, json, math, sys, time, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

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
    """Loads embeddings + images from prepared_data (uint8 -> float [-1,1])."""
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


class ZTInversionModel(torch.nn.Module):
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
    ap = argparse.ArgumentParser(description='Diffusion-model-based inversion: embedding -> UNet -> z_T -> DM decode')
    ap.add_argument('--prepared-root', type=str, default='prepared_data')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--proj-channels', type=int, default=128)
    ap.add_argument('--spatial', type=int, default=0, help='Internal UNet spatial size; 0=auto (latent grid).')
    ap.add_argument('--mixed-precision', action='store_true')
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-dir', type=str, default='checkpoints_dmb')
    ap.add_argument('--print-freq', type=int, default=50)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--num-inference-steps', type=int, default=30, help='Diffusion steps during training inference')
    ap.add_argument('--predict-latents', action='store_true', help='Predict initial latent then decode with VAE (skip full SD pipeline).')
    ap.add_argument('--guidance-scale', type=float, default=1.0, help='Classifier-free guidance scale for DM (1.0 = off).')
    ap.add_argument('--normalize-latents', action='store_true', help='Normalize predicted latents to scheduler init sigma per-sample.')
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    print(f'Device: {device}')

    train_ds = MemmapImageDataset(args.prepared_root, 'train')
    test_ds = MemmapImageDataset(args.prepared_root, 'test')
    print(f'Train size: {len(train_ds)}  Test size: {len(test_ds)}')

    emb_dim = train_ds.emb.shape[1]
    # Read latent grid from meta for auto sizing
    with open(os.path.join(args.prepared_root, 'meta.json')) as f:
        meta = json.load(f)
    lat_h, lat_w = meta['latent_shape'][-2], meta['latent_shape'][-1]
    spatial = args.spatial if args.spatial > 0 else lat_h
    if spatial != lat_h:
        print(f"[INFO] Using internal spatial={spatial}, latent grid={lat_h}; will upsample predicted z_T to latent grid.")
    model = ZTInversionModel(embedding_dim=emb_dim, proj_C=args.proj_channels, spatial=spatial, out_channels=4).to(device)

    # For predict-latents fast path use only VAE (like decoder-based), else full SD pipeline from predicted z_T
    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='vae').to(device)
    vae.eval()
    pipe = None
    if not args.predict_latents:
        pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1')
        # optional memory savings
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        except Exception:
            pass
        pipe = pipe.to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

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

    def decode_from_pred(pred):
        if args.predict_latents:
            with torch.no_grad():
                # Ensure latent grid
                if pred.shape[-1] != lat_w or pred.shape[-2] != lat_h:
                    pred = torch.nn.functional.interpolate(pred, size=(lat_h, lat_w), mode='bilinear', align_corners=False)
                return vae.decode(pred / vae.config.scaling_factor).sample
        # Full diffusion path: treat pred as starting latent (initial noise) and run denoise
        with torch.no_grad():
            # Match latent grid and scale to scheduler init sigma if requested
            if pred.shape[-1] != lat_w or pred.shape[-2] != lat_h:
                pred = torch.nn.functional.interpolate(pred, size=(lat_h, lat_w), mode='bilinear', align_corners=False)
            if args.normalize_latents:
                sigma = pipe.scheduler.init_noise_sigma
                std = pred.flatten(1).std(dim=1, keepdim=True).clamp(min=1e-6)
                pred = pred / std.view(-1,1,1,1) * sigma
            out = pipe(prompt=[""] * pred.shape[0], latents=pred, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, output_type='pt')
            return out.images

    def train_epoch(epoch):
        model.train()
        total = 0.0
        nb = 0
        start = time.time()
        opt.zero_grad(set_to_none=True)
        for bi, (emb, img) in enumerate(train_loader):
            emb, img = emb.to(device, non_blocking=True), img.to(device, non_blocking=True)
            autocast_enabled = bool(scaler and device.startswith('cuda'))
            ctx = torch.amp.autocast(device_type='cuda', enabled=autocast_enabled) if device.startswith('cuda') else torch.amp.autocast(device_type='cpu', enabled=False)
            with ctx:
                pred_lat = model(emb)
                recon = decode_from_pred(pred_lat)
                # Resize target if needed to model spatial *some mismatch due to 512 vs chosen spatial upscale*
                if recon.shape[-1] != img.shape[-1]:
                    img_rs = torch.nn.functional.interpolate(img, size=recon.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    img_rs = img
                loss = F.mse_loss(recon, img_rs) / args.grad_accum
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
            for emb, img in test_loader:
                emb, img = emb.to(device), img.to(device)
                pred_lat = model(emb)
                recon = decode_from_pred(pred_lat)
                if recon.shape[-1] != img.shape[-1]:
                    img = torch.nn.functional.interpolate(img, size=recon.shape[-2:], mode='bilinear', align_corners=False)
                p = psnr(recon, img)
                psum += p.item()
                nb += 1
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
