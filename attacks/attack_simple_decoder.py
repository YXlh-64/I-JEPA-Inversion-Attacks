import argparse
import torch
import numpy as np
from torch.nn.functional import mse_loss
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from pytorch_msssim import ssim
from transformers import AutoProcessor, AutoModel
import lpips
from PIL import Image
import os


parser = argparse.ArgumentParser()
parser.add_argument("--custom", action="store_true", default=False,
                    help="If set, run on custom images instead of CIFAR test pairs.")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation over test set")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
os.makedirs('results', exist_ok=True)

# ------------------------------
# Model definitions (must match training)
# ------------------------------
class SmallDecoder(torch.nn.Module):
    def __init__(self, in_ch=128, out_ch=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_ch, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(size=(224,224), mode='bilinear', align_corners=False),
            torch.nn.Conv2d(32, out_ch, kernel_size=3, padding=1),
            torch.nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class InversionModel(torch.nn.Module):
    def __init__(self, embedding_dim=1280, proj_C=128, proj_H=16, proj_W=16):
        super().__init__()
        self.projector = torch.nn.Linear(embedding_dim, proj_C * proj_H * proj_W)
        self.decoder = SmallDecoder(in_ch=proj_C, out_ch=3)
    def forward(self, emb):
        x = self.projector(emb)
        x = x.view(-1, 128, 16, 16)
        return self.decoder(x)

# load trained inversion model
f_inv = InversionModel().to(device)
f_inv.load_state_dict(torch.load('saved_models/sd_inv.pth', map_location=device))
f_inv.eval()

# LPIPS
loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

to_tensor = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)   # scale [0,1] -> [-1,1]
])

# ------------------------------
# MODE 1: Custom Images
# ------------------------------
if args.custom:
    custom_dir = "custom"
    os.makedirs(custom_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained("facebook/ijepa_vith14_1k")
    ijepa = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").to(device).eval()

    files = [f for f in os.listdir(custom_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]

    originals, reconstructions = [], []

    for i, fname in enumerate(files):
        path = os.path.join(custom_dir, fname)
        img = Image.open(path).convert("RGB").resize((224,224))

        img_tensor = to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            inputs = processor(img, return_tensors="pt").to(device)
            outputs = ijepa(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)  # [1,1280]

            recon = f_inv(emb)

            # Metrics
            mse_val = mse_loss(recon, img_tensor).item()
            img_norm = (img_tensor + 1) / 2
            recon_norm = (recon + 1) / 2
            ssim_val = ssim(recon_norm, img_norm, data_range=1.0).item()
            mse_pixel = mse_loss(recon_norm, img_norm).item()
            psnr = 10 * torch.log10(torch.tensor(1.0 / (mse_pixel + 1e-12))).item()
            lpips_val = loss_fn_lpips(recon, img_tensor).mean().item()

            print(f"[{i+1}/{len(files)}] {fname} | "
                  f"MSE={mse_val:.4f}, SSIM={ssim_val:.4f}, "
                  f"PSNR={psnr:.2f}, LPIPS={lpips_val:.4f}")

            originals.append(img_norm.squeeze(0).cpu())
            reconstructions.append(recon_norm.squeeze(0).cpu())

    # save grid
    pairs = []
    for orig, rec in zip(originals, reconstructions):
        pairs.append(orig)
        pairs.append(rec)

    grid = make_grid(pairs, nrow=2)
    save_path = os.path.join(custom_dir, "custom_grid.png")
    save_image(grid, save_path)
    print(f"Saved comparison grid to {save_path}")

# ------------------------------
# MODE 2: Default CIFAR10 Test Pairs
# ------------------------------
else:
    pairs = np.load('data/pairs_sd.npy', allow_pickle=True).item()
    embeddings_full = torch.from_numpy(pairs['test_embeddings'])
    images_full = torch.from_numpy(pairs['test_images'])

    total_mse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    n_images = images_full.shape[0]

    # For saving a visual grid of first few reconstructions
    vis_orig = []
    vis_rec = []

    bs = args.batch_size
    with torch.no_grad():
        for start in range(0, n_images, bs):
            end = min(start + bs, n_images)
            emb_batch = embeddings_full[start:end].to(device)
            img_batch = images_full[start:end].to(device)
            recon = f_inv(emb_batch)

            # scale to [0,1] for metrics needing that
            images_norm = (img_batch + 1) / 2
            recon_norm = (recon + 1) / 2

            # Per-image MSE in original scale [-1,1]
            mse_vals = torch.mean((recon - img_batch) ** 2, dim=[1,2,3])
            total_mse += mse_vals.sum().item()

            # SSIM expects [0,1], compute per batch (library returns batch mean)
            ssim_batch = ssim(recon_norm, images_norm, data_range=1.0)
            total_ssim += ssim_batch.item() * (end - start)

            # PSNR per-image: 10 log10(1 / mse_pixel) where mse_pixel computed on [0,1]
            mse_pixel = torch.mean((recon_norm - images_norm) ** 2, dim=[1,2,3])
            psnr_vals = 10 * torch.log10(1.0 / (mse_pixel + 1e-12))
            total_psnr += psnr_vals.sum().item()

            # LPIPS per-image (returns [B,1,1,1] or [B]) depending on version
            lp = loss_fn_lpips(recon, img_batch)
            if lp.dim() > 1:
                lp = lp.view(lp.size(0), -1).mean(dim=1)
            total_lpips += lp.sum().item()

            if len(vis_orig) < 8:  # store up to first 8 for visualization
                for i in range(min(end - start, 4)):
                    vis_orig.append(images_norm[i].cpu())
                    vis_rec.append(recon_norm[i].cpu())

    avg_mse = total_mse / n_images
    avg_ssim = total_ssim / n_images
    avg_psnr = total_psnr / n_images
    avg_lpips = total_lpips / n_images

    print(f"Averages over {n_images} images -> MSE={avg_mse:.6f} SSIM={avg_ssim:.6f} PSNR={avg_psnr:.6f} LPIPS={avg_lpips:.6f}")

    if vis_orig:
        pairs_list = []
        for o, r in zip(vis_orig, vis_rec):
            pairs_list.append(o)
            pairs_list.append(r)
        grid = make_grid(pairs_list, nrow=2)
        save_image(grid, 'results/sd_comparison.png')
        print("Saved results/sd_comparison.png")