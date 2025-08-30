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
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--custom", action="store_true", default=False,
                    help="If set, run on custom images instead of CIFAR pairs.")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
os.makedirs('results', exist_ok=True)

pairs = np.load('data/pairs_sd.npy', allow_pickle=True).item()
embeddings = torch.from_numpy(pairs['embeddings'])[90:]  # still on CPU
images = torch.from_numpy(pairs['images'])[90:]

# Recreate model architecture exactly as training (proj_C=128, 16x16)
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

f_inv = InversionModel().to(device)
f_inv.load_state_dict(torch.load('saved_models/sd_inv.pth', map_location=device))
f_inv.eval()

# LPIPS model
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
    custom_dir = "custom_data"
    os.makedirs(custom_dir, exist_ok=True)

    # HuggingFace I-JEPA
    processor = AutoProcessor.from_pretrained("facebook/ijepa-vith14")
    ijepa = AutoModel.from_pretrained("facebook/ijepa-vith14").to(device).eval()

    files = [f for f in os.listdir(custom_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]

    originals = []
    reconstructions = []

    for i, fname in enumerate(files):
        path = os.path.join(custom_dir, fname)
        img = Image.open(path).convert("RGB").resize((224,224))

        # Normalize to [-1,1] for consistency with training
        img_tensor = to_tensor(img).unsqueeze(0).to(device) * 2 - 1  

        with torch.no_grad():
            # Extract embedding
            inputs = processor(img, return_tensors="pt").to(device)
            outputs = ijepa(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)  # [1,1280]

            # Reconstruct
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

            # Collect for grid
            originals.append(img_norm.squeeze(0).cpu())
            reconstructions.append(recon_norm.squeeze(0).cpu())

    # --- Save as grid: row = original vs reconstruction ---
    pairs = []
    for orig, rec in zip(originals, reconstructions):
        pairs.append(orig)
        pairs.append(rec)

    grid = make_grid(pairs, nrow=2)  # each row: orig | recon
    save_path = os.path.join(custom_dir, "custom_grid.png")
    save_image(grid, save_path)
    print(f"Saved comparison grid to {save_path}")

# ------------------------------
# MODE 2: Default (pairs_sd.npy)
# ------------------------------
else:
    os.makedirs('results', exist_ok=True)

    pairs = np.load('data/pairs_sd.npy', allow_pickle=True).item()
    embeddings = torch.from_numpy(pairs['embeddings'])[90:]
    images = torch.from_numpy(pairs['images'])[90:]

    with torch.no_grad():
        emb_batch = embeddings.to(device)
        img_batch = images.to(device)
        recon = f_inv(emb_batch)

        mse_val = mse_loss(recon, img_batch).item()
        images_norm = (img_batch + 1) / 2
        recon_norm = (recon + 1) / 2
        ssim_val = ssim(recon_norm, images_norm, data_range=1.0).item()
        mse_pixel = mse_loss(recon_norm, images_norm).item()
        psnr = 10 * torch.log10(torch.tensor(1.0 / (mse_pixel + 1e-12))).item()
        lpips_val = loss_fn_lpips(recon, img_batch).mean().item()

        print(f"MSE={mse_val:.6f}, SSIM={ssim_val:.6f}, "
              f"PSNR={psnr:.6f}, LPIPS={lpips_val:.6f}")

        paired_images = torch.cat([images_norm, recon_norm], dim=3)
        save_image(paired_images, 'results/sd_comparison.png', nrow=1)
        print("Saved results/sd_comparison.png")