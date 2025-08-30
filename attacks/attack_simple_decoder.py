# attack_do.py (revised)
import torch
import numpy as np
from torch.nn.functional import mse_loss
from torchvision.utils import save_image
from pytorch_msssim import ssim
import lpips
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
os.makedirs('results', exist_ok=True)

pairs = np.load('data/pairs_do.npy', allow_pickle=True).item()
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
f_inv.load_state_dict(torch.load('saved_models/do_inv.pth', map_location=device))
f_inv.eval()

# LPIPS model
loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

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

    print(f"MSE={mse_val:.6f}, SSIM={ssim_val:.6f}, PSNR={psnr:.6f}, LPIPS={lpips_val:.6f}")

    # Save a few examples
    # convert to [0,1]
    paired_images = torch.cat([images_norm, recon_norm], dim=3)  # N, C, H, 2W
    save_image(paired_images, 'results/do_comparison.png', nrow=1)
    print("Saved results/do_comparison.png")
