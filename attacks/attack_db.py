import torch
import numpy as np
from torch.nn.functional import mse_loss
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from pytorch_msssim import ssim
import lpips
from ..models.unet import UNet

# Load pairs
pairs = np.load('data/pairs_db.npy', allow_pickle=True).item()
embeddings = torch.from_numpy(pairs['embeddings'])[90:]
images = torch.from_numpy(pairs['images'])[90:]

# Load VAE
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
vae.eval()

# Inversion model
class InversionModel(torch.nn.Module):
    def __init__(self, embedding_dim=1280, input_channels=32, feature_size=64, output_channels=4):
        super().__init__()
        self.projector = torch.nn.Linear(embedding_dim, input_channels * feature_size * feature_size)
        self.unet = UNet(input_channels, output_channels, feature_size=feature_size)

    def forward(self, x):
        proj = self.projector(x)
        proj = proj.view(-1, 32, 64, 64)
        return self.unet(proj)

inv_model = InversionModel()
inv_model.load_state_dict(torch.load('saved_models/db_inv.pth'))
inv_model.eval()

# Initialize LPIPS
loss_fn_lpips = lpips.LPIPS(net='vgg')

# Invert
with torch.no_grad():
    pred_latent = inv_model(embeddings)
    recon = vae.decode(pred_latent / vae.config.scaling_factor).sample
    # Compute metrics
    mse = mse_loss(recon, images * 2 - 1)
    # Normalize to [0,1] for SSIM and PSNR
    images_norm = images
    recon_norm = (recon + 1) / 2
    ssim_val = ssim(recon_norm, images_norm, data_range=1.0)
    mse_pixel = mse_loss(recon_norm, images_norm)
    psnr = 10 * torch.log10(1.0 / mse_pixel)
    # LPIPS expects [-1,1]
    lpips_val = loss_fn_lpips(recon, images * 2 - 1).mean()
    print(f"Test Metrics: MSE={mse.item():.6f}, SSIM={ssim_val.item():.6f}, PSNR={psnr.item():.6f}, LPIPS={lpips_val.item():.6f}")

    # Save original vs reconstructed images
    paired_images = torch.cat([images_norm, recon_norm], dim=3)
    save_image(paired_images, 'results/db_comparison.png', nrow=2)
    print("Saved original vs reconstructed images to results/db_comparison.png")