import torch
import numpy as np
from torch.nn.functional import mse_loss
from torchvision.utils import save_image
from pytorch_msssim import ssim
import lpips
from models.unet import UNet

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Clear GPU memory
if device == "cuda":
    torch.cuda.empty_cache()

# Load pairs
pairs = np.load('data/pairs_do.npy', allow_pickle=True).item()
embeddings = torch.from_numpy(pairs['embeddings'])[90:].to(device)  # Test set
images = torch.from_numpy(pairs['images'])[90:].to(device)

# Inversion model
class InversionModel(torch.nn.Module):
    def __init__(self, embedding_dim=1280, input_channels=32, feature_size=224, output_channels=3):
        super().__init__()
        self.projector = torch.nn.Linear(embedding_dim, input_channels * feature_size * feature_size)
        self.unet = UNet(input_channels, output_channels, feature_size=feature_size)

    def forward(self, x):
        proj = self.projector(x)
        proj = proj.view(-1, 32, 224, 224)
        return self.unet(proj)

f_inv = InversionModel().to(device)
f_inv.load_state_dict(torch.load('saved_models/do_inv.pth', map_location=device))
f_inv.eval()

# Initialize LPIPS
loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

# Invert
with torch.no_grad():
    recon = f_inv(embeddings)
    # Compute metrics
    mse = mse_loss(recon, images)
    # Normalize to [0,1] for SSIM and PSNR
    images_norm = (images + 1) / 2
    recon_norm = (recon + 1) / 2
    ssim_val = ssim(recon_norm, images_norm, data_range=1.0)
    # PSNR: 10 * log10(MAX^2 / MSE)
    mse_pixel = mse_loss(recon_norm, images_norm)
    psnr = 10 * torch.log10(1.0 / mse_pixel)
    # LPIPS expects [-1,1]
    lpips_val = loss_fn_lpips(recon, images).mean()
    print(f"Test Metrics: MSE={mse.item():.6f}, SSIM={ssim_val.item():.6f}, PSNR={psnr.item():.6f}, LPIPS={lpips_val.item():.6f}")

    # Save original vs reconstructed images
    paired_images = torch.cat([images_norm, recon_norm], dim=3)  # Stack horizontally
    save_image(paired_images, 'results/do_comparison.png', nrow=2)
    print("Saved original vs reconstructed images to results/do_comparison.png")
