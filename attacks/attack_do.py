import torch
import numpy as np
from torch.nn.functional import mse_loss
from torchvision.utils import save_image
from pytorch_msssim import ssim
import lpips
from models.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.cuda.empty_cache()

# Load pairs
pairs = np.load('data/pairs_do.npy', allow_pickle=True).item()
embeddings = torch.from_numpy(pairs['embeddings'])[90:].to(device)
images = torch.from_numpy(pairs['images'])[90:].to(device)

# Inversion model
class InversionModel(torch.nn.Module):
    def __init__(self, embedding_dim=1280, input_channels=32, feature_size=64, output_channels=3):
        super().__init__()
        self.projector = torch.nn.Linear(embedding_dim, input_channels * feature_size * feature_size)
        self.unet = UNet(input_channels, output_channels, feature_size=feature_size)

    def forward(self, x):
        proj = self.projector(x)
        proj = proj.view(-1, 32, 64, 64)
        return self.unet(proj)

inv_model = InversionModel().to(device)
inv_model.load_state_dict(torch.load('saved_models/do_inv.pth', map_location=device))
inv_model.eval()

# LPIPS
loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

# Evaluate
with torch.no_grad():
    recon = inv_model(embeddings)
    mse = mse_loss(recon, images)
    ssim_val = ssim(recon, images, data_range=1.0)
    mse_pixel = mse_loss(recon, images)
    psnr = 10 * torch.log10(1.0 / mse_pixel)
    lpips_val = loss_fn_lpips(recon * 2 - 1, images * 2 - 1).mean()
    print(f"Metrics: MSE={mse.item():.6f}, SSIM={ssim_val.item():.6f}, "
          f"PSNR={psnr.item():.6f}, LPIPS={lpips_val.item():.6f}")

    paired = torch.cat([images, recon], dim=3)
    save_image(paired, 'results/do_comparison.png', nrow=2)
    print("Saved to results/do_comparison.png")
