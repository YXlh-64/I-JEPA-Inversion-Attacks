import torch
import numpy as np
from torch.nn.functional import mse_loss
from diffusers import StableDiffusionPipeline, DDIMScheduler
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
pairs = np.load('data/pairs_dmb.npy', allow_pickle=True).item()
embeddings = torch.from_numpy(pairs['embeddings'])[90:].to(device)
images = torch.from_numpy(pairs['images'])[90:].to(device)

# Load pipe
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

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

inv_model = InversionModel().to(device)
inv_model.load_state_dict(torch.load('saved_models/dmb_inv.pth', map_location=device))
inv_model.eval()

# Initialize LPIPS
loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

# Invert
with torch.no_grad():
    pred_z_t = inv_model(embeddings)
    pipe_output = pipe(prompt=[""] * len(embeddings), latents=pred_z_t, num_inference_steps=50, output_type="pt")
    recon = pipe_output.images
    # Compute metrics
    mse = mse_loss(recon, images)
    # Normalize to [0,1] for SSIM and PSNR
    images_norm = images
    recon_norm = recon
    ssim_val = ssim(recon_norm, images_norm, data_range=1.0)
    mse_pixel = mse_loss(recon_norm, images_norm)
    psnr = 10 * torch.log10(1.0 / mse_pixel)
    # LPIPS expects [-1,1]
    lpips_val = loss_fn_lpips(recon * 2 - 1, images * 2 - 1).mean()
    print(f"Test Metrics: MSE={mse.item():.6f}, SSIM={ssim_val.item():.6f}, PSNR={psnr.item():.6f}, LPIPS={lpips_val.item():.6f}")

    # Save original vs reconstructed images
    paired_images = torch.cat([images_norm, recon_norm], dim=3)
    save_image(paired_images, 'results/dmb_comparison.png', nrow=2)
    print("Saved original vs reconstructed images to results/dmb_comparison.png")