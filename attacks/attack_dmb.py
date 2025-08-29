import torch
import numpy as np
from torch.nn.functional import mse_loss
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision.utils import save_image
from ..models.unet import UNet

# Load pairs
pairs = np.load('data/pairs_dmb.npy', allow_pickle=True).item()
embeddings = torch.from_numpy(pairs['embeddings'])[90:]
images = torch.from_numpy(pairs['images'])[90:]

# Load pipe
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
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

inv_model = InversionModel()
inv_model.load_state_dict(torch.load('saved_models/dmb_inv.pth'))
inv_model.eval()

# Invert
with torch.no_grad():
    pred_z_t = inv_model(embeddings)
    pipe_output = pipe(prompt=[""] * len(embeddings), latents=pred_z_t, num_inference_steps=50, output_type="pt")
    recon = pipe_output.images
    loss = mse_loss(recon, images)
print(f"Test MSE Loss: {loss.item()}")

# Save original vs reconstructed images
images_norm = images  # Already [0,1]
recon_norm = recon  # Already [0,1] from pipeline
paired_images = torch.cat([images_norm, recon_norm], dim=3)
save_image(paired_images, 'results/dmb_comparison.png', nrow=2)
print("Saved original vs reconstructed images to results/dmb_comparison.png")