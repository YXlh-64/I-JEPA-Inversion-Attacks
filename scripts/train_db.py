
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from diffusers import AutoencoderKL
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from torch.nn.functional import mse_loss
import argparse
from prepare_dataset import InversionDataset
from models.unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Clear GPU memory
if device == "cuda":
    torch.cuda.empty_cache()

# Load SD VAE
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(device)
vae.eval()

# load the training data
train_data = np.load("data/train_pairs.npy", allow_pickle=True).item()
dataset = InversionDataset(
    torch.tensor(train_data["embeddings"], dtype=torch.float32),
    torch.tensor(train_data["latents"], dtype=torch.float32),
)
print(f"Train dataset size: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
# Inversion model
class InversionModel(nn.Module):
    def __init__(self, embedding_dim=1280, proj_C=128, proj_H=16, proj_W=16, output_channels=4):
        super().__init__()
        self.proj_C, self.proj_H, self.proj_W = proj_C, proj_H, proj_W
        self.projector = nn.Linear(embedding_dim, proj_C * proj_H * proj_W)
        self.unet = UNet(input_channels=proj_C, output_channels=output_channels, feature_size=proj_H)

    def forward(self, x):
        proj = self.projector(x)
        proj = proj.view(-1, self.proj_C, self.proj_H, self.proj_W)
        return self.unet(proj)

inv_model = InversionModel(proj_C=128, proj_H=16, proj_W=16).to(device)

optimizer = optim.Adam(inv_model.parameters(), lr=args.lr)

# Training: loss on decoded images
for epoch in range(args.epochs):
    for emb, target_latent in dataloader:
        emb, target_latent = emb.to(device), target_latent.to(device)
        pred_latent = inv_model(emb)
        with torch.no_grad():
            recon = vae.decode(pred_latent / vae.config.scaling_factor).sample
            target_img = vae.decode(target_latent / vae.config.scaling_factor).sample
        loss = mse_loss(recon, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")

torch.save(inv_model.state_dict(), 'saved_models/db_inv.pth')