
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from prepare_dataset import InversionDataset
import torchvision.transforms.functional as TF
from torch.nn.functional import mse_loss
import argparse
from models.unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=2)  # Reduced for Kaggle GPU
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Clear GPU memory
if device == "cuda":
    torch.cuda.empty_cache()

# Load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# load the training data
train_data = np.load("data/train_pairs.npy", allow_pickle=True).item()
dataset = InversionDataset(
    torch.tensor(train_data["embeddings"], dtype=torch.float32),
    torch.tensor(train_data["latents"], dtype=torch.float32),
)
print(f"Train dataset size: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Inversion model for z_T
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

# Training
for epoch in range(args.epochs):
    for emb, target_img in dataloader:
        emb, target_img = emb.to(device), target_img.to(device)
        pred_z_t = inv_model(emb)
        # Run diffusion
        with torch.no_grad():
            pipe_output = pipe(prompt=[""] * len(emb), latents=pred_z_t, num_inference_steps=50, output_type="pt")
            recon_tensors = pipe_output.images  # tensor
        loss = mse_loss(recon_tensors, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")

torch.save(inv_model.state_dict(), 'saved_models/dmb_inv.pth')
