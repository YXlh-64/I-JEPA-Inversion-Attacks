import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from torchvision import datasets
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.cuda.empty_cache()

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
    def __init__(self, embedding_dim=1280, input_channels=32, feature_size=64, output_channels=3):
        super().__init__()
        self.projector = nn.Linear(embedding_dim, input_channels * feature_size * feature_size)
        self.unet = UNet(input_channels, output_channels, feature_size=feature_size)

    def forward(self, x):
        proj = self.projector(x)
        proj = proj.view(-1, 32, 64, 64)
        return self.unet(proj)

inv_model = InversionModel().to(device)
optimizer = optim.Adam(inv_model.parameters(), lr=args.lr)

# Training loop
for epoch in range(args.epochs):
    for emb, target_img in dataloader:
        emb, target_img = emb.to(device), target_img.to(device)
        recon = inv_model(emb)
        loss = mse_loss(recon, target_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")

torch.save(inv_model.state_dict(), 'saved_models/do_inv.pth')