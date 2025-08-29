# scripts/train_db.py
# This script trains the inversion network for Decoder-Based (DB).

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
from models.unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

# Load I-JEPA
processor = AutoProcessor.from_pretrained("facebook/ijepa_vith14_1k")
model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k")
model.eval()

# Load SD VAE
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
vae.eval()

# Extract embedding
def extract_embedding(image):
    # Resize to 224 for I-JEPA
    img_224 = image.resize((224, 224))
    inputs = processor(img_224, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)

# Dataset: CIFAR resized to 512
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToPILImage())

images = [cifar10[i][0].resize((512, 512)) for i in range(100)]
embeddings = [extract_embedding(img) for img in images]

image_tensors = torch.stack([TF.to_tensor(img) for img in images])  # [100, 3, 512, 512]

# Precompute latents
with torch.no_grad():
    latents = vae.encode(image_tensors * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

embedding_tensors = torch.stack(embeddings)

# Save pairs (images at 512, embeddings from 224)
np.save('data/pairs_db.npy', {'images': image_tensors.cpu().numpy(), 'embeddings': embedding_tensors.cpu().numpy()})

# Dataset: emb to latent
class InversionDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, latents):
        self.embeddings = embeddings
        self.latents = latents

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.latents[idx]

dataset = InversionDataset(embedding_tensors, latents)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Inversion model
class InversionModel(nn.Module):
    def __init__(self, embedding_dim=1280, input_channels=32, feature_size=64, output_channels=4):
        super().__init__()
        self.projector = nn.Linear(embedding_dim, input_channels * feature_size * feature_size)
        self.unet = UNet(input_channels, output_channels, feature_size=feature_size)

    def forward(self, x):
        proj = self.projector(x)
        proj = proj.view(-1, 32, 64, 64)
        return self.unet(proj)

inv_model = InversionModel()
optimizer = optim.Adam(inv_model.parameters(), lr=args.lr)

# Training: loss on decoded images
for epoch in range(args.epochs):
    for emb, target_latent in dataloader:
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