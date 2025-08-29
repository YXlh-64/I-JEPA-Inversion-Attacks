# scripts/train_do.py
# This script trains the inversion network for Direct Optimization (DO).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from torch.nn.functional import mse_loss
import argparse
from models.unet import UNet  # Assume models is in PYTHONPATH or adjust import

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

# Load I-JEPA model and processor
processor = AutoProcessor.from_pretrained("facebook/ijepa_vith14_1k")
model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k")
model.eval()

# Function to extract context embedding
def extract_embedding(image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 1280]
    return embedding.squeeze(0)

# Prepare dataset: use CIFAR10 resized to 224x224
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToPILImage())

# For demo, small subset
images = [cifar10[i][0].resize((224, 224)) for i in range(100)]
embeddings = [extract_embedding(img) for img in images]

image_tensors = torch.stack([TF.to_tensor(img) * 2 - 1 for img in images])  # To -1 to 1
embedding_tensors = torch.stack(embeddings)  # [100, 1280]

# Save pairs
np.save('data/pairs_do.npy', {'images': image_tensors.cpu().numpy(), 'embeddings': embedding_tensors.cpu().numpy()})

# Dataset
class InversionDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, images):
        self.embeddings = embeddings
        self.images = images

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.images[idx]

dataset = InversionDataset(embedding_tensors, image_tensors)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Inversion model with projector + UNet
class InversionModel(nn.Module):
    def __init__(self, embedding_dim=1280, input_channels=32, feature_size=224, output_channels=3):
        super().__init__()
        self.projector = nn.Linear(embedding_dim, input_channels * feature_size * feature_size)
        self.unet = UNet(input_channels, output_channels, feature_size=feature_size)

    def forward(self, x):
        proj = self.projector(x)
        proj = proj.view(-1, 32, 224, 224)
        return self.unet(proj)

f_inv = InversionModel()
optimizer = optim.Adam(f_inv.parameters(), lr=args.lr)

# TV loss
def tv_loss(img):
    return torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]) + torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))

# Training loop
for epoch in range(args.epochs):
    for emb, img in dataloader:
        recon = f_inv(emb)
        loss_re = mse_loss(recon, img)
        loss_tv = tv_loss(recon)
        loss = loss_re + 0.1 * loss_tv
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")

# Save model
torch.save(f_inv.state_dict(), 'saved_models/do_inv.pth')