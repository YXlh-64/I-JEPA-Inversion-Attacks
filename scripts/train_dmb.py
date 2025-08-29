# scripts/train_dmb.py
# This script trains the inversion network for Diffusion Model-Based (DMB).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from torch.nn.functional import mse_loss
import argparse
from models.unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)  # Small for diffusion
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

# Load I-JEPA
processor = AutoProcessor.from_pretrained("facebook/ijepa_vith14_1k")
model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k")
model.eval()

# Load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # For speed

# Extract embedding
def extract_embedding(image):
    img_224 = image.resize((224, 224))
    inputs = processor(img_224, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)

# Dataset
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToPILImage())

images = [cifar10[i][0].resize((512, 512)) for i in range(100)]
embeddings = [extract_embedding(img) for img in images]

image_tensors = torch.stack([TF.to_tensor(img) for img in images])  # 0-1

embedding_tensors = torch.stack(embeddings)

# Save pairs
np.save('data/pairs_dmb.npy', {'images': image_tensors.cpu().numpy(), 'embeddings': embedding_tensors.cpu().numpy()})

# Dataset: emb to image
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

# Inversion model for z_T
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

# Training
for epoch in range(args.epochs):
    for emb, target_img in dataloader:
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