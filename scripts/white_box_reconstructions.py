# train_do.py (revised)
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
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

device = args.device
print(f"Using device: {device}")


# --- STEP 1: extract embeddings on CPU/GPU then free model ---
processor = AutoProcessor.from_pretrained("facebook/ijepa_vith14_1k")
ijepa = AutoModel.from_pretrained("facebook/ijepa_vith14_1k")
# prefer extracting on CPU if GPU memory is tight:
ijepa_device = 'cpu' if device == 'cpu' else device
ijepa.to(ijepa_device).eval()

# Prepare dataset: CIFAR10 resized to 224
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Resize((224,224)))
# small subset for demo
N = 100
images_pil = [cifar10[i][0].convert('RGB') for i in range(N)]

# helper to extract embeddings (do in batches to avoid keeping model + big tensors)
embeddings = []
with torch.no_grad():
    for img in images_pil:
        inputs = processor(img, return_tensors="pt")
        inputs = {k: v.to(ijepa_device) for k, v in inputs.items()}
        outputs = ijepa(**inputs)
        context_mask = inputs["bool_masked_pos"] == 0
        hidden = outputs.last_hidden_state  # [1, num_patches, dim]
        emb = hidden[context_mask].mean(dim=0).cpu()  # average context patches
        embeddings.append(emb)

# free ijepa
del ijepa
del processor
torch.cuda.empty_cache()

# convert images to tensors (keep on CPU)
image_tensors_cpu = torch.stack([TF.to_tensor(img) * 2 - 1 for img in images_pil])  # [-1,1], CPU
embedding_tensors_cpu = torch.stack(embeddings)  # CPU

np.save('data/pairs_do.npy', {'images': image_tensors_cpu.numpy(), 'embeddings': embedding_tensors_cpu.numpy()})
print("Saved pairs to data/pairs_do.npy")

# --- Dataset / Dataloader (keep tensors on CPU, move per-batch to GPU) ---
class InversionDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, images):
        self.embeddings = embeddings
        self.images = images
    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.images[idx]

dataset = InversionDataset(embedding_tensors_cpu, image_tensors_cpu)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

# --- Compact inversion model: projector -> small spatial map -> conv decoder to 224x224 ---
class SmallDecoder(nn.Module):
    def __init__(self, in_ch=128, out_ch=3):
        super().__init__()
        # upsample from 16x16 -> 32 -> 64 -> 128 -> 224 (approx)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # 32->64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),      # 64->128
            nn.ReLU(inplace=True),
            nn.Upsample(size=(224,224), mode='bilinear', align_corners=False),
            nn.Conv2d(32, out_ch, kernel_size=3, padding=1),
            nn.Tanh()  # output in [-1,1]
        )
    def forward(self, x):
        return self.net(x)

class InversionModel(nn.Module):
    def __init__(self, embedding_dim=1280, proj_C=128, proj_H=16, proj_W=16):
        super().__init__()
        self.proj_C = proj_C
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.projector = nn.Linear(embedding_dim, proj_C * proj_H * proj_W)
        self.decoder = SmallDecoder(in_ch=proj_C, out_ch=3)
    def forward(self, emb):
        # emb: [B, 1280]
        x = self.projector(emb)                        # [B, C*H*W]
        x = x.view(-1, self.proj_C, self.proj_H, self.proj_W)
        return self.decoder(x)

f_inv = InversionModel().to(device)

optimizer = optim.Adam(f_inv.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler(enabled=(device!='cpu'))

# TV loss
def tv_loss(img):
    # img shape: [batch_size, channels, height, width]
    h_diff = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])  # Shape: [batch_size, channels, height-1, width]
    w_diff = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])  # Shape: [batch_size, channels, height, width-1]
    # Crop h_diff to match w_diff's height
    h_diff = h_diff[:, :, :, :-1]  # Shape: [batch_size, channels, height-1, width-1]
    w_diff = w_diff[:, :, :-1, :]  # Shape: [batch_size, channels, height-1, width-1]
    return torch.mean(h_diff + w_diff)

# Training loop
for epoch in range(args.epochs):
    f_inv.train()
    running_loss = 0.0
    for emb_cpu, img_cpu in dataloader:
        # move batch to device (CPU->GPU): use non-blocking when pin_memory True
        emb = emb_cpu.to(device, non_blocking=True)
        img = img_cpu.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device!='cpu')):
            recon = f_inv(emb)
            loss_re = mse_loss(recon, img)
            loss_tv = tv_loss(recon)
            loss = loss_re + 0.1 * loss_tv
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    avg = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg:.6f}")
    torch.cuda.empty_cache()

# Save model
torch.save(f_inv.state_dict(), 'saved_models/do_inv.pth')
print("Saved model to saved_models/do_inv.pth")
