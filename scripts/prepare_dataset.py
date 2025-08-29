import os
import torch
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, AutoModel
from diffusers import AutoencoderKL

# -------------------------------
# Device setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    torch.cuda.empty_cache()

# -------------------------------
# Load I-JEPA
# -------------------------------
print("Loading I-JEPA...")
processor = AutoProcessor.from_pretrained("facebook/ijepa_vith14_1k")
ijepa = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").to(device)
ijepa.eval()

# -------------------------------
# Load Stable Diffusion VAE
# -------------------------------
print("Loading Stable Diffusion VAE...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="vae"
).to(device)
vae.eval()

# -------------------------------
# Embedding extraction function
# -------------------------------


# extract the embeddings of the context encoder only!
def extract_embedding(image, mask_ratio=0.6):
    """Extract I-JEPA context embedding from visible patches only."""
    img_224 = image.resize((224, 224))
    inputs = processor(img_224, return_tensors="pt").to(device)
    
    # Apply masking to simulate I-JEPA's context encoder
    # (You'd need to implement the masking strategy I-JEPA uses)
    with torch.no_grad():
        # Get embeddings only from visible (context) patches
        context_outputs = ijepa.get_context_embedding(inputs, mask_ratio=mask_ratio)
    return context_outputs

# -------------------------------
# Load CIFAR-10 dataset
# -------------------------------
print("Loading CIFAR-10...")
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True)

# Resize to 512 for VAE
train_images = [cifar_train[i][0].resize((512, 512)) for i in range(len(cifar_train))]
test_images  = [cifar_test[i][0].resize((512, 512)) for i in range(len(cifar_test))]

# -------------------------------
# Compute embeddings & tensors
# -------------------------------
print("Extracting embeddings... (train)")
train_embeddings = [extract_embedding(img).cpu() for img in train_images]
train_tensors = torch.stack([TF.to_tensor(img) for img in train_images]).to(device)

print("Extracting embeddings... (test)")
test_embeddings = [extract_embedding(img).cpu() for img in test_images]
test_tensors = torch.stack([TF.to_tensor(img) for img in test_images]).to(device)

# -------------------------------
# Compute latents via VAE
# -------------------------------
def encode_latents(images):
    with torch.no_grad():
        return vae.encode(images * 2 - 1).latent_dist.sample() * vae.config.scaling_factor

print("Encoding latents...")
train_latents = encode_latents(train_tensors)
test_latents  = encode_latents(test_tensors)

train_embeddings = torch.stack(train_embeddings)
test_embeddings  = torch.stack(test_embeddings)

# -------------------------------
# Save precomputed dataset
# -------------------------------
os.makedirs("data", exist_ok=True)

np.save("data/train_pairs.npy", {
    "images": train_tensors.cpu().numpy(),
    "embeddings": train_embeddings.cpu().numpy(),
    "latents": train_latents.cpu().numpy(),
})
np.save("data/test_pairs.npy", {
    "images": test_tensors.cpu().numpy(),
    "embeddings": test_embeddings.cpu().numpy(),
    "latents": test_latents.cpu().numpy(),
})

print("Saved: data/train_pairs.npy & data/test_pairs.npy")




# -------------------------------
# Dataset class for later training
# -------------------------------
class InversionDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, latents):
        self.embeddings = embeddings
        self.latents = latents

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.latents[idx]