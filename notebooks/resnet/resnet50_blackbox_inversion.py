#!/usr/bin/env python3
"""
ResNet-50 Black-Box Feature Inversion Attack

This script implements the black-box feature inversion attack on ResNet-50 as described in 
the research paper "Inverting Features with Diffusion Priors".

The implementation includes three baseline models for comparison:
1. DO (Direct Output): Direct reconstruction without LDM
2. DB (Decoder-Based): Integrated LDM decoder  
3. DMB (Diffusion-based Model with Black-box): U-Net + Frozen LDM Decoder

Author: Based on research paper specifications
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os
import random
from typing import Optional, Union, List, Tuple, Dict, Any
import lpips

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration based on paper's experimental settings
CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 128,  # Paper uses batch size 128
    'epochs': 96,       # Paper uses 96 epochs
    'lr': 0.1,          # Paper uses initial learning rate 0.1
    'beta1': 0.9,       # Paper uses Adam optimizer with beta=(0.9, 0.999)
    'beta2': 0.999,
    'lambda_s': 1.0,    # Paper uses λ_s = 1 for equations 11, 12, 13
    'training_samples': 4096,  # Paper uses 4096 training images
    'testing_samples': 1024,   # Paper uses 1024 testing images
    'image_size': 224,  # ResNet-50 standard input size
    'latent_size': 64,  # LDM latent size (512/8 = 64 for 8x downsampling)
    'latent_channels': 4,  # LDM latent channels
    'results_dir': 'results_resnet50_blackbox',
    'checkpoint_dir': 'checkpoints_resnet50_blackbox'
}

# Create directories
os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

print(f"Using device: {CONFIG['device']}")
print(f"Configuration: {CONFIG}")

class ResNet50Wrapper(nn.Module):
    """
    Wrapper for ResNet-50 model to extract features
    Based on the paper's target model F₁(.)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pre-trained ResNet-50
        self.model = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer to get features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # ResNet-50 features: [B, 2048, 1, 1] -> [B, 2048]
        self.feature_dim = 2048
        
    def forward(self, x):
        # Extract features from the last layer before classification
        features = self.model(x)
        # Flatten features: [B, 2048, 1, 1] -> [B, 2048]
        features = features.view(features.size(0), -1)
        return features

class UNetInversion(nn.Module):
    """
    U-Net component of the inversion DNN F_u(.)
    Takes ResNet-50 features as input and generates latent variables for LDM
    """
    def __init__(self, input_dim=2048, latent_channels=4, latent_size=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        
        # U-Net architecture for feature to latent mapping
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.enc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.enc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.dec2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.dec1 = nn.Sequential(
            nn.Linear(2048, latent_channels * latent_size * latent_size),
            nn.Tanh()  # Output in [-1, 1] range for LDM
        )
        
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        bottleneck = self.bottleneck(e3)
        
        # Decoder path with skip connections
        d3 = self.dec3(bottleneck + e3)
        d2 = self.dec2(d3 + e2)
        d1 = self.dec1(d2 + e1)
        
        # Reshape to latent format [B, C, H, W]
        latent = d1.view(-1, self.latent_channels, self.latent_size, self.latent_size)
        
        return latent

class InversionDNN(nn.Module):
    """
    Complete inversion DNN F_θ^inv(.) as described in the paper
    Consists of U-Net F_u(.) and LDM decoder D(.)
    """
    def __init__(self, input_dim=2048, latent_channels=4, latent_size=64):
        super().__init__()
        self.unet = UNetInversion(input_dim, latent_channels, latent_size)
        
        # Load pre-trained LDM components
        # Using Stable Diffusion's VAE decoder
        self.ldm_decoder = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae"
        ).decoder
        
        # Freeze LDM decoder parameters
        for param in self.ldm_decoder.parameters():
            param.requires_grad = False
            
        self.ldm_decoder.eval()
        
    def forward(self, features):
        # U-Net generates latent variables
        latent = self.unet(features)
        
        # Scale latent to match LDM's expected range
        latent = latent * 0.18215  # LDM scaling factor
        
        # LDM decoder reconstructs the image
        with torch.no_grad():
            reconstructed = self.ldm_decoder(latent)
        
        return reconstructed

class DOInversionDNN(nn.Module):
    """
    DO (Direct Output) variant
    Directly reconstructs user input x without relying on LDM
    """
    def __init__(self, input_dim=2048, output_channels=3, output_size=224):
        super().__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.output_size = output_size
        
        # Direct reconstruction network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(4096, 8192),
            nn.LayerNorm(8192),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(8192, 16384),
            nn.LayerNorm(16384),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(16384, output_channels * output_size * output_size),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def forward(self, features):
        output = self.network(features)
        return output.view(-1, self.output_channels, self.output_size, self.output_size)

class DBInversionDNN(nn.Module):
    """
    DB (Decoder-Based) variant
    Integrates LDM decoder into the inversion DNN
    """
    def __init__(self, input_dim=2048, latent_channels=4, latent_size=64):
        super().__init__()
        self.unet = UNetInversion(input_dim, latent_channels, latent_size)
        
        # Integrated LDM decoder (trainable)
        self.ldm_decoder = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae"
        ).decoder
        
        # Make LDM decoder trainable for DB variant
        for param in self.ldm_decoder.parameters():
            param.requires_grad = True
            
    def forward(self, features):
        latent = self.unet(features)
        latent = latent * 0.18215
        reconstructed = self.ldm_decoder(latent)
        return reconstructed

class BlackBoxDataset(Dataset):
    """
    Dataset for black-box feature inversion
    Creates pairs of (input_image, resnet_features)
    """
    def __init__(self, dataset, resnet_model, transform=None, num_samples=None):
        self.dataset = dataset
        self.resnet_model = resnet_model
        self.transform = transform
        self.num_samples = num_samples if num_samples else len(dataset)
        
        # Limit dataset size
        self.indices = list(range(min(self.num_samples, len(dataset))))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        if isinstance(self.dataset, datasets.ImageFolder):
            image, _ = self.dataset[actual_idx]
        else:
            image = self.dataset[actual_idx]
            
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Extract ResNet-50 features
        with torch.no_grad():
            features = self.resnet_model(image.unsqueeze(0))
            features = features.squeeze(0)  # Remove batch dimension
            
        return image, features

def total_variation_loss(x):
    """
    Total Variation loss for smoothness
    TV = Σ |x[i,j] - x[i,j-1]| + |x[i,j] - x[i-1,j]|
    """
    batch_size = x.size(0)
    h_x = x.size(2)
    w_x = x.size(3)
    
    count_h = h_x * w_x
    count_w = h_x * w_x
    
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
    
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def reconstruction_loss(pred, target):
    """L1 reconstruction loss"""
    return F.l1_loss(pred, target)

def train_inversion_dnn(model, train_loader, val_loader, config):
    """
    Train the inversion DNN according to paper's specifications
    """
    device = config['device']
    model = model.to(device)
    
    # Optimizer: Adam with paper's hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2'])
    )
    
    # Loss functions
    recon_criterion = reconstruction_loss
    tv_criterion = total_variation_loss
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, features) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')):
            images = images.to(device)
            features = features.to(device)
            
            # Forward pass
            reconstructed = model(features)
            
            # Compute loss according to equation 11
            recon_loss = recon_criterion(reconstructed, images)
            tv_loss = tv_criterion(reconstructed)
            
            total_loss = recon_loss + config['lambda_s'] * tv_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, features in val_loader:
                images = images.to(device)
                features = features.to(device)
                
                reconstructed = model(features)
                
                recon_loss = recon_criterion(reconstructed, images)
                tv_loss = tv_criterion(reconstructed)
                
                total_loss = recon_loss + config['lambda_s'] * tv_loss
                val_loss += total_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f"{config['checkpoint_dir']}/best_model.pth")
            
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{config['results_dir']}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, config):
    """
    Evaluate the trained inversion DNN
    """
    device = config['device']
    model.eval()
    
    # Metrics
    recon_losses = []
    tv_losses = []
    lpips_scores = []
    
    # LPIPS for perceptual similarity
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    with torch.no_grad():
        for images, features in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            features = features.to(device)
            
            reconstructed = model(features)
            
            # Reconstruction loss
            recon_loss = reconstruction_loss(reconstructed, images)
            recon_losses.append(recon_loss.item())
            
            # Total variation loss
            tv_loss = total_variation_loss(reconstructed)
            tv_losses.append(tv_loss.item())
            
            # LPIPS score
            lpips_score = lpips_fn(reconstructed, images).mean()
            lpips_scores.append(lpips_score.item())
    
    # Calculate average metrics
    avg_recon_loss = np.mean(recon_losses)
    avg_tv_loss = np.mean(tv_losses)
    avg_lpips = np.mean(lpips_scores)
    
    print(f"Evaluation Results:")
    print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"Average Total Variation Loss: {avg_tv_loss:.4f}")
    print(f"Average LPIPS Score: {avg_lpips:.4f}")
    
    return {
        'recon_loss': avg_recon_loss,
        'tv_loss': avg_tv_loss,
        'lpips': avg_lpips
    }

def visualize_results(model, test_loader, config, num_samples=8):
    """
    Visualize reconstruction results
    """
    device = config['device']
    model.eval()
    
    # Get a batch of samples
    images, features = next(iter(test_loader))
    images = images[:num_samples].to(device)
    features = features[:num_samples].to(device)
    
    with torch.no_grad():
        reconstructed = model(features)
    
    # Convert to numpy for visualization
    images_np = images.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    
    # Denormalize images (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    images_np = images_np * std + mean
    reconstructed_np = reconstructed_np * std + mean
    
    # Clip to [0, 1]
    images_np = np.clip(images_np, 0, 1)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(np.transpose(images_np[i], (1, 2, 0)))
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(np.transpose(reconstructed_np[i], (1, 2, 0)))
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config['results_dir']}/reconstruction_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the ResNet-50 black-box inversion attack
    """
    print("Initializing ResNet-50 Black-Box Inversion Attack...")
    
    # Initialize ResNet-50 model (target model F₁)
    print("Loading ResNet-50 model...")
    resnet_model = ResNet50Wrapper(pretrained=True)
    resnet_model.eval()
    
    # Data transforms
    transform = T.Compose([
        T.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset (ImageNet or CIFAR-100 as fallback)
    print("Loading dataset...")
    try:
        # Try to load ImageNet first (as mentioned in paper)
        dataset = datasets.ImageFolder(root='./data/imagenet', transform=transform)
        print(f"Using ImageNet dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"ImageNet not available: {e}")
        try:
            # Fallback to CIFAR-100
            dataset = datasets.CIFAR100(root='./data', train=True, download=True)
            print(f"Using CIFAR-100 dataset with {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading CIFAR-100: {e}")
            return
    
    # Create black-box datasets
    train_dataset = BlackBoxDataset(
        dataset, resnet_model, transform, CONFIG['training_samples']
    )
    test_dataset = BlackBoxDataset(
        dataset, resnet_model, transform, CONFIG['testing_samples']
    )
    
    # Split training data for validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Initialize models for different variants
    print("Initializing inversion DNN models...")
    
    # Model variants as described in the paper
    models = {
        'DMB': InversionDNN(2048),  # Main model with U-Net + LDM
        'DO': DOInversionDNN(2048),  # Direct reconstruction without LDM
        'DB': DBInversionDNN(2048)   # Integrated LDM decoder
    }
    
    # Train and evaluate each model variant
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} model...")
        print(f"{'='*50}")
        
        # Train model
        trained_model, train_losses, val_losses = train_inversion_dnn(
            model, train_loader, val_loader, CONFIG
        )
        
        # Evaluate model
        metrics = evaluate_model(trained_model, test_loader, CONFIG)
        
        # Visualize results
        visualize_results(trained_model, test_loader, CONFIG)
        
        # Store results
        results[model_name] = {
            'model': trained_model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics
        }
        
        # Save model
        torch.save(trained_model.state_dict(), 
                  f"{config['checkpoint_dir']}/{model_name}_final.pth")
    
    # Print final comparison
    print(f"\n{'='*50}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*50}")
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Reconstruction Loss: {result['metrics']['recon_loss']:.4f}")
        print(f"  Total Variation Loss: {result['metrics']['tv_loss']:.4f}")
        print(f"  LPIPS Score: {result['metrics']['lpips']:.4f}")
    
    print(f"\nResults saved to: {CONFIG['results_dir']}")
    print(f"Checkpoints saved to: {CONFIG['checkpoint_dir']}")

if __name__ == "__main__":
    main()
