"""Prepare full CIFAR-10 embeddings + VAE latents.

Changes vs old script:
 - Uses all 50k CIFAR-10 training images: first 48k for train, last 2k as test (deterministic split).
 - Batched extraction for I-JEPA embeddings and Stable Diffusion VAE latents (memory efficient).
 - Memory-mapped storage (.mmap) for images, embeddings, latents to avoid loading all into RAM.
 - Atomic finalization: build temp directory then rename.
 - CLI arguments for batch sizes, output dir, precision, etc.
 - Dataset class loads from memmap on demand.
"""

import os
import argparse
import numpy as np
import torch
from torchvision import datasets
from transformers import AutoProcessor, AutoModel
from diffusers import AutoencoderKL
from PIL import Image

# ---------------------------------------------------------------------------
# Lightweight tensor-backed dataset used by training scripts (DO / DB / DMB).
# The scripts expect: from prepare_dataset import InversionDataset
# It simply wraps two tensors (embeddings, targets) and returns pairs.
# targets can be images or latents depending on the training variant.
# ---------------------------------------------------------------------------
class InversionDataset(torch.utils.data.Dataset):
    """Simple dataset for (embedding, target) pairs.

    Parameters
    ----------
    embeddings : torch.Tensor (N, D)
        Feature embeddings (e.g. IJepa pooled representations).
    targets : torch.Tensor
        Corresponding target tensors (e.g. images (N,3,H,W) or latents (N,C,H,W)).
    dtype : torch.dtype, optional
        If provided, casts embeddings & targets to this dtype (default: keep as is except ensure float32 for floating types).
    """
    def __init__(self, embeddings: torch.Tensor, targets: torch.Tensor, dtype: torch.dtype | None = None):
        if len(embeddings) != len(targets):
            raise ValueError(f"Mismatched lengths: {len(embeddings)} vs {len(targets)}")
        # Avoid unnecessary copies; cast if requested.
        if dtype is not None:
            embeddings = embeddings.to(dtype)
            # Only cast targets if floating; keep integer types (e.g. uint8 images) intact.
            if torch.is_floating_point(targets):
                targets = targets.to(dtype)
        else:
            # Ensure float32 for common floating inputs to match model expectations.
            if torch.is_floating_point(embeddings) and embeddings.dtype != torch.float32:
                embeddings = embeddings.float()
            if torch.is_floating_point(targets) and targets.dtype != torch.float32:
                targets = targets.float()
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.targets[idx]

__all__ = ["InversionDataset"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ijepa-name', type=str, default='facebook/ijepa_vith14_1k')
    p.add_argument('--vae-name', type=str, default='stabilityai/stable-diffusion-2-1')
    p.add_argument('--vae-subfolder', type=str, default='vae')
    p.add_argument('--resize-ijepa', type=int, default=224)
    p.add_argument('--resize-vae', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=128, help='Batch size for embedding + latent extraction pipeline')
    p.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16'])
    p.add_argument('--out-dir', type=str, default='prepared_cifar')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def prepare_memmaps(out_dir, n_train, n_test, emb_dim, latent_shape):
    os.makedirs(out_dir, exist_ok=True)
    # Shapes
    img_shape = (3, 512, 512)
    train = {
        'images': np.memmap(os.path.join(out_dir, 'train_images.mmap'), dtype='uint8', mode='w+', shape=(n_train, *img_shape)),
        'embeddings': np.memmap(os.path.join(out_dir, 'train_embeddings.mmap'), dtype='float32', mode='w+', shape=(n_train, emb_dim)),
        'latents': np.memmap(os.path.join(out_dir, 'train_latents.mmap'), dtype='float32', mode='w+', shape=(n_train, *latent_shape)),
    }
    test = {
        'images': np.memmap(os.path.join(out_dir, 'test_images.mmap'), dtype='uint8', mode='w+', shape=(n_test, *img_shape)),
        'embeddings': np.memmap(os.path.join(out_dir, 'test_embeddings.mmap'), dtype='float32', mode='w+', shape=(n_test, emb_dim)),
        'latents': np.memmap(os.path.join(out_dir, 'test_latents.mmap'), dtype='float32', mode='w+', shape=(n_test, *latent_shape)),
    }
    return train, test


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    print('Loading CIFAR-10 (train split only)...')
    cifar = datasets.CIFAR10(root='./data', train=True, download=True)
    total = len(cifar.data)  # 50000
    assert total == 50000, 'Unexpected CIFAR-10 size'
    train_indices = list(range(0, 48000))
    test_indices = list(range(48000, 50000))

    # Load models
    print('Loading I-JEPA backbone...')
    processor = AutoProcessor.from_pretrained(args.ijepa_name)
    ijepa = AutoModel.from_pretrained(args.ijepa_name).to(device)
    ijepa.eval()

    print('Loading SD VAE...')
    vae = AutoencoderKL.from_pretrained(args.vae_name, subfolder=args.vae_subfolder).to(device)
    vae.eval()

    # Determine embedding dimension
    with torch.no_grad():
        dummy = processor(Image.fromarray(cifar.data[0]).resize((args.resize_ijepa, args.resize_ijepa)), return_tensors='pt')
        dummy = {k: v.to(device) for k, v in dummy.items()}
        emb_dim = ijepa(**dummy).last_hidden_state.shape[-1]
    print(f'Embedding dim: {emb_dim}')

    # Determine latent shape
    with torch.no_grad():
        from torchvision.transforms.functional import to_tensor
        pil = Image.fromarray(cifar.data[0]).resize((args.resize_vae, args.resize_vae))
        t = to_tensor(pil).unsqueeze(0).to(device) * 2 - 1
        lat = vae.encode(t).latent_dist.sample() * vae.config.scaling_factor
        latent_shape = tuple(lat.shape[1:])
    print(f'Latent shape: {latent_shape}')

    # Prepare memmaps
    temp_dir = args.out_dir + '_tmp'
    if os.path.exists(temp_dir):
        print('Temporary directory exists, removing old files (partial previous run).')
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
    os.makedirs(temp_dir, exist_ok=True)
    train_maps, test_maps = prepare_memmaps(temp_dir, len(train_indices), len(test_indices), emb_dim, latent_shape)

    # Extraction function (batched)
    def process_indices(indices, maps, split_name):
        from torchvision.transforms.functional import to_tensor
        bs = args.batch_size
        n = len(indices)
        print(f'Processing {split_name}: {n} images')
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch_indices = indices[start:end]
            pil_batch_vae = [Image.fromarray(cifar.data[i]).resize((args.resize_vae, args.resize_vae)) for i in batch_indices]
            pil_batch_emb = [img.resize((args.resize_ijepa, args.resize_ijepa)) for img in pil_batch_vae]

            # Prepare processor inputs for embeddings
            inputs = processor(pil_batch_emb, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = ijepa(**inputs).last_hidden_state.mean(dim=1)  # [B, emb_dim]

            # Images tensor for VAE (scaled to [-1,1])
            imgs_tensor = torch.stack([to_tensor(p) for p in pil_batch_vae]).to(device) * 2 - 1
            with torch.no_grad():
                latents = vae.encode(imgs_tensor).latent_dist.sample() * vae.config.scaling_factor

            # Store (convert images to uint8 [0,255])
            imgs_uint8 = ( (imgs_tensor * 0.5 + 0.5).clamp(0,1) * 255 ).byte().cpu().numpy()
            feats_np = feats.cpu().numpy()
            lat_np = latents.cpu().numpy()

            maps['images'][start:end] = imgs_uint8
            maps['embeddings'][start:end] = feats_np
            maps['latents'][start:end] = lat_np

            if (start // bs) % 20 == 0:
                print(f'  [{end}/{n}] done')

    process_indices(train_indices, train_maps, 'train')
    process_indices(test_indices, test_maps, 'test')

    # Flush memmaps
    for d in (train_maps, test_maps):
        for mm in d.values():
            mm.flush()

    # Write meta file
    meta = {
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'embedding_dim': emb_dim,
        'latent_shape': latent_shape,
        'image_shape': (3, 512, 512),
    'ijepa_model': args.ijepa_name,
    'vae_model': args.vae_name,
        'split': {'train': [0, 48000], 'test': [48000, 50000]},
    }
    import json
    with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Atomic rename
    if os.path.exists(args.out_dir):
        print('Removing previous output directory (backup not kept).')
        for f in os.listdir(args.out_dir):
            os.remove(os.path.join(args.out_dir, f))
        os.rmdir(args.out_dir)
    os.rename(temp_dir, args.out_dir)
    print(f'Finished. Data stored under {args.out_dir}')

    # Provide dataset class snippet
    print('\nUse the following dataset class to load memory-mapped data:')
    print('''\nclass MemmapInversionDataset(torch.utils.data.Dataset):\n    def __init__(self, root, split="train"):\n        import json, os, numpy as np\n        with open(os.path.join(root, 'meta.json')) as f: meta = json.load(f)\n        self.split = split\n        if split == 'train': size = meta['train_size']\n        else: size = meta['test_size']\n        self.emb = np.memmap(os.path.join(root, f"{split}_embeddings.mmap"), mode='r', dtype='float32', shape=(size, meta['embedding_dim']))\n        latent_shape = tuple(meta['latent_shape'])\n        self.lat = np.memmap(os.path.join(root, f"{split}_latents.mmap"), mode='r', dtype='float32', shape=(size, *latent_shape))\n    def __len__(self): return self.emb.shape[0]\n    def __getitem__(self, idx):\n        return torch.from_numpy(self.emb[idx]), torch.from_numpy(self.lat[idx])\n''')


if __name__ == '__main__':
    main()