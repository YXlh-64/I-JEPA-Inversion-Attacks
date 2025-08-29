import os
from typing import Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets


class TinyImageNetDataset(Dataset):
    """Minimal Tiny ImageNet dataset loader matching the notebook.

    Expects directory structure of the official Tiny ImageNet:
      root/train/<class>/images/*.JPEG
      root/val/images/*.JPEG
    """

    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None, limit: Optional[int] = None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []

        if split == 'train':
            train_dir = os.path.join(root_dir, 'train')
            if os.path.isdir(train_dir):
                for class_dir in os.listdir(train_dir):
                    class_path = os.path.join(train_dir, class_dir, 'images')
                    if os.path.isdir(class_path):
                        for img_file in os.listdir(class_path):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.jpeg')):
                                self.samples.append(os.path.join(class_path, img_file))
        else:
            val_dir = os.path.join(root_dir, 'val', 'images')
            if os.path.isdir(val_dir):
                for img_file in os.listdir(val_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.jpeg')):
                        self.samples.append(os.path.join(val_dir, img_file))

        if limit:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_dataloader(split: str = 'train', batch_size: int = 4, num_workers: int = 2):
    """Return a DataLoader similar to the notebook setup.

    Tries TinyImageNet at ./tiny-imagenet-200, falls back to CIFAR10.
    Images are resized to 224 and normalized to [-1, 1].
    """
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Prefer Tiny ImageNet if present
    tiny_root = './tiny-imagenet-200'
    try:
        ds = TinyImageNetDataset(tiny_root, split=split, transform=tfm, limit=1000 if split == 'train' else 200)
        if len(ds) == 0:
            raise RuntimeError('Empty TinyImageNet dataset list')
    except Exception:
        # Fallback: CIFAR10
        is_train = split == 'train'
        ds = datasets.CIFAR10(root='./data', train=is_train, download=is_train, transform=tfm)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers)
    return loader

