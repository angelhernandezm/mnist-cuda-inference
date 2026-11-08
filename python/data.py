"""MNIST loaders (torchvision downloads the dataset into ../data on first use)."""

import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import MEAN, STD

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])


def train_val_loaders(batch_size: int, seed: int = 42, val_fraction: float = 0.1):
    full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=TRANSFORM)
    val_size = int(len(full) * val_fraction)
    train_ds, val_ds = random_split(full, [len(full) - val_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2))


def test_dataset():
    return datasets.MNIST(DATA_DIR, train=False, download=True, transform=TRANSFORM)


def test_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """The whole 10k test set as two tensors: images (N,1,28,28) and labels (N,)."""
    ds = test_dataset()
    images = (ds.data.unsqueeze(1).float() / 255.0 - MEAN) / STD
    return images, ds.targets.clone()
