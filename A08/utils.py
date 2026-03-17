"""Utilities for A08."""

from __future__ import annotations

import random

import numpy as np
import torch
from torchvision import transforms


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(train: bool) -> transforms.Compose:
    """Create 224x224 ImageNet-normalized transforms."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
