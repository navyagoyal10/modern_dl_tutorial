"""Utilities for A04."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
