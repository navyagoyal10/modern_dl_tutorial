"""Utilities for A07."""

from __future__ import annotations

import math
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs used in training and generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def perplexity(loss_value: float) -> float:
    """Convert average NLL loss to perplexity."""
    return float(math.exp(min(loss_value, 20.0)))
