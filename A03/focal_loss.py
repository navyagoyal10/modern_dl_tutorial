"""Focal loss skeleton for A03."""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss module for multi-class classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0) -> None:
        """Initialize loss hyperparameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (B, C)
            targets: (B,)
        """
        raise NotImplementedError("Implement focal loss forward computation.")
