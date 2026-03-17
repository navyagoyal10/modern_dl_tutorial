"""Core SimCLR modules for A04."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """MLP projection head mapping encoder features to contrastive space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize projection layers.

        Learner task:
            Build a 2-layer MLP projection head.
        """
        super().__init__()
        # TODO: define projection head modules.
        raise NotImplementedError("Implement ProjectionHead layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder output features."""
        raise NotImplementedError("Implement ProjectionHead.forward.")


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss."""

    def __init__(self, temperature: float = 0.2) -> None:
        """Initialize NT-Xent with temperature scaling."""
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss for positive pairs.

        Args:
            z1: (B, D) projected features for view 1.
            z2: (B, D) projected features for view 2.
        """
        raise NotImplementedError("Implement NT-Xent forward computation.")


class SimCLR(nn.Module):
    """Wrapper around encoder and projection head."""

    def __init__(self, encoder: nn.Module, projector: ProjectionHead) -> None:
        """Store encoder and projector."""
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project a batch of images."""
        h = self.encoder(x)
        if h.ndim > 2:
            h = torch.flatten(h, 1)
        return self.projector(h)
