"""Dynamic Tanh skeleton for A09."""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    """Dynamic Tanh normalization replacement module."""

    def __init__(self, hidden_dim: int, alpha_init: float = 1.0) -> None:
        """Initialize learnable DyT parameters.

        Learner task:
            Initialize alpha, gamma, beta parameters with proper shapes.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha_init = alpha_init
        # TODO: initialize alpha, gamma, beta as learnable nn.Parameter.
        raise NotImplementedError("Implement DyT parameter initialization.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Dynamic Tanh transformation.

        Args:
            x: Tensor of shape (..., hidden_dim)
        """
        # TODO: implement Dynamic Tanh forward pass.
        raise NotImplementedError("Implement DynamicTanh.forward.")
