"""Multi-head attention skeleton for A06."""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Custom multi-head attention module.

    Input shape:
        x: (B, L, D)
    Output shape:
        y: (B, L, D)
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        """Initialize projection layers for Q, K, V and output."""
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads
        # TODO: define q_proj, k_proj, v_proj, out_proj.
        raise NotImplementedError("Implement projection matrices in __init__.")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head self-attention."""
        # TODO: project x to Q, K, V.
        # TODO: split into heads: (B, L, D) -> (B, H, L, Hd).
        # TODO: call scaled_dot_product_attention.
        # TODO: concatenate heads back to (B, L, D).
        # TODO: final output projection.
        raise NotImplementedError("Implement MultiHeadAttention.forward.")
