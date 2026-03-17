"""Attention primitives for A06."""

from __future__ import annotations

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Shapes:
        q, k, v: (B, H, L, D)
        mask: (B, 1, L, L) or broadcastable

    Returns:
        output: (B, H, L, D)
        weights: (B, H, L, L)
    """
    # TODO: implement scaled dot-product attention core math.
    raise NotImplementedError("Implement scaled_dot_product_attention.")


def make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Create an upper-triangular causal mask for sequence length L."""
    # TODO: return boolean mask where future positions are masked.
    raise NotImplementedError("Implement make_causal_mask.")
