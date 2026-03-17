"""Attention map extraction skeleton for A08."""

from __future__ import annotations

import numpy as np
import torch


class AttentionMapExtractor:
    """Helper to capture transformer attention maps from DeiT."""

    def __init__(self, model: torch.nn.Module):
        """Store model and register attention hooks."""
        self.model = model
        self.attn_maps = []
        # TODO: register hooks to capture attention tensors.

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward and return aggregated attention map."""
        # TODO: forward pass and aggregate CLS attention across heads/layers.
        raise NotImplementedError("Implement attention map extraction.")


def overlay(image: np.ndarray, attn_map: np.ndarray,
            alpha: float = 0.5) -> np.ndarray:
    """Overlay normalized attention map on RGB image."""
    attn = attn_map.astype(np.float32)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    heat = np.stack([attn, np.zeros_like(attn), 1.0 - attn], axis=-1)
    out = np.clip((1.0 - alpha) * image + alpha * 255.0 * heat, 0, 255)
    return out.astype(np.uint8)
