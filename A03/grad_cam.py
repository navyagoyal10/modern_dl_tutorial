"""Grad-CAM utilities for A03."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class GradCAM:
    """Grad-CAM helper for a target convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Initialize and register hooks.

        Learner task:
            Register forward/backward hooks to cache activations and gradients.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # TODO: register hooks here.

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Generate a normalized Grad-CAM heatmap for a single image."""
        raise NotImplementedError("Implement Grad-CAM map generation.")


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay Grad-CAM heatmap on image in uint8 RGB format."""
    h, w = image.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    out = np.clip(0.6 * image.astype(np.float32) + 0.4 * color, 0, 255)
    return out.astype(np.uint8)
