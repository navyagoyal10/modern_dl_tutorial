"""Model skeleton for A01."""

from __future__ import annotations

import torch
import torch.nn as nn


class MyCNN(nn.Module):
    """Simple CNN for MNIST classification.

    Input shape:
        x: (B, 1, 28, 28)
    Output shape:
        logits: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize model layers.

        Args:
            num_classes: Number of target classes.
        """
        super().__init__()
        # TODO: Define convolution and classifier layers.
        # Example blocks: conv -> relu -> pool, repeated, then linear head.
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of images.

        Args:
            x: Input image tensor of shape (B, 1, 28, 28).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        raise NotImplementedError(
            "Implement MyCNN.forward with the planned conv/classifier path."
        )
