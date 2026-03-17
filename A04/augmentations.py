"""Augmentations for SimCLR (A04)."""

from __future__ import annotations

from torchvision import transforms


def simclr_augmentation(image_size: int) -> transforms.Compose:
    """Build SimCLR augmentation pipeline."""
    # TODO: Implement SimCLR augmentations (crop, color jitter, blur, etc.).
    raise NotImplementedError("Implement simclr_augmentation pipeline.")


class TwoViewTransform:
    """Apply base transform twice to create a positive pair."""

    def __init__(self, base_transform: transforms.Compose) -> None:
        """Store base transform."""
        self.base_transform = base_transform

    def __call__(self, x):
        """Return two independently augmented views."""
        return self.base_transform(x), self.base_transform(x)
