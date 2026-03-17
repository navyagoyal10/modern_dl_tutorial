"""Sanity check script for NT-Xent implementation."""

from __future__ import annotations

import torch

from simclr import NTXentLoss


if __name__ == "__main__":
    z1 = torch.randn(8, 128)
    z2 = z1 + 0.01 * torch.randn_like(z1)
    loss_fn = NTXentLoss(temperature=0.2)
    loss = loss_fn(z1, z2)
    print(f"NT-Xent sanity loss: {loss.item():.6f}")
