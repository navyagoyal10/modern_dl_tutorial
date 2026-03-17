"""Verification script for LayerNorm to DyT replacement."""

from __future__ import annotations

import torch
import torch.nn as nn

from replace_norm import replace_layernorm_with_dyt


if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.ReLU())
    _ = replace_layernorm_with_dyt(model, alpha_init=1.0)
    x = torch.randn(4, 16)
    y = model(x)
    print(f"Output shape after replacement: {tuple(y.shape)}")
