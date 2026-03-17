"""Verification script for LoRA layer behavior."""

from __future__ import annotations

import torch

from lora_linear import LoRALinear


if __name__ == "__main__":
    layer = LoRALinear(16, 8, rank=4, alpha=8.0)
    x = torch.randn(2, 16)
    y = layer(x)
    print(f"LoRA output shape: {tuple(y.shape)}")
