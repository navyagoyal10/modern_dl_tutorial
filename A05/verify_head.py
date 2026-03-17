"""Shape sanity check for SE detection head."""

from __future__ import annotations

import torch

from se_detect_head import SEDetectHead


if __name__ == "__main__":
    head = SEDetectHead(in_channels=256, reduction=16)
    x = torch.randn(2, 256, 20, 20)
    y = head(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
