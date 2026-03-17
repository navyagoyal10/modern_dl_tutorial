"""Verification script for custom MHA against PyTorch MHA."""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import make_causal_mask
from mha import MultiHeadAttention


def main() -> None:
    """Run numerical equivalence and causal perturbation checks."""
    torch.manual_seed(42)
    b, l, d, h = 2, 12, 64, 8
    x = torch.randn(b, l, d)

    ref = nn.MultiheadAttention(d, h, batch_first=True)
    custom = MultiHeadAttention(d, h)

    # TODO: copy weights from ref MHA modules into custom projections.
    raise NotImplementedError("Copy weights and run max error assertion.")

    mask = make_causal_mask(l, x.device)
    _ = mask
    # TODO: run perturbation test with and without mask.


if __name__ == "__main__":
    main()
