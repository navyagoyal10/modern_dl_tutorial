"""Visualize attention maps for A06."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from attention import make_causal_mask
from mha import MultiHeadAttention


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Visualize custom attention maps")
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--heads", type=int, default=8)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Generate and save masked/unmasked attention heatmaps."""
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, args.length, args.d_model)
    mha = MultiHeadAttention(args.d_model, args.heads)

    _, w_no = mha(x, None)
    _, w_ca = mha(x, make_causal_mask(args.length, x.device))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(w_no[0, 0].detach().numpy(), ax=axes[0], cbar=False)
    sns.heatmap(w_ca[0, 0].detach().numpy(), ax=axes[1], cbar=False)
    axes[0].set_title("No Mask")
    axes[1].set_title("Causal Mask")
    fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "attention_maps.png")


if __name__ == "__main__":
    main(parse_args())
