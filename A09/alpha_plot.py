"""Plot learned DyT alpha values across layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Plot DyT alpha parameters")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--save_path", type=str, default="./artifacts/alpha_plot.png")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Collect alpha values and save bar chart."""
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state"]
    keys = [k for k in state if k.endswith("alpha")]
    values = [float(state[k].mean().item()) for k in keys]

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(values)), values)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean alpha")
    ax.set_title("DyT alpha by layer")
    fig.tight_layout()
    fig.savefig(args.save_path)


if __name__ == "__main__":
    main(parse_args())
