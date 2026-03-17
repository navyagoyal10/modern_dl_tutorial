"""Rank-vs-loss experiment scaffold for LoRA."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run LoRA rank experiment")
    p.add_argument("--save_path", type=str,
                   default="./artifacts/rank_vs_loss.png")
    return p.parse_args()


def run_single_rank(rank: int) -> float:
    """Run training for one rank and return validation loss.

    Learner task:
        Integrate training call and parse best validation loss.
    """
    # TODO: launch fine-tuning with given rank and capture val loss.
    raise NotImplementedError("Implement rank experiment training routine.")


def main(args: argparse.Namespace) -> None:
    """Run ranks {4, 8, 16, 32} and save comparison curve."""
    ranks = [4, 8, 16, 32]
    losses = [run_single_rank(r) for r in ranks]

    out = Path(args.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, losses, marker="o")
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Rank vs Validation Loss")
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main(parse_args())
