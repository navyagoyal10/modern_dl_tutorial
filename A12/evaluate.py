"""Ablation evaluation script for A12."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate multimodal ablations")
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run image-only, text-only, and joint ablation evaluations."""
    # TODO: load model and dataset, run three ablation modes.
    print("Mode         | Accuracy")
    print("image-only   | TODO")
    print("text-only    | TODO")
    print("both         | TODO")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main(parse_args())
