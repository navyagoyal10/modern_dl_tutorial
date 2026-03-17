"""Evaluation helper for A09."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="A09 evaluation placeholder")
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Print guidance for A09 evaluation."""
    print(f"Checkpoint ready for analysis: {args.checkpoint}")


if __name__ == "__main__":
    main(parse_args())
