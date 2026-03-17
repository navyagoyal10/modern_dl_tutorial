"""Train placeholder for A06."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return argparse.ArgumentParser(description="A06 has no training script").parse_args()


def main(_: argparse.Namespace) -> None:
    """Provide guidance for this assignment."""
    print("Use verify.py and visualise.py for A06 exercises.")


if __name__ == "__main__":
    main(parse_args())
