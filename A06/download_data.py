"""No dataset download needed for A06."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return argparse.ArgumentParser(description="A06 uses synthetic tensors").parse_args()


def main(_: argparse.Namespace) -> None:
    """Print guidance for this assignment."""
    print("A06 uses synthetic inputs in verify.py and visualise.py.")


if __name__ == "__main__":
    main(parse_args())
