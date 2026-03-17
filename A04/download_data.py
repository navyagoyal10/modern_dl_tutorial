"""Download CIFAR-10 for A04 SimCLR."""

from __future__ import annotations

import argparse

from torchvision.datasets import CIFAR10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download CIFAR-10")
    parser.add_argument("--data_dir", type=str, default="./data")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Download train and test splits."""
    CIFAR10(root=args.data_dir, train=True, download=True)
    CIFAR10(root=args.data_dir, train=False, download=True)
    print(f"CIFAR-10 downloaded to {args.data_dir}")


if __name__ == "__main__":
    main(parse_args())
