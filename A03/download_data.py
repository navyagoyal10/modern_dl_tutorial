"""Download CIFAR-10 for A03 experiments."""

from __future__ import annotations

import argparse

from torchvision.datasets import CIFAR10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Download CIFAR-10")
    p.add_argument("--data_dir", type=str, default="./data")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Download CIFAR-10 train and test splits."""
    CIFAR10(root=args.data_dir, train=True, download=True)
    CIFAR10(root=args.data_dir, train=False, download=True)
    print(f"CIFAR-10 downloaded to {args.data_dir}")


if __name__ == "__main__":
    main(parse_args())
