"""Download helper for A09 (EuroSAT)."""

from __future__ import annotations

import argparse

from torchvision.datasets import EuroSAT



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Download EuroSAT for A09")
    p.add_argument("--data_dir", type=str, default="./data")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Download EuroSAT data."""
    EuroSAT(root=args.data_dir, download=True)
    print(f"EuroSAT downloaded to {args.data_dir}")


if __name__ == "__main__":
    main(parse_args())
