"""Download script for Hindi CC-100 subset (A07)."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Download Hindi CC-100 text")
    p.add_argument("--url", type=str, required=True)
    p.add_argument("--out_path", type=str, default="./data/cc100_hi.txt")
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Download corpus file to local path."""
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(args.url, timeout=args.timeout)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    print(f"Downloaded corpus to {out}")


if __name__ == "__main__":
    main(parse_args())
