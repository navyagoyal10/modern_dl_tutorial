"""Download helper for Alpaca-style instruction dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Download Alpaca dataset")
    p.add_argument("--out_path", type=str, default="./data/alpaca_1k.jsonl")
    p.add_argument("--max_rows", type=int, default=1000)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Download subset of Alpaca and save as JSONL."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = ds.select(range(min(args.max_rows, len(ds))))
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                f"{ {'instruction': row['instruction'], 'input': row['input'], 'output': row['output']} }\n"
            )
    print(f"Saved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main(parse_args())
