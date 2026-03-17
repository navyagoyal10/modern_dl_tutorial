"""Dataset availability check for A10."""

from __future__ import annotations

import argparse

from datasets import load_dataset



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return argparse.ArgumentParser(description="Cache SST-2 and CoNLL-2003").parse_args()


def main(_: argparse.Namespace) -> None:
    """Download and cache NLP datasets used in A10."""
    load_dataset("glue", "sst2")
    load_dataset("conll2003")
    print("Cached SST-2 and CoNLL-2003.")


if __name__ == "__main__":
    main(parse_args())
