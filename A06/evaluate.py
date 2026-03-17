"""Evaluation placeholder for A06."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return argparse.ArgumentParser(description="A06 has no eval loop").parse_args()


def main(_: argparse.Namespace) -> None:
    """Print evaluation guidance."""
    print("Evaluate by running verify.py and inspecting visualise.py output.")


if __name__ == "__main__":
    main(parse_args())
