"""Compare baseline and SE YOLO checkpoints for A05."""

from __future__ import annotations

import argparse

from ultralytics import YOLO



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate A05 checkpoints")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--baseline", type=str, required=True)
    p.add_argument("--se", type=str, required=True)
    return p.parse_args()


def report(label: str, model_path: str, data: str) -> None:
    """Run validation and print summary metrics."""
    model = YOLO(model_path)
    res = model.val(data=data)
    print(f"[{label}] mAP@50: {res.box.map50:.4f}")
    print(f"[{label}] mAP@50-95: {res.box.map:.4f}")
    if hasattr(res.box, "maps"):
        for i, ap in enumerate(res.box.maps):
            print(f"[{label}] class_{i:02d} AP: {float(ap):.4f}")


def main(args: argparse.Namespace) -> None:
    """Evaluate both baseline and SE checkpoints."""
    report("baseline", args.baseline, args.data)
    report("se", args.se, args.data)


if __name__ == "__main__":
    main(parse_args())
