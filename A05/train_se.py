"""SE-head YOLOv8 training for A05."""

from __future__ import annotations

import argparse

from se_detect_head import load_se_yolo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train YOLOv8 with SE head")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--weights", type=str, default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--project", type=str, default="runs/a05")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Train SE-patched YOLO model."""
    model = load_se_yolo(args.weights, reduction=args.reduction)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name="se_head",
    )


if __name__ == "__main__":
    main(parse_args())
