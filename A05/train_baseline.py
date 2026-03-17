"""Baseline YOLOv8 training for A05."""

from __future__ import annotations

import argparse

from ultralytics import YOLO



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train YOLOv8 baseline")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--project", type=str, default="runs/a05")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Fine-tune yolov8s.pt with Ultralytics API."""
    model = YOLO("yolov8s.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name="baseline",
    )


if __name__ == "__main__":
    main(parse_args())
