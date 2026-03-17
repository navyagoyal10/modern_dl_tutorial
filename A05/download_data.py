"""VisDrone data preparation scaffold for YOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Prepare VisDrone for YOLOv8")
    p.add_argument("--data_dir", type=str, default="./data/visdrone")
    p.add_argument("--images_dir", type=str, default="images")
    p.add_argument("--labels_dir", type=str, default="labels")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Create expected YOLO directory tree and dataset YAML."""
    data_dir = Path(args.data_dir)
    for split in ["train", "val", "test"]:
        (data_dir / args.images_dir / split).mkdir(parents=True, exist_ok=True)
        (data_dir / args.labels_dir / split).mkdir(parents=True, exist_ok=True)

    yaml_text = """path: .
train: images/train
val: images/val
test: images/test
nc: 10
names: [pedestrian, person, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]
"""
    (data_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    print("Directory structure ready. Add download/convert logic in TODO areas.")


if __name__ == "__main__":
    main(parse_args())
