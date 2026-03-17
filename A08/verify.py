"""Quick construction verification for A08 model builder."""

from __future__ import annotations

from model import build_deit3


if __name__ == "__main__":
    net = build_deit3(num_classes=10, pretrained=False)
    print(type(net).__name__)
