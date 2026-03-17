"""Throughput profiling for A12 multimodal model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from multimodal_model import MultimodalClassifier



def measure_throughput(
    model: MultimodalClassifier,
    batch_size: int,
    device: str,
    n_runs: int = 20,
) -> float:
    """Measure forward throughput in images/second for one batch size."""
    model.eval().to(device)
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    input_ids = torch.ones(batch_size, 64, dtype=torch.long, device=device)
    attn_mask = torch.ones(batch_size, 64, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(3):
            _ = model(images, input_ids, attn_mask)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(images, input_ids, attn_mask)
    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    throughput = (n_runs * batch_size) / elapsed
    print(f"Batch size {batch_size:4d}: {throughput:7.1f} images/sec")
    return throughput


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Profile A12 multimodal throughput")
    p.add_argument("--save_path", type=str,
                   default="./artifacts/throughput_plot.png")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run throughput profile for several batch sizes and plot results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalClassifier()

    batch_sizes, throughputs = [], []
    for bs in [1, 8, 32, 128]:
        try:
            thr = measure_throughput(model, bs, device)
            batch_sizes.append(bs)
            throughputs.append(thr)
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {bs}: OOM")
            break

    out = Path(args.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(batch_sizes, throughputs, marker="o")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Images / second")
    ax.set_title("A12 Throughput")
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main(parse_args())
