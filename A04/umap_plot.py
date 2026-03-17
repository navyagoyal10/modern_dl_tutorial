"""UMAP plot for A04 encoder embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from umap import UMAP


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="UMAP embedding visualization")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--save_path", type=str, default="./artifacts/umap.png")
    p.add_argument("--max_samples", type=int, default=2000)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Extract features, fit UMAP, and save scatter plot."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    ds = datasets.CIFAR10(
        args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()
    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.to(device).eval()

    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(encoder(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
            if sum(len(b) for b in labels) >= args.max_samples:
                break

    x = np.concatenate(feats, axis=0)[: args.max_samples]
    y = np.concatenate(labels, axis=0)[: args.max_samples]

    emb = UMAP(n_components=2, random_state=42).fit_transform(x)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, s=8, cmap="tab10")
    ax.set_title("UMAP of SimCLR Embeddings")
    fig.colorbar(sc)
    fig.tight_layout()
    fig.savefig(args.save_path)


if __name__ == "__main__":
    main(parse_args())
