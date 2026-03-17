"""Evaluation and attention map export for A08."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import EuroSAT

from attention_extractor import AttentionMapExtractor, overlay
from model import build_deit3
from utils import build_transform


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate A08 checkpoint")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Compute metrics and save 2x5 attention-map visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    ds = EuroSAT(args.data_dir, download=True, transform=build_transform(False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = build_deit3(num_classes=10, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs_all, preds_all, labels_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            probs_all.append(probs)
            preds_all.append(probs.argmax(axis=1))
            labels_all.append(y.numpy())

    probs = np.concatenate(probs_all)
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)

    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"Macro F1: {f1_score(labels, preds, average='macro'):.4f}")
    print(f"ROC-AUC: {roc_auc_score(labels, probs, multi_class='ovr'):.4f}")

    extractor = AttentionMapExtractor(model)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    shown = 0
    for img_t, _ in ds:
        if shown >= 10:
            break
        attn = extractor.extract(img_t.unsqueeze(0).to(device)).squeeze().cpu()
        attn = torch.nn.functional.interpolate(
            attn.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        img = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        vis = overlay(img, attn)
        ax = axes.flat[shown]
        ax.imshow(vis)
        ax.axis("off")
        shown += 1

    fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "attention_grid.png")


if __name__ == "__main__":
    main(parse_args())
