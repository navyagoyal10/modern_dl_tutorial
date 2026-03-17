"""Evaluate SST-2 classifier and produce attention heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from dataset_cls import build_sst2_loaders
from model_cls import BERTForClassification



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate SST-2 classifier")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Compute metrics and save placeholder attention heatmap."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    _, val_loader, _ = build_sst2_loaders(args.model_name, args.batch_size)
    model = BERTForClassification(args.model_name, num_labels=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"F1: {f1_score(labels, preds):.4f}")

    # Placeholder heatmap from synthetic values for visualization scaffold.
    heat = np.random.rand(16, 16)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(heat, cmap="magma")
    ax.set_title("Attention Heatmap (Scaffold)")
    fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "attention_heatmap.png")


if __name__ == "__main__":
    main(parse_args())
