"""Evaluation and Grad-CAM export for A03."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from grad_cam import GradCAM, overlay_heatmap
from se_block import SEResNet18


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate A03 checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Compute metrics and save Grad-CAM grid."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10(root=args.data_dir, train=False, transform=tfm,
                          download=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = SEResNet18(num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            probs_all.append(probs)
            labels_all.append(y.numpy())
            preds_all.append(preds)

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = np.concatenate(preds_all)

    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"Macro F1: {f1_score(labels, preds, average='macro'):.4f}")
    print(f"ROC-AUC: {roc_auc_score(labels, probs, multi_class='ovr'):.4f}")

    # One image per class for Grad-CAM grid.
    cam = GradCAM(model, model.base.layer4)
    images = []
    for c in range(10):
        idx = next(i for i, (_, y) in enumerate(ds) if y == c)
        img_t, _ = ds[idx]
        x = img_t.unsqueeze(0).to(device)
        heat = cam.generate(x, class_idx=c)
        img = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        images.append(overlay_heatmap(img, heat))

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f"Class {i}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "gradcam_grid.png")
    plt.close(fig)


if __name__ == "__main__":
    main(parse_args())
