"""Linear probe training for A04."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Linear probe on frozen encoder")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def train_fraction(
    frac: float,
    encoder: nn.Module,
    train_ds,
    test_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Train probe on a fraction of labels and return test accuracy."""
    n = max(1, int(frac * len(train_ds)))
    subset = Subset(train_ds, list(range(n)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

    head = nn.Linear(512, 10).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    encoder.eval()
    for _ in range(args.epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                feats = encoder(x)
            logits = head(feats)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = head(encoder(x))
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main(args: argparse.Namespace) -> None:
    """Evaluate linear probing across label fractions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(args.data_dir, train=True, transform=tfm,
                                download=True)
    test_ds = datasets.CIFAR10(args.data_dir, train=False, transform=tfm,
                               download=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()
    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.to(device)

    fracs = [0.01, 0.05, 0.1, 0.5, 1.0]
    accs = [
        train_fraction(f, encoder, train_ds, test_loader, args, device)
        for f in fracs
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, accs, marker="o")
    ax.set_xlabel("Label Fraction")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear Probe Accuracy")
    fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "linear_probe_curve.png")


if __name__ == "__main__":
    main(parse_args())
