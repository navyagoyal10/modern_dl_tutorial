"""Fine-tuning script for A09 with DyT replacement."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from replace_norm import replace_layernorm_with_dyt
from utils import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train DeiT with DyT replacement")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--alpha_init", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Apply replacement and fine-tune model."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full = datasets.EuroSAT(args.data_dir, download=True, transform=tfm)
    n_train = int(0.8 * len(full))
    train_ds, val_ds = random_split(full, [n_train, len(full) - n_train])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = timm.create_model("deit3_small_patch16_224", pretrained=True,
                              num_classes=10)
    model = replace_layernorm_with_dyt(model, args.alpha_init).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = ce(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_seen += x.size(0)
        tr_loss /= max(tr_seen, 1)

        model.eval()
        va_loss, va_seen = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device))
                loss = ce(out, y.to(device))
                va_loss += loss.item() * x.size(0)
                va_seen += x.size(0)
        va_loss /= max(va_seen, 1)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {tr_loss:.4f} | Val Acc: {va_loss:.4f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_metric": va_loss,
                },
                Path(args.save_dir) / "best.pt",
            )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(args.save_dir) / "loss_curves.png")


if __name__ == "__main__":
    main(parse_args())
