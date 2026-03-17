"""Training script for A08 DeiT-3 fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EuroSAT

from model import build_deit3
from utils import build_transform, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train DeiT-3 on EuroSAT")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def eval_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute macro ROC-AUC on validation data."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            probs.append(torch.softmax(out, dim=1).cpu())
            labels.append(y)
    p = torch.cat(probs).numpy()
    y = torch.cat(labels).numpy()
    return float(roc_auc_score(y, p, multi_class="ovr"))


def main(args: argparse.Namespace) -> None:
    """Run fine-tuning with early stopping on val ROC-AUC."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    full = EuroSAT(args.data_dir, download=True, transform=build_transform(True))
    n_train = int(0.8 * len(full))
    n_val = len(full) - n_train
    train_ds, val_ds = random_split(full, [n_train, n_val])
    val_ds.dataset.transform = build_transform(False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_deit3(num_classes=10, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_auc, bad_epochs = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = ce(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = run_loss / max(seen, 1)
        val_auc = eval_auc(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_metric": val_auc,
                },
                Path(args.save_dir) / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main(parse_args())
