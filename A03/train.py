"""Training scaffold for A03 with CE/Focal options."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from focal_loss import FocalLoss
from se_block import SEResNet18


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train A03 model")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./artifacts")
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def eval_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute one-vs-rest ROC-AUC."""
    model.eval()
    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs_all.append(torch.softmax(logits, dim=1).cpu())
            labels_all.append(y)
    probs = torch.cat(probs_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return float(roc_auc_score(labels, probs, multi_class="ovr"))


def main(args: argparse.Namespace) -> None:
    """Run training and save best model by validation ROC-AUC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10(root=args.data_dir, train=True, transform=tfm,
                          download=True)
    n_train = int(0.9 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SEResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss() if args.loss == "ce" else FocalLoss(
        gamma=args.gamma
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metric": val_auc,
                },
                Path(args.save_dir) / "best.pt",
            )


if __name__ == "__main__":
    main(parse_args())
