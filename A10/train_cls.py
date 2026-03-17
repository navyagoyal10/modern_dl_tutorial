"""Train BERT classifier on SST-2 for A10."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from dataset_cls import build_sst2_loaders
from model_cls import BERTForClassification
from utils import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train SST-2 classifier")
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Train with warmup scheduler and save best validation accuracy model."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_sst2_loaders(args.model_name,
                                                     args.batch_size)
    model = BERTForClassification(args.model_name, num_labels=2).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = ce(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * batch["labels"].size(0)
            seen += batch["labels"].size(0)

        train_loss = run_loss / max(seen, 1)

        model.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"], batch["attention_mask"])
                pred = logits.argmax(dim=1)
                corr += (pred == batch["labels"]).sum().item()
                tot += batch["labels"].numel()
        val_acc = corr / max(tot, 1)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_metric": val_acc,
                },
                Path(args.save_dir) / "best_cls.pt",
            )


if __name__ == "__main__":
    main(parse_args())
