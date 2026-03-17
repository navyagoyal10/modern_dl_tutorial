"""Train BERT NER model for A10."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from dataset_ner import build_conll_loaders
from model_ner import BERTForNER
from utils import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train CoNLL-2003 NER")
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Train token classifier with warmup scheduler."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, label_list, _ = build_conll_loaders(
        args.model_name, args.batch_size
    )
    model = BERTForNER(args.model_name, num_labels=len(label_list)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = ce(logits.view(-1, logits.size(-1)),
                      batch["labels"].view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * batch["input_ids"].size(0)
            seen += batch["input_ids"].size(0)

        train_loss = run_loss / max(seen, 1)

        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = ce(logits.view(-1, logits.size(-1)),
                          batch["labels"].view(-1))
                val_loss += loss.item() * batch["input_ids"].size(0)
                val_seen += batch["input_ids"].size(0)
        val_score = -val_loss / max(val_seen, 1)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_score:.4f}"
        )

        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_metric": val_score,
                },
                Path(args.save_dir) / "best_ner.pt",
            )


if __name__ == "__main__":
    main(parse_args())
