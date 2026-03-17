"""Training script for A07 decoder LM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import TextDataset
from model import DecoderLM
from tokenizer import BPETokenizer
from utils import perplexity, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train decoder LM")
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--corpus_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Train decoder LM and save best checkpoint by val perplexity."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tok = BPETokenizer.load(args.tokenizer_path)
    text = Path(args.corpus_path).read_text(encoding="utf-8")
    ids = tok.encode(text)
    ds = TextDataset(ids, seq_len=args.seq_len)
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DecoderLM(
        vocab_size=len(tok.vocab),
        max_len=args.seq_len,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_ppl = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = run_loss / max(seen, 1)

        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item() * x.size(0)
                val_seen += x.size(0)
        val_loss /= max(val_seen, 1)
        val_ppl = perplexity(val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_ppl:.4f}"
        )

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_metric": val_ppl,
                },
                Path(args.save_dir) / "best.pt",
            )


if __name__ == "__main__":
    main(parse_args())
