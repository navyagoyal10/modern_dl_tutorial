"""
evaluate.py — Compute test perplexity for trained models and print a summary.

Usage:
    python evaluate.py                    # evaluates both rnn and lstm
    python evaluate.py --model rnn        # evaluates rnn only

Loads best checkpoints from outputs/checkpoints/.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from data  import get_dataloaders
from rnn   import CharRNN
from lstm  import CharLSTM
from utils import get_device, compute_perplexity, plot_comparison


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_test_perplexity(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    is_lstm:    bool,
) -> tuple[float, float]:
    """
    Compute average cross-entropy loss and perplexity on a DataLoader.

    Returns:
        avg_loss:   mean cross-entropy per character
        perplexity: exp(avg_loss)
    """
    model.eval()
    total_loss  = 0.0
    total_chars = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        hidden = model.init_hidden(x_batch.size(0), device)

        if is_lstm:
            logits, _ = model(x_batch, hidden)
        else:
            logits, _ = model(x_batch, hidden)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y_batch.view(-1),
        )
        total_loss  += loss.item() * y_batch.numel()
        total_chars += y_batch.numel()

    avg_loss = total_loss / total_chars
    return avg_loss, compute_perplexity(avg_loss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_model(model_name: str, device: torch.device, test_loader, vocab) -> tuple[float, float]:
    """Load best checkpoint, infer model dims from it, evaluate on test set."""
    ckpt_path = Path(f"outputs/checkpoints/{model_name}_best.pt")
    if not ckpt_path.exists():
        print(f"  [{model_name.upper()}] No checkpoint found at {ckpt_path} — skipping.")
        return float("nan"), float("nan")

    ckpt       = torch.load(str(ckpt_path), map_location=device)
    cfg        = ckpt.get("model_config", {})
    vocab_size = cfg.get("vocab_size", vocab.size)
    embed_dim  = cfg.get("embed_dim",  64)
    hidden_dim = cfg.get("hidden_dim", 256)

    is_lstm    = model_name == "lstm"
    ModelClass = CharLSTM if is_lstm else CharRNN

    model = ModelClass(
        vocab_size = vocab_size,
        embed_dim  = embed_dim,
        hidden_dim = hidden_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded {model_name.upper()} — vocab={vocab_size}, embed={embed_dim}, hidden={hidden_dim}")

    criterion = nn.CrossEntropyLoss()
    avg_loss, ppl = compute_test_perplexity(model, test_loader, criterion, device, is_lstm)
    return avg_loss, ppl


def main(args: argparse.Namespace) -> None:
    device = get_device()
    _, _, test_loader, vocab = get_dataloaders(seq_len=100, batch_size=64)

    models_to_eval = ["rnn", "lstm"] if args.model == "both" else [args.model]

    results = {}
    for name in models_to_eval:
        print(f"\n[{name.upper()}]")
        loss, ppl = evaluate_model(name, device, test_loader, vocab)
        results[name] = (loss, ppl)

    # Summary table
    print("\n" + "=" * 60)
    print("Test Set Results")
    print("=" * 60)
    print(f"  {'Model':<10} {'Avg CE Loss':>12}  {'Perplexity':>12}  {'Target':>10}  {'Status':>6}")
    print(f"  {'-'*10} {'-'*12}  {'-'*12}  {'-'*10}  {'-'*6}")

    targets = {"rnn": 8.0, "lstm": 5.0}
    for name, (loss, ppl) in results.items():
        if math.isnan(ppl):
            continue
        target = targets.get(name, "—")
        status = "PASS" if ppl < target else "FAIL"
        print(
            f"  {name.upper():<10} {loss:>12.4f}  {ppl:>12.2f}  "
            f"{'< ' + str(target):>10}  {status:>6}"
        )

    # Comparison plot if both models evaluated
    if "rnn" in results and "lstm" in results:
        print("\nNote: comparison.png requires train loss histories.")
        print("Run train.py for both models, then this plot is auto-generated.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CharRNN/CharLSTM on test set")
    p.add_argument("--model", type=str, default="both", choices=["rnn", "lstm", "both"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())