"""
generate.py — Sample text from a trained CharRNN or CharLSTM.

Usage:
    python generate.py --model lstm --seed "ROMEO:" --temperatures 0.5 0.8 1.2
    python generate.py --model rnn  --seed "To be"  --length 500

Saves samples to outputs/samples/{model}_T{temp}.txt and prints to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data  import get_dataloaders, Vocabulary
from rnn   import CharRNN
from lstm  import CharLSTM
from utils import get_device


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model:       torch.nn.Module,
    vocab:       Vocabulary,
    seed:        str,
    length:      int,
    temperature: float,
    device:      torch.device,
    is_lstm:     bool,
) -> str:
    """
    Autoregressively sample characters from a trained model.

    Args:
        model:       trained CharRNN or CharLSTM
        vocab:       vocabulary used during training
        seed:        starting string; the model is primed with this before sampling
        length:      number of new characters to generate (seed not counted)
        temperature: softmax temperature T > 0
                     T→0: greedy (always picks most likely character)
                     T=1: sample from raw trained distribution
                     T>1: more uniform, more random
        device:      torch device
        is_lstm:     True for CharLSTM

    Returns:
        generated string (seed + length new characters)
    """
    model.eval()

    # Encode seed
    indices = vocab.encode(seed)
    if not indices:
        raise ValueError("Seed string is empty or contains unknown characters.")

    # Prime the hidden state by feeding the seed (all but last char)
    hidden = model.init_hidden(1, device)
    for idx in indices[:-1]:
        x = torch.tensor([[idx]], dtype=torch.long, device=device)  # (1, 1)
        if is_lstm:
            _, hidden = model(x, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            _, hidden = model(x, hidden)
            hidden = hidden.detach()

    # Start generating from the last seed character
    generated = list(seed)
    current_idx = indices[-1]

    for _ in range(length):
        x = torch.tensor([[current_idx]], dtype=torch.long, device=device)  # (1, 1)

        if is_lstm:
            logits, hidden = model(x, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            logits, hidden = model(x, hidden)
            hidden = hidden.detach()

        # logits: (1, 1, vocab_size) → (vocab_size,)
        logits = logits.squeeze() / temperature
        probs  = F.softmax(logits, dim=-1)

        # Sample from the distribution
        current_idx = torch.multinomial(probs, num_samples=1).item()
        generated.append(vocab.idx2char[current_idx])

    return "".join(generated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device  = get_device()
    is_lstm = args.model == "lstm"

    # Load data (we only need the vocab)
    _, _, _, vocab = get_dataloaders(seq_len=100, batch_size=1)

    # Load checkpoint and infer model dims from it
    ckpt_path = f"outputs/checkpoints/{args.model}_best.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)
    cfg       = ckpt.get("model_config", {})
    vocab_size = cfg.get("vocab_size", vocab.size)
    embed_dim  = cfg.get("embed_dim",  64)
    hidden_dim = cfg.get("hidden_dim", 256)

    ModelClass = CharLSTM if is_lstm else CharRNN
    model = ModelClass(
        vocab_size = vocab_size,
        embed_dim  = embed_dim,
        hidden_dim = hidden_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.model.upper()} — vocab={vocab_size}, embed={embed_dim}, hidden={hidden_dim}")

    # Generate at each temperature
    out_dir = Path("outputs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    for T in args.temperatures:
        print(f"\n{'='*60}")
        print(f"Model: {args.model.upper()} | Seed: {repr(args.seed)} | Temp: {T}")
        print("=" * 60)

        sample = generate(
            model       = model,
            vocab       = vocab,
            seed        = args.seed,
            length      = args.length,
            temperature = T,
            device      = device,
            is_lstm     = is_lstm,
        )
        print(sample)

        # Save to file
        out_path = out_dir / f"{args.model}_T{T:.2f}.txt"
        out_path.write_text(sample, encoding="utf-8")
        print(f"\nSaved → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from trained CharRNN or CharLSTM")
    p.add_argument("--model",        type=str,   default="lstm", choices=["rnn", "lstm"])
    p.add_argument("--seed",         type=str,   default="ROMEO:")
    p.add_argument("--length",       type=int,   default=500)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.8, 1.2])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())