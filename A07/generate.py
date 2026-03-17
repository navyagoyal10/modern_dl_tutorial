"""Generation script for A07 decoder LM."""

from __future__ import annotations

import argparse
import random

import torch

from model import DecoderLM
from tokenizer import BPETokenizer



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Generate samples from decoder LM")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


def sample_next(logits: torch.Tensor, mode: str, top_k: int) -> int:
    """Sample one token via greedy or top-k sampling."""
    if mode == "greedy":
        return int(logits.argmax().item())
    topv, topi = torch.topk(logits, k=min(top_k, logits.numel()))
    probs = torch.softmax(topv, dim=0)
    pick = torch.multinomial(probs, num_samples=1)
    return int(topi[pick].item())


def main(args: argparse.Namespace) -> None:
    """Generate five text samples from random prefixes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(args.tokenizer_path)

    model = DecoderLM(
        vocab_size=len(tok.vocab),
        max_len=128,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prefixes = [random.randint(0, max(0, len(tok.vocab) - 1)) for _ in range(5)]
    for i, p in enumerate(prefixes, start=1):
        for mode in ["greedy", "topk"]:
            ids = [p]
            for _ in range(args.max_new_tokens):
                x = torch.tensor([ids], dtype=torch.long, device=device)
                logits = model(x)[0, -1] / max(args.temperature, 1e-6)
                nxt = sample_next(logits, mode=mode, top_k=args.top_k)
                ids.append(nxt)
            text = tok.decode(ids)
            print(f"Sample {i} [{mode}]: {text}")


if __name__ == "__main__":
    main(parse_args())
