"""Compare base and fine-tuned instruction model outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Compare generations")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--finetuned_model", type=str, required=True)
    p.add_argument("--save_path", type=str, default="./artifacts/compare.txt")
    return p.parse_args()


def generate(model, tok, prompt: str) -> str:
    """Generate a single response for a prompt."""
    x = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        y = model.generate(**x, max_new_tokens=128)
    return tok.decode(y[0], skip_special_tokens=True)


def main(args: argparse.Namespace) -> None:
    """Generate side-by-side outputs for held-out instructions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    ft = AutoModelForCausalLM.from_pretrained(args.finetuned_model).to(device)

    prompts = [
        "Explain gradient descent in simple terms.",
        "Write Python code to reverse a linked list.",
        "Summarize the benefits of unit testing.",
        "Give three tips for debugging GPU memory issues.",
        "Draft a polite email requesting deadline extension.",
    ]

    lines = []
    for i, p in enumerate(prompts, start=1):
        b = generate(base, tok, p)
        f = generate(ft, tok, p)
        lines.append(f"Prompt {i}: {p}\nBASE:\n{b}\nFT:\n{f}\n")

    out = Path(args.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(parse_args())
