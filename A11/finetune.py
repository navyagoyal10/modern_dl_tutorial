"""Instruction fine-tuning scaffold with Unsloth TODO integration."""

from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="LoRA fine-tune with Unsloth")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--output_dir", type=str, default="./artifacts")
    p.add_argument("--max_steps", type=int, default=200)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Set up SFT trainer with TODO-marked Unsloth model loading."""
    ds = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

    # TODO: Load quantized base model and tokenizer via Unsloth API.
    # TODO: Apply LoRA adapters with rank=args.rank and alpha=args.alpha.
    raise NotImplementedError("Fill in Unsloth loading and LoRA application.")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        max_steps=args.max_steps,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        args=train_args,
    )
    trainer.train()


if __name__ == "__main__":
    main(parse_args())
