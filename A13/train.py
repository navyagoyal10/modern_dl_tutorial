"""
GRPO fine-tuning of a small LLM (Qwen2.5-1.5B-Instruct) to solve Sudoku moves.

Uses Unsloth for fast LoRA + 4-bit loading and TRL's GRPOTrainer.

Reward functions:
  1. format_reward    — does response contain <x,y,num>?
  2. valid_move_reward — is <x,y,num> a legal Sudoku move on the board?
  3. correct_reward   — does the move match the solution?

Run:
    python generate_data.py   # first
    python train.py
"""

import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from sudoku_engine import SudokuGame, parse_move

# --------------------------------------------------------------------------- #
# Model config                                                                 #
# --------------------------------------------------------------------------- #

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"  # swap to Llama-3.2-1B-Instruct if preferred
MAX_SEQ_LEN = 1024
LORA_RANK = 16

# --------------------------------------------------------------------------- #
# Load model with Unsloth                                                      #
# --------------------------------------------------------------------------- #

def load_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,          # auto: bf16 on Ampere+, fp16 elsewhere
        load_in_4bit=True,   # QLoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

def load_dataset(path: str = "data/sudoku_train.jsonl") -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


# --------------------------------------------------------------------------- #
# Reward functions                                                             #
# --------------------------------------------------------------------------- #

_MOVE_RE = re.compile(r"<(\d+),(\d+),(\d+)>")


def format_reward(completions, **kwargs) -> list[float]:
    """Reward 0.2 if response contains <x,y,num> pattern."""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        rewards.append(0.2 if _MOVE_RE.search(text) else 0.0)
    return rewards


def valid_move_reward(completions, puzzle_flat, solution_flat, **kwargs) -> list[float]:
    """
    Reward 0.4 if the parsed move is a legal Sudoku move (not violating row/col/box),
    regardless of whether it matches the solution.
    """
    rewards = []
    for c, pf, sf in zip(completions, puzzle_flat, solution_flat):
        text = c[0]["content"] if isinstance(c, list) else c
        move = parse_move(text)
        if move is None:
            rewards.append(0.0)
            continue
        # reconstruct game from flat puzzle string
        puzzle = [[int(pf[r*9+col]) for col in range(9)] for r in range(9)]
        solution = [[int(sf[r*9+col]) for col in range(9)] for r in range(9)]
        game = SudokuGame(puzzle, solution)
        result = game.apply_move(*move)
        rewards.append(0.4 if result.ok else 0.0)
    return rewards


def correct_reward(completions, puzzle_flat, solution_flat, **kwargs) -> list[float]:
    """
    Reward 1.0 if parsed move matches the correct solution digit for that cell.
    (This stacks on top of valid_move_reward since a correct move is always valid.)
    """
    rewards = []
    for c, pf, sf in zip(completions, puzzle_flat, solution_flat):
        text = c[0]["content"] if isinstance(c, list) else c
        move = parse_move(text)
        if move is None:
            rewards.append(0.0)
            continue
        x, y, num = move
        if not (1 <= x <= 9 and 1 <= y <= 9 and 1 <= num <= 9):
            rewards.append(0.0)
            continue
        r, col = y - 1, x - 1
        correct_num = int(sf[r*9 + col])
        rewards.append(1.0 if num == correct_num else 0.0)
    return rewards


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

def main():
    if not Path("data/sudoku_train.jsonl").exists():
        print("Dataset not found. Run: python generate_data.py")
        return

    model, tokenizer = load_model()
    dataset = load_dataset()

    training_args = GRPOConfig(
        output_dir="artifacts/grpo_sudoku",
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=6,          # completions sampled per prompt
        max_prompt_length=768,
        max_completion_length=256,
        temperature=0.8,
        beta=0.001,                 # KL penalty coefficient
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, valid_move_reward, correct_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained("artifacts/grpo_sudoku/lora_adapter")
    tokenizer.save_pretrained("artifacts/grpo_sudoku/lora_adapter")
    print("Saved LoRA adapter → artifacts/grpo_sudoku/lora_adapter")


if __name__ == "__main__":
    main()
