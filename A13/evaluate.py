"""
Evaluate the GRPO-trained Sudoku model.

Plays complete games against fresh puzzles and reports:
  - % of moves that are legal
  - % of moves that match the solution
  - % of puzzles fully solved
"""

import argparse
import random
from pathlib import Path

from sudoku_engine import SudokuGame, generate_puzzle, parse_move
from generate_data import board_to_prompt, SYSTEM_PROMPT


def load_model(adapter_path: str):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_move(model, tokenizer, game: SudokuGame, device) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": board_to_prompt(game)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with __import__("torch").no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded


def play_game(model, tokenizer, device, num_clues: int = 30, seed: int = 0, verbose: bool = False):
    puzzle, solution = generate_puzzle(num_clues=num_clues, seed=seed)
    game = SudokuGame(puzzle, solution)
    total_empty = game.total_empty_at_start()

    legal_moves = 0
    correct_moves = 0
    total_moves = 0

    for _ in range(total_empty + 5):  # allow a few extra tries
        if game.count_empty() == 0:
            break
        response = generate_move(model, tokenizer, game, device)
        move = parse_move(response)
        total_moves += 1

        if move is None:
            if verbose:
                print(f"  No move parsed from: {response[:80]}")
            continue

        x, y, num = move
        result = game.apply_move(x, y, num)
        is_legal = result.ok

        if is_legal:
            legal_moves += 1
            r, c = y - 1, x - 1
            if game.board[r][c] == game.solution[r][c]:
                correct_moves += 1

        if verbose:
            status = "OK" if is_legal else f"ILLEGAL: {result.reason}"
            print(f"  <{x},{y},{num}> → {status}")

    solved = game.is_solved()
    return {
        "solved": solved,
        "total_moves": total_moves,
        "legal_moves": legal_moves,
        "correct_moves": correct_moves,
        "total_empty": total_empty,
        "remaining_empty": game.count_empty(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="artifacts/grpo_sudoku/lora_adapter")
    parser.add_argument("--n_games", type=int, default=20)
    parser.add_argument("--clues", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(args.adapter)

    results = []
    for i in range(args.n_games):
        print(f"Game {i+1}/{args.n_games}...", end=" ", flush=True)
        r = play_game(model, tokenizer, device, num_clues=args.clues, seed=1000+i, verbose=args.verbose)
        results.append(r)
        print(f"solved={r['solved']}, correct={r['correct_moves']}/{r['total_empty']}")

    solved_pct = sum(r["solved"] for r in results) / len(results) * 100
    total_legal = sum(r["legal_moves"] for r in results)
    total_correct = sum(r["correct_moves"] for r in results)
    total_moves = sum(r["total_moves"] for r in results)
    total_empty = sum(r["total_empty"] for r in results)

    print(f"\n=== Results over {args.n_games} games ===")
    print(f"Puzzles solved:   {solved_pct:.1f}%")
    print(f"Legal move rate:  {100*total_legal/total_moves:.1f}%")
    print(f"Correct placements: {total_correct}/{total_empty} ({100*total_correct/total_empty:.1f}%)")


if __name__ == "__main__":
    main()
