"""
Generate a Sudoku dataset for GRPO training.

Each example is a prompt describing a Sudoku board and asking for the next move.
The model must respond with a move in <x,y,num> format.

We generate multiple (puzzle, solution) pairs and build training samples
where the model is given a partially-played game state and must choose a correct move.
"""

import json
import random
from pathlib import Path
from sudoku_engine import generate_puzzle, SudokuGame

SYSTEM_PROMPT = (
    "You are a Sudoku solver. "
    "The board is 9×9. Rows and columns are numbered 1–9 (top-left is row 1, col 1). "
    "Given the current board state, output your next move as <x,y,num> where x=column, y=row, num=digit. "
    "Think step by step inside <think>...</think> tags, then give your move."
)


def board_to_prompt(game: SudokuGame) -> str:
    flat = game.to_string_flat()
    board_str = game.render()
    return (
        f"Current Sudoku board (. = empty, [n] = your prior placement):\n"
        f"{board_str}\n\n"
        f"Board as flat string (row-major, 0=empty): {flat}\n\n"
        f"Give your next move as <x,y,num>."
    )


def collect_correct_moves(game: SudokuGame) -> list[tuple[int, int, int]]:
    """Return all currently valid moves (cells that match solution)."""
    moves = []
    for r in range(9):
        for c in range(9):
            if not game.starter[r][c] and game.board[r][c] == 0:
                num = game.solution[r][c]
                moves.append((c + 1, r + 1, num))  # x=col+1, y=row+1
    return moves


def generate_dataset(
    n_puzzles: int = 500,
    clues_range: tuple[int, int] = (25, 35),
    samples_per_puzzle: int = 3,
    seed: int = 0,
) -> list[dict]:
    random.seed(seed)
    samples = []
    for i in range(n_puzzles):
        num_clues = random.randint(*clues_range)
        puzzle, solution = generate_puzzle(num_clues=num_clues, seed=i)
        game = SudokuGame(puzzle, solution)

        # optionally play a few random correct moves to vary board state
        pre_moves = random.randint(0, 10)
        all_moves = collect_correct_moves(game)
        random.shuffle(all_moves)
        for move in all_moves[:pre_moves]:
            game.apply_move(*move)

        correct_moves = collect_correct_moves(game)
        if not correct_moves:
            continue

        for _ in range(samples_per_puzzle):
            target_move = random.choice(correct_moves)
            x, y, num = target_move
            prompt = board_to_prompt(game)
            samples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "answer_x": x,
                "answer_y": y,
                "answer_num": num,
                "puzzle_flat": game.to_string_flat(),
                "solution_flat": "".join(
                    str(game.solution[r][c]) for r in range(9) for c in range(9)
                ),
            })

    return samples


if __name__ == "__main__":
    out_path = Path("data/sudoku_train.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    print("Generating training dataset...")
    samples = generate_dataset(n_puzzles=500, samples_per_puzzle=3)
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Saved {len(samples)} samples → {out_path}")
