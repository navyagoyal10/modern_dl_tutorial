"""
Interactive: play Sudoku alongside the trained model.

Each turn you enter a move as <x,y,num> OR type 'model' to let the LLM play.
The engine validates all moves (yours and the model's) with the same rules.
"""

import argparse
import torch

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


def model_turn(model, tokenizer, game: SudokuGame, device: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": board_to_prompt(game)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="artifacts/grpo_sudoku/lora_adapter")
    parser.add_argument("--clues", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_model", action="store_true", help="Human-only mode (no model loaded)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = (None, None) if args.no_model else load_model(args.adapter)

    puzzle, solution = generate_puzzle(num_clues=args.clues, seed=args.seed)
    game = SudokuGame(puzzle, solution)

    print("\nSudoku Game")
    print("Commands: <x,y,num> to place, 'model' for AI move, 'hint' for correct move, 'quit' to exit\n")
    print(game.render())

    while not game.is_solved():
        print(f"\nEmpty cells: {game.count_empty()}")
        user_input = input("Your move: ").strip().lower()

        if user_input == "quit":
            break

        if user_input == "hint":
            for r in range(9):
                for c in range(9):
                    if not game.starter[r][c] and game.board[r][c] == 0:
                        print(f"Hint: <{c+1},{r+1},{solution[r][c]}>")
                        break
                else:
                    continue
                break
            continue

        if user_input == "model":
            if model is None:
                print("Model not loaded (--no_model flag set)")
                continue
            response = model_turn(model, tokenizer, game, device)
            print(f"Model response:\n{response}")
            move = parse_move(response)
            if move is None:
                print("Model gave no valid move.")
                continue
        else:
            move = parse_move(user_input if "<" in user_input else f"<{user_input}>")
            if move is None:
                print("Parse error. Use format: <x,y,num> or x,y,num")
                continue

        x, y, num = move
        result = game.apply_move(x, y, num)
        print(f"<{x},{y},{num}> → {result}")
        print()
        print(game.render())

    if game.is_solved():
        print("\nPuzzle solved!")
    else:
        print("\nGame ended.")


if __name__ == "__main__":
    main()
