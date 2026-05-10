"""
Sudoku game engine.

Move format: <x,y,num>
  x   — column (1-9, left to right)
  y   — row    (1-9, top to bottom)
  num — digit  (1-9)

Starter cells are locked; the model can overwrite its own prior placements but
not cells that were part of the original puzzle.
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
# Result type                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class MoveResult:
    ok: bool
    reason: Optional[str] = None   # None when ok=True

    def __str__(self) -> str:
        return "OK" if self.ok else f"ILLEGAL: {self.reason}"


# --------------------------------------------------------------------------- #
# Core board                                                                   #
# --------------------------------------------------------------------------- #

class SudokuGame:
    """
    Tracks a live Sudoku game.

    Attributes
    ----------
    board      : 9×9 grid, 0 = empty
    starter    : boolean mask — True where a cell is a given/starter clue
    solution   : full solution (used for reward computation)
    """

    def __init__(self, puzzle: list[list[int]], solution: list[list[int]]):
        assert len(puzzle) == 9 and all(len(r) == 9 for r in puzzle)
        assert len(solution) == 9 and all(len(r) == 9 for r in solution)
        self.board: list[list[int]] = copy.deepcopy(puzzle)
        self.solution: list[list[int]] = copy.deepcopy(solution)
        self.starter: list[list[bool]] = [
            [puzzle[r][c] != 0 for c in range(9)] for r in range(9)
        ]

    # ----------------------------------------------------------------------- #
    # Move validation                                                          #
    # ----------------------------------------------------------------------- #

    def apply_move(self, x: int, y: int, num: int) -> MoveResult:
        """
        Apply move <x,y,num>. x=column (1-9), y=row (1-9), num=digit (1-9).
        Returns MoveResult(ok=True) on success, MoveResult(ok=False, reason=…) on failure.
        Illegal moves do NOT modify board state.
        """
        # --- range checks ---
        if not (1 <= x <= 9):
            return MoveResult(False, f"column x={x} out of range 1-9")
        if not (1 <= y <= 9):
            return MoveResult(False, f"row y={y} out of range 1-9")
        if not (1 <= num <= 9):
            return MoveResult(False, f"digit num={num} out of range 1-9")

        r, c = y - 1, x - 1  # convert to 0-indexed

        # --- starter cell lock ---
        if self.starter[r][c]:
            return MoveResult(False, f"({x},{y}) is a starter/given cell and cannot be overwritten")

        # --- Sudoku constraint checks ---
        # temporarily clear current cell so self-overwrite doesn't false-positive
        old = self.board[r][c]
        self.board[r][c] = 0

        result = self._check_constraints(r, c, num)
        if not result.ok:
            self.board[r][c] = old   # restore
            return result

        self.board[r][c] = num
        return MoveResult(True)

    def _check_constraints(self, r: int, c: int, num: int) -> MoveResult:
        # row
        if num in self.board[r]:
            return MoveResult(False, f"{num} already in row {r+1}")
        # column
        col = [self.board[row][c] for row in range(9)]
        if num in col:
            return MoveResult(False, f"{num} already in column {c+1}")
        # 3×3 box
        br, bc = (r // 3) * 3, (c // 3) * 3
        box = [self.board[br+dr][bc+dc] for dr in range(3) for dc in range(3)]
        if num in box:
            return MoveResult(False, f"{num} already in 3×3 box at rows {br+1}-{br+3}, cols {bc+1}-{bc+3}")
        return MoveResult(True)

    # ----------------------------------------------------------------------- #
    # State queries                                                            #
    # ----------------------------------------------------------------------- #

    def is_solved(self) -> bool:
        return all(self.board[r][c] != 0 for r in range(9) for c in range(9)) \
               and all(self._check_constraints(r, c, self.board[r][c]).ok
                       for r in range(9) for c in range(9))

    def count_correct(self) -> int:
        """Number of non-starter cells filled with the correct digit."""
        return sum(
            1
            for r in range(9)
            for c in range(9)
            if not self.starter[r][c]
            and self.board[r][c] != 0
            and self.board[r][c] == self.solution[r][c]
        )

    def count_empty(self) -> int:
        return sum(1 for r in range(9) for c in range(9) if self.board[r][c] == 0)

    def total_empty_at_start(self) -> int:
        return sum(1 for r in range(9) for c in range(9) if not self.starter[r][c])

    # ----------------------------------------------------------------------- #
    # Rendering                                                                #
    # ----------------------------------------------------------------------- #

    def render(self) -> str:
        lines = []
        for r in range(9):
            if r in (3, 6):
                lines.append("------+-------+------")
            row_str = ""
            for c in range(9):
                if c in (3, 6):
                    row_str += " | "
                v = self.board[r][c]
                if v == 0:
                    row_str += "."
                elif self.starter[r][c]:
                    row_str += str(v)
                else:
                    row_str += f"[{v}]"  # model-placed cells in brackets
            lines.append(row_str)
        return "\n".join(lines)

    def to_string_flat(self) -> str:
        """81-char string for prompting the model."""
        return "".join(str(self.board[r][c]) for r in range(9) for c in range(9))


# --------------------------------------------------------------------------- #
# Puzzle generator                                                             #
# --------------------------------------------------------------------------- #

def _solve(board: list[list[int]]) -> bool:
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                random_digits = list(range(1, 10))
                random.shuffle(random_digits)
                for num in random_digits:
                    # inline constraint check (no SudokuGame overhead)
                    if (num not in board[r] and
                            num not in [board[row][c] for row in range(9)] and
                            num not in [board[(r//3)*3+dr][(c//3)*3+dc]
                                        for dr in range(3) for dc in range(3)]):
                        board[r][c] = num
                        if _solve(board):
                            return True
                        board[r][c] = 0
                return False
    return True


def generate_puzzle(num_clues: int = 30, seed: Optional[int] = None) -> tuple[list[list[int]], list[list[int]]]:
    """
    Returns (puzzle, solution). puzzle has num_clues filled cells; rest are 0.
    num_clues should be in [17, 81]; fewer clues = harder puzzle.
    """
    if seed is not None:
        random.seed(seed)

    board = [[0] * 9 for _ in range(9)]
    _solve(board)
    solution = copy.deepcopy(board)

    # remove cells to create puzzle
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    to_remove = 81 - num_clues
    for r, c in cells[:to_remove]:
        board[r][c] = 0

    return board, solution


# --------------------------------------------------------------------------- #
# CLI helper — parse model output                                              #
# --------------------------------------------------------------------------- #

import re

_MOVE_RE = re.compile(r"<(\d+),(\d+),(\d+)>")


def parse_move(text: str) -> Optional[tuple[int, int, int]]:
    """Extract first <x,y,num> triple from model output. Returns None if not found."""
    m = _MOVE_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


# --------------------------------------------------------------------------- #
# Quick smoke test                                                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    puzzle, solution = generate_puzzle(num_clues=30, seed=42)
    game = SudokuGame(puzzle, solution)
    print("Initial board:")
    print(game.render())
    print()

    # Try a few moves
    tests = [
        (1, 1, 5),   # might be starter — should be rejected
        (1, 1, 5),   # duplicate
        (99, 1, 5),  # out of range
        (1, 1, 0),   # invalid digit
    ]
    for x, y, num in tests:
        result = game.apply_move(x, y, num)
        print(f"<{x},{y},{num}> → {result}")

    print(f"\nEmpty cells: {game.count_empty()}")
    print(f"Flat: {game.to_string_flat()}")
