# A13 — Reinforcement Learning for LLMs: Teaching Qwen to Solve Sudoku

## Overview

Supervised fine-tuning teaches a model to imitate labelled examples. But for tasks with verifiable correctness — mathematics, code, puzzles — there is a better signal: the answer is either right or wrong, and we can check automatically. Reinforcement learning with verifiable rewards (RLVR) exploits this.

This assignment applies **GRPO** (Group Relative Policy Optimization, DeepSeek-R1) to train a 1.5B parameter LLM to solve Sudoku cells one move at a time. The model outputs `<x,y,num>` — a column, row, and digit — and a game engine validates every move with Sudoku rules. The model learns from reward signals alone: format compliance, move legality, and correctness.

You will implement the Sudoku game engine, the reward functions, and the full GRPO training loop using Unsloth + TRL.

---

## Theory

### Reinforcement Learning from Verifiable Rewards (RLVR)

In standard RLHF, a human preference model scores model outputs. For verifiable tasks, we skip the preference model: the task itself is the judge. Sudoku is ideal — a placement is either consistent with the rules or it is not, and we can check instantly.

The training signal:

$$r(\hat{y}) = \underbrace{r_{\text{format}}}_{\text{has <x,y,num>?}} + \underbrace{r_{\text{valid}}}_{\text{legal Sudoku move?}} + \underbrace{r_{\text{correct}}}_{\text{matches solution?}}$$

Each component is binary and automatically computable. No human labelling required.

### GRPO — Group Relative Policy Optimization

GRPO (Shao et al., 2024; used in DeepSeek-R1) is a policy gradient method designed specifically for LLM training. For each prompt $q$, sample a group of $G$ completions $\{o_1, \ldots, o_G\}$ from the current policy $\pi_\theta$. Compute rewards $\{r_1, \ldots, r_G\}$. The GRPO objective normalises rewards within the group:

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

The policy gradient loss is:

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\sum_{i=1}^G \hat{A}_i \log \pi_\theta(o_i \mid q)\right] + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

Key differences from PPO:
- **No value network** — the baseline is intra-group mean reward, not a learned critic.
- **No reward model** — reward is computed by a pure function (our engine).
- **Group sampling** — diversity within the group is essential for the relative advantage estimate to be informative.

The KL term $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ prevents the model from drifting too far from the pre-trained distribution.

### Why chain-of-thought helps here

Sudoku requires constraint propagation — eliminating candidates across rows, columns, and boxes. Allowing the model to emit reasoning inside `<think>...</think>` before the move gives it "scratchpad" capacity to do this. The GRPO reward only evaluates the final `<x,y,num>` tag, so the model is free to develop any internal reasoning strategy that improves its move accuracy.

### LoRA under GRPO

Full fine-tuning of even a 1.5B model requires ~12GB VRAM per gradient step when using Adam. LoRA (rank 16) reduces trainable parameters to ~0.5% of total, and 4-bit quantisation (QLoRA via Unsloth) of the frozen base reduces memory by ~70%. This makes GRPO training feasible on a 16GB consumer GPU.

### Sudoku as an RL environment

Standard RL environments have:
- **State** — current board configuration
- **Action** — `<x,y,num>` triple
- **Reward** — sparse (only at episode end) or shaped (per-move)

We use shaped per-move rewards because sparse-only rewards with a 51-step horizon (a 30-clue puzzle has 51 empty cells) lead to extreme credit assignment difficulty. The format and validity rewards provide a learning signal even for incorrect placements.

---

## Game Engine

The `SudokuGame` class in `sudoku_engine.py` enforces:

1. **Range check** — $x, y \in [1,9]$, $\text{num} \in [1,9]$
2. **Starter cell lock** — cells given in the original puzzle cannot be overwritten, even by the model
3. **Row constraint** — num must not appear elsewhere in the same row
4. **Column constraint** — num must not appear elsewhere in the same column
5. **Box constraint** — num must not appear elsewhere in the 3×3 sub-box

On failure the engine returns an `ILLEGAL` result with a human-readable reason. On success it returns `OK` and mutates the board. The model **can** overwrite its own prior placements (to allow self-correction), but **cannot** overwrite starter clues.

Move format: `<x,y,num>` where `x` = column (1–9, left→right), `y` = row (1–9, top→bottom).

---

## Reading Material

- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024) — introduces GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025): https://arxiv.org/abs/2501.12948
- Unsloth RL guide: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- TRL GRPOTrainer docs: https://huggingface.co/docs/trl/grpo_trainer
- Hu et al., "LoRA" (2021): https://arxiv.org/abs/2106.09685
- Dettmers et al., "QLoRA" (2023): https://arxiv.org/abs/2305.14314

---

## Assignment

### Model

`unsloth/Qwen2.5-1.5B-Instruct` — 1.5B parameters, fits in ~8GB VRAM with 4-bit QLoRA. Alternatively `unsloth/Llama-3.2-1B-Instruct` for the same VRAM budget.

### Task

Train the model to emit correct `<x,y,num>` moves for partially-filled Sudoku boards using GRPO. The model must learn:
- Format: always output a `<x,y,num>` tag
- Validity: the move must not violate Sudoku rules
- Correctness: the digit must match the puzzle solution

**Part 1 — Game engine:** Implement and test `sudoku_engine.py`. Verify that starter cells are locked, illegal moves are rejected with specific reasons, and legal moves update the board correctly.

**Part 2 — Dataset:** Generate training prompts using `generate_data.py`. Each prompt shows a board state; the ground-truth answer is any one correct move.

**Part 3 — GRPO training:** Run `train.py`. Monitor the three reward components over training. The format reward should saturate quickly (~50 steps); the correctness reward takes longer.

**Part 4 — Evaluation:** Run `evaluate.py` against 20 fresh puzzles. Report:
- % of moves that are legal
- % of moves that match the solution
- % of puzzles fully solved (all empty cells correctly filled)

**Part 5 — Analysis (optional):** Compare the model's chain-of-thought reasoning (inside `<think>` tags) at step 0 vs step 500. Does the model develop a discernible strategy?

### What to implement

1. `sudoku_engine.py` — game engine with `SudokuGame.apply_move(x, y, num) → MoveResult` ✓ (provided)
2. `generate_data.py` — dataset generation ✓ (provided)
3. `train.py` — GRPO training with Unsloth ✓ (provided)
4. `evaluate.py` — evaluation across N fresh puzzles ✓ (provided)
5. `play.py` — interactive human+model play ✓ (provided)

### Deliverables

- [ ] `reward_curves.png` — format / valid / correct reward vs training step
- [ ] `eval_results.txt` — evaluation output from `evaluate.py`
- [ ] `sample_cot.txt` — 3 examples of model chain-of-thought before and after training
- [ ] `notes.md` — Does the model generalise? Where does it fail?

---

## Starter Workflow

```bash
# 1. Generate training data
python generate_data.py

# 2. Train (requires GPU, ~16 GB VRAM)
python train.py

# 3. Evaluate trained model
python evaluate.py --adapter artifacts/grpo_sudoku/lora_adapter --n_games 20

# 4. Play interactively
python play.py --adapter artifacts/grpo_sudoku/lora_adapter

# 5. Play without model (human only, test the engine)
python play.py --no_model
```

---

## Notes

A 30-clue puzzle has 51 empty cells. If the model places each correctly, it wins. But with stochastic decoding and no look-ahead, this is hard. Expect 60–80% correct placement rate after ~500 GRPO steps on a 1.5B model. Full solve rate on easy puzzles (40+ clues) should reach 20–40%.

The key diagnostic: if `correct_reward` grows but `valid_move_reward` does not, the model is memorising solutions without learning the rules. If `valid_move_reward` grows but `correct_reward` does not, the model places legal-but-wrong digits — it needs more training or a higher reward signal for correctness.

GRPO with $G = 6$ generations per prompt means 6× more model forward passes per step than supervised training. Expect ~3× slower wall-clock time vs LoRA SFT with the same batch size.
