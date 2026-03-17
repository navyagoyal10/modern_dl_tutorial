# A11 — Parameter-Efficient Fine-Tuning with LoRA

## Overview

Fine-tuning a 7B-parameter language model updates ~7 billion floats. At fp32, that is 28GB just for the gradients — before you account for optimiser states, which Adam doubles. This is why full fine-tuning of large models is inaccessible to anyone without a large cluster.

Low-Rank Adaptation (LoRA) sidesteps this by observing that the weight updates during fine-tuning are empirically low-rank. Instead of updating the full weight matrix $W$, LoRA freezes $W$ and adds a trainable low-rank decomposition $\Delta W = BA$. This reduces trainable parameters by 99%+ while matching full fine-tuning performance on many tasks.

This assignment implements LoRA conceptually, then uses Unsloth (an optimised LoRA library) for practical fine-tuning of a 7B model on instruction-following data.

---

## Theory

### The low-rank hypothesis

When a large pretrained model is fine-tuned on a downstream task, the weight change $\Delta W = W_{\text{fine-tuned}} - W_{\text{pretrained}}$ has been empirically observed to have low intrinsic rank — even though $W$ itself may be full-rank. This suggests that fine-tuning explores a low-dimensional subspace of the full parameter space.

**Intrinsic dimensionality** (Aghajanyan et al., 2020): The effective number of dimensions needed to represent the fine-tuning update is much smaller than the number of parameters. For BERT-base fine-tuned on many tasks, the intrinsic dimensionality is estimated at ~200, despite having 110M parameters.

### LoRA

For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA parameterises the update as:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$.

The forward pass becomes:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

$W_0$ is frozen (no gradients computed or stored). Only $A$ and $B$ are trained.

**Initialisation:** $A$ is initialised with random Gaussian entries, $B$ is initialised to zero. This ensures $\Delta W = 0$ at the start of training, so the model behaviour is identical to the pretrained model at initialisation.

**Scaling:** The output is scaled by $\alpha / r$:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

where $\alpha$ is a hyperparameter (often set equal to $r$). This controls the magnitude of the LoRA update relative to the pretrained weights.

**Parameter count reduction:** For a weight matrix $W \in \mathbb{R}^{4096 \times 4096}$ with rank $r = 16$:
- Full fine-tuning: $4096^2 = 16{,}777{,}216$ trainable parameters per matrix
- LoRA: $(4096 + 4096) \times 16 = 131{,}072$ trainable parameters per matrix
- Reduction: ~128×

A 7B model typically has ~$10^3$ such matrices. Full fine-tuning: ~7B parameters. LoRA with $r=16$ targeting all attention matrices: ~40M parameters. That is a 175× reduction.

### Which matrices to target?

In transformer attention: $W^Q, W^K, W^V, W^O$ are the standard targets. The original LoRA paper focused on $W^Q$ and $W^V$ only, finding $W^K$ and $W^O$ less important. Targeting all four consistently performs better at the cost of slightly more parameters.

Some implementations also add LoRA to the FFN layers. For this assignment, attention matrices only is sufficient.

### Memory savings

Full fine-tuning of a 7B model in fp16:
- Model weights: 14 GB
- Gradients: 14 GB
- Adam optimizer states (first + second moment): 28 GB
- **Total: ~56 GB** — requires multiple A100s

LoRA (only A, B updated):
- Model weights: 14 GB (frozen, no gradient tracking after loading)
- LoRA gradients: ~80 MB
- Adam optimizer states for LoRA: ~160 MB
- **Total: ~15 GB** — fits on a single 16GB GPU

Unsloth further reduces memory by ~40% through kernel fusion, 4-bit quantisation (QLoRA), and gradient checkpointing.

### QLoRA

QLoRA (Dettmers et al., 2023) combines LoRA with 4-bit NormalFloat (NF4) quantisation of the frozen base model weights. The base model is stored at 4 bits (~3.5 GB for 7B parameters), and LoRA adapters are trained in fp16. This enables fine-tuning 7B models on a 12GB GPU.

Unsloth implements QLoRA efficiently. You will use it in this assignment.

---

## Reading Material

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021): https://arxiv.org/abs/2106.09685
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023): https://arxiv.org/abs/2305.14314
- Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" (2020): https://arxiv.org/abs/2012.13255
- Unsloth documentation: https://docs.unsloth.ai
- Alpaca dataset (instruction fine-tuning): https://huggingface.co/datasets/tatsu-lab/alpaca

---

## Assignment

### Dataset

Alpaca-cleaned (52k instruction-following examples): `tatsu-lab/alpaca` from HuggingFace. Formatted as `{instruction, input, output}` triplets.

### Model

`unsloth/Llama-3.2-1B-Instruct` (1B parameters — fits in 6GB VRAM with QLoRA, much more accessible than 7B). If VRAM permits, `unsloth/Llama-3.1-8B-Instruct` is the preferred target.

### Task

**Part 1 — Manual LoRA layer (conceptual):** Implement a `LoRALinear` module that wraps `nn.Linear` and adds the low-rank adapter. Verify that it produces the same output as the base linear layer when $B$ is zero (at initialisation). This is pure PyTorch — no Unsloth.

**Part 2 — LoRA fine-tuning with Unsloth:** Fine-tune the LLaMA model on Alpaca for 1 epoch using Unsloth's LoRA utilities. Target all attention projection matrices ($W^Q, W^K, W^V, W^O$). Use rank $r = 16$, $\alpha = 16$.

**Part 3 — Analysis:**
- Report trainable parameter count vs total parameter count
- Compare outputs (generated text) from the base model vs fine-tuned model for 5 held-out instructions
- Experiment with $r \in \{4, 8, 16, 32\}$ on a small subset (1k examples) and plot validation loss vs rank. This shows the rank-performance tradeoff.

### What to implement

1. `lora_linear.py` — `LoRALinear` module
2. `finetune.py` — Unsloth-based fine-tuning script
3. `compare_outputs.py` — generate from base vs fine-tuned, print side by side
4. `rank_experiment.py` — rank sweep on small subset

### Deliverables

- [ ] `lora_linear.py` — passes zero-init output equality test
- [ ] `finetune.py`
- [ ] `compare_outputs.txt` — 5 instruction comparisons
- [ ] `rank_vs_loss.png` — validation loss for each rank value
- [ ] `param_count_table.txt` — trainable vs total params for each rank
- [ ] `notes.md` — at what rank does performance plateau? Does increasing rank beyond that hurt?

---

## Starter Code

```python
# lora_linear.py  — fill in TODOs
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 16, alpha: float = 16.0, bias: bool = True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base linear layer
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad = False
        if bias:
            self.base.bias.requires_grad = False

        # TODO: define LoRA matrices A (in_features → rank) and B (rank → out_features)
        # A: initialise with Gaussian, B: initialise to zeros
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: base(x) + scaling * B(A(x))
        raise NotImplementedError

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA back into the base weight for inference (no overhead)."""
        # TODO: return a new nn.Linear with W = W_0 + scaling * B @ A
        raise NotImplementedError
```

```python
# verify_lora.py — zero-init test
import torch
from lora_linear import LoRALinear

torch.manual_seed(0)
layer = LoRALinear(64, 128, rank=4, alpha=4.0)
x = torch.randn(8, 64)

with torch.no_grad():
    lora_out = layer(x)
    base_out = layer.base(x)

diff = (lora_out - base_out).abs().max().item()
print(f"Max difference at init: {diff:.6f}")  # Should be 0.0 (B is all zeros)
assert diff < 1e-6, "B is not zero at initialisation!"
print("PASS")
```

```python
# finetune.py  — fill in TODOs
# Requires: pip install unsloth transformers datasets trl

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

MAX_SEQ_LEN = 2048
MODEL_NAME  = "unsloth/Llama-3.2-1B-Instruct"

def format_alpaca(example):
    """Format Alpaca example into instruction prompt."""
    if example['input']:
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {'text': text}

def main(rank: int = 16, output_dir: str = './lora_model'):
    # TODO: load model with FastLanguageModel.from_pretrained
    # TODO: apply LoRA with get_peft_model, targeting q_proj, k_proj, v_proj, o_proj
    # TODO: load and format Alpaca dataset
    # TODO: set up SFTTrainer and train
    # TODO: save the LoRA adapter
    raise NotImplementedError

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='./lora_model')
    args = parser.parse_args()
    main(args.rank, args.output_dir)
```

---

## Notes

Install Unsloth with: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`. The installation is environment-sensitive — use their latest installation instructions from the GitHub README.

The rank experiment (Part 3) is the most intellectually valuable part. The common expectation is "higher rank = better performance". The reality is more nuanced: very high ranks can overfit on small datasets, and performance often plateaus at $r = 16$ or $r = 32$ for instruction-following tasks. The optimal rank depends on task complexity and dataset size.

When merging LoRA weights for inference, `merge_and_unload()` in Unsloth produces a standard model with no inference overhead — the adapter is baked into the weights. This is an important practical consideration: LoRA adds overhead during training only.
