# A06 — Attention Mechanism from Scratch

## Overview

Before fine-tuning Vision Transformers or using pretrained language models, you need to understand what the attention operation actually computes. Not conceptually — mathematically and mechanically. This assignment has one goal: implement multi-head self-attention from first principles, add causal masking for autoregressive use, and verify your implementation numerically against PyTorch's built-in `nn.MultiheadAttention`. No training involved. Just the mechanism.

If you cannot build attention from scratch and explain every dimension transformation, you are not ready to debug transformer code — and transformer bugs are notoriously silent (wrong shapes silently broadcast, attention over wrong dimensions, masking applied to wrong positions).

---

## Theory

### Queries, Keys, and Values

Attention takes three inputs: queries $Q$, keys $K$, and values $V$. Intuitively:
- A **query** is "what I'm looking for"
- A **key** is "what I have to offer"
- A **value** is "what I'll actually contribute if selected"

For self-attention, all three come from the same sequence. Given an input sequence $X \in \mathbb{R}^{L \times d_{\text{model}}}$ (length $L$, dimension $d_{\text{model}}$), we project:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ are learned projection matrices.

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Step by step:**

1. $QK^\top \in \mathbb{R}^{L \times L}$: the $(i, j)$ entry measures how much position $i$'s query aligns with position $j$'s key. High dot product = high alignment.

2. Scale by $1/\sqrt{d_k}$: without this, the dot products grow with dimension $d_k$, pushing softmax into saturation where gradients vanish. For $d_k$-dimensional random unit vectors, the expected dot product is $\sqrt{d_k}$, so dividing restores unit variance.

3. Softmax: converts the score vector for each query position into a probability distribution over all key positions. Each row of the result sums to 1.

4. Multiply by $V$: the output at position $i$ is a weighted sum of value vectors, with weights determined by how much position $i$'s query matched each key.

### Multi-head attention

A single attention head looks at the sequence from one "perspective". Multi-head attention runs $h$ parallel attention operations, each with different learned projections:

$$\text{head}_k = \text{Attention}(Q W^Q_k,\, K W^K_k,\, V W^V_k)$$

The heads are concatenated and projected back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

Typically $d_k = d_v = d_{\text{model}} / h$, so the total compute is the same as single-head attention with $d_{\text{model}}$ dimensions.

**Why multiple heads?** Different heads learn to attend to different aspects — one might attend to syntactic relationships, another to semantic co-occurrence, another to positional proximity. The model can simultaneously attend to multiple positions for different reasons.

### Causal masking

In a language model, position $i$ should only attend to positions $j \leq i$ (it cannot look into the future). This is enforced by adding a mask to the attention scores before softmax:

$$S_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_k}} + M_{ij}$$

where $M_{ij} = 0$ if $j \leq i$ and $M_{ij} = -\infty$ if $j > i$. After softmax, $e^{-\infty} = 0$, so future positions receive zero attention weight.

In PyTorch, $-\infty$ is approximated by a very large negative number, e.g., `-1e9` or `float('-inf')`.

The mask is an upper-triangular matrix of $-\infty$ values (excluding the diagonal):

```
M = [[  0, -inf, -inf, -inf],
     [  0,    0, -inf, -inf],
     [  0,    0,    0, -inf],
     [  0,    0,    0,    0]]
```

### Attention complexity

The attention matrix $QK^\top$ has $L^2$ entries. Computing it requires $O(L^2 d)$ operations. For long sequences, this is the bottleneck — quadratic in sequence length. This is why transformers with long contexts are expensive, and why linear attention approximations (Performer, Flash Attention) are an active research area.

### Numerical stability in softmax

Naive softmax $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$ overflows for large $x_i$. The safe version subtracts the maximum:

$$\text{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}$$

This is mathematically equivalent but numerically stable. Always use this in practice (PyTorch's built-in softmax does this automatically, but you should know why).

---

## Reading Material

- Vaswani et al., "Attention Is All You Need" (2017): https://arxiv.org/abs/1706.03762
- Alammar, "The Illustrated Transformer" (blog, highly recommended visual explanation): https://jalammar.github.io/illustrated-transformer/
- Flash Attention (efficient attention, good to read after understanding naive): https://arxiv.org/abs/2205.14135
- PyTorch `nn.MultiheadAttention` source: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/attention.py

---

## Assignment

### Dataset

None. This is a purely implementation and verification assignment.

### Task

**Part 1 — Scaled dot-product attention:** Implement the function `scaled_dot_product_attention(Q, K, V, mask=None)`. Verify it matches `torch.nn.functional.scaled_dot_product_attention` (available from PyTorch 2.0+) on random inputs.

**Part 2 — Multi-head attention:** Implement `MultiHeadAttention` as an `nn.Module` with weight matrices $W^Q_k, W^K_k, W^V_k$ for each head and an output projection $W^O$. Verify it matches `nn.MultiheadAttention` numerically (after copying weights).

**Part 3 — Causal masking:** Add an `is_causal` flag. When enabled, construct and apply the upper-triangular mask. Verify that positions in the output at time $t$ do not depend on any input at time $t' > t$ (test by perturbing future positions and checking that the output at $t$ does not change).

**Part 4 — Attention visualisation:** Pass a short hand-crafted sequence (e.g., 8 tokens of random embeddings) through your multi-head attention. Visualise the attention weight matrix for each head as a heatmap. With causal masking, the upper triangle should be exactly zero.

### What to implement

1. `attention.py` — `scaled_dot_product_attention` function
2. `mha.py` — `MultiHeadAttention` module
3. `verify.py` — numerical comparison against PyTorch built-ins
4. `visualise.py` — attention heatmap plots

### Deliverables

- [ ] `attention.py`
- [ ] `mha.py`
- [ ] `verify.py` — all assertions pass, max absolute error < 1e-5
- [ ] `attn_weights_no_mask.png` — attention weights heatmap (no causal mask)
- [ ] `attn_weights_causal.png` — attention weights heatmap (with causal mask, upper triangle = 0)
- [ ] `notes.md` — answer: why does removing the $1/\sqrt{d_k}$ scaling hurt training?

---

## Starter Code

```python
# attention.py  — fill in TODOs
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,   # (B, H, L, d_k)
    K: torch.Tensor,   # (B, H, L, d_k)
    V: torch.Tensor,   # (B, H, L, d_v)
    mask: torch.Tensor = None  # (L, L) or (B, H, L, L), additive mask
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        output: (B, H, L, d_v)
        attn_weights: (B, H, L, L)
    """
    d_k = Q.size(-1)
    # TODO: compute scores = Q @ K^T / sqrt(d_k)
    # TODO: add mask if provided
    # TODO: apply softmax
    # TODO: compute output = attn_weights @ V
    raise NotImplementedError


def make_causal_mask(L: int, device=None) -> torch.Tensor:
    """
    Returns additive causal mask of shape (L, L).
    Upper triangle (excluding diagonal) filled with -inf.
    """
    # TODO: construct upper-triangular mask
    raise NotImplementedError
```

```python
# mha.py  — fill in TODOs
import torch
import torch.nn as nn
import math
from attention import scaled_dot_product_attention, make_causal_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: define W_q, W_k, W_v (each d_model → d_model), and W_o (d_model → d_model)
        # Use nn.Linear with bias=False for the projection matrices
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,  # (B, L, d_model)
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (B, L, d_model)
            attn_weights: (B, num_heads, L, L)
        """
        B, L, _ = query.shape
        # TODO: project Q, K, V
        # TODO: reshape to (B, num_heads, L, d_k) — split heads
        # TODO: apply scaled_dot_product_attention with optional causal mask
        # TODO: concatenate heads, apply W_o
        raise NotImplementedError
```

```python
# verify.py — numerical verification
import torch
import torch.nn as nn
from mha import MultiHeadAttention

torch.manual_seed(42)
B, L, d_model, num_heads = 2, 10, 64, 4

our_mha = MultiHeadAttention(d_model, num_heads)
pt_mha  = nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True)

# Copy weights from PyTorch MHA to ours
# PyTorch packs all projection weights as in_proj_weight (shape 3*d_model x d_model)
# and out_proj.weight (shape d_model x d_model)
# TODO: copy weights and verify outputs match within 1e-5

x = torch.randn(B, L, d_model)
with torch.no_grad():
    our_out, our_attn  = our_mha(x, x, x)
    pt_out,  pt_attn   = pt_mha(x, x, x)

print(f"Max output difference: {(our_out - pt_out).abs().max().item():.2e}")
assert (our_out - pt_out).abs().max() < 1e-4, "Outputs do not match!"
print("PASS")
```

---

## Notes

The most common mistake is getting the head-splitting wrong. The reshape from `(B, L, d_model)` to `(B, num_heads, L, d_k)` requires `view` followed by `transpose` or `permute` — not just `view`. If you `view(B, L, num_heads, d_k)`, you need to transpose dims 1 and 2 to get `(B, num_heads, L, d_k)`. Getting this wrong produces correct shapes but numerically wrong outputs because you've scrambled which features belong to which head.

The causal mask perturbation test is a critical sanity check. Implement it: take an input sequence $x$, compute output at position 3, then modify $x[4:]$ to random noise. If the output at position 3 changes, your mask is wrong.
