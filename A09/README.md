# A09 — Transformers Without Normalization: Dynamic Tanh (DyT)

## Overview

Layer Normalisation is present in virtually every transformer published in the last seven years. It is so standard that most practitioners treat it as an immutable design choice. Zhu et al. (CVPR 2025) challenged this: they showed that LayerNorm can be replaced with a single element-wise operation — Dynamic Tanh (DyT) — while matching or exceeding normalised performance across a wide range of tasks.

This assignment asks you to implement DyT, replace every LayerNorm in DeiT-3 with it, understand *why* this works (the key empirical observation about LayerNorm's input-output behaviour), and compare training dynamics against the standard normalised model from A08.

---

## Theory

### What does Layer Normalisation do?

LayerNorm normalises each token's feature vector to zero mean and unit variance, then scales and shifts with learned parameters:

$$\text{LN}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta$$

where $\mu = \frac{1}{d}\sum_j x_j$, $\sigma = \sqrt{\frac{1}{d}\sum_j (x_j - \mu)^2 + \epsilon}$, and $\gamma, \beta \in \mathbb{R}^d$ are learned.

LayerNorm stabilises training by preventing activations from exploding or vanishing. Without it, deep transformers are difficult to optimise — the loss landscape is poorly conditioned.

### The key empirical observation

Zhu et al. observed that when you plot the input-output mapping of LayerNorm layers in a trained transformer, the curve strongly resembles an S-shaped (sigmoid-like) function — specifically, a scaled hyperbolic tangent. This is not because LayerNorm is designed this way; it is an emergent property of the learned $\gamma$ and $\beta$ parameters combined with the normalisation step.

The intuition: LayerNorm clamps large activations (via the division by $\sigma$) and allows near-linear behaviour around zero. The tanh function does exactly this — approximately linear near the origin, saturating at large values:

$$\tanh(x) \approx x \quad \text{for small } |x|$$
$$\tanh(x) \to \pm 1 \quad \text{for large } |x|$$

The difference is that LayerNorm's "clamping threshold" adapts to the scale of the current inputs via $\sigma$. DyT replicates this with a learnable scalar $\alpha$ that plays the same role.

### Dynamic Tanh (DyT)

$$\text{DyT}(x) = \tanh(\alpha x)$$

where $\alpha \in \mathbb{R}$ is a learnable scalar (not a vector — just one parameter per DyT layer), initialised to a small value (e.g., 0.5).

The element-wise $\gamma$ and $\beta$ scaling and shifting from LayerNorm are retained (they are cheap and important):

$$\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$$

So the full substitution for `nn.LayerNorm(d)` is:

```python
class DynamicTanh(nn.Module):
    def __init__(self, dim, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
```

**Parameter count:** DyT adds 1 scalar ($\alpha$) per replaced layer, plus it retains the $\gamma$ and $\beta$ vectors. Compared to LayerNorm (which also has $\gamma$ and $\beta$), DyT adds only $L$ extra parameters where $L$ is the number of layers — negligible.

### Why does this work?

Three reasons:

1. **Functional equivalence at convergence:** If the trained LayerNorm maps look like tanh, then tanh is a good approximation to what LayerNorm is learning.

2. **Gradient stability without statistics:** LayerNorm stabilises gradients by normalising activations. DyT achieves similar gradient stability through the saturation of tanh for large activations — large values get clipped to ±1, preventing runaway gradients.

3. **Simpler computation:** LayerNorm requires computing mean and variance across the feature dimension, which involves reduction operations and is relatively expensive. tanh is a single element-wise operation, parallelism-friendly on modern hardware.

**Where it fails:** DyT does not work well in the very early layers of models trained from scratch (without pretraining), where activations can be very large and the adaptive normalisation of LayerNorm is critical. For fine-tuning pretrained models (as in this assignment), the activations are already in a reasonable range, making DyT more suitable.

### $\alpha$ initialisation

The $\alpha$ parameter controls how "sharp" the tanh compression is:
- Large $\alpha$: tanh saturates quickly — strong compression, like a very small $\sigma$ in LayerNorm
- Small $\alpha$: tanh is nearly linear — weak compression, like a large $\sigma$

A good initialisation matches the effective compression of the pretrained LayerNorm. The original paper recommends $\alpha_0 = 0.5$ as a default. You can experiment with values in $\{0.1, 0.5, 1.0\}$ on the validation set.

---

## Reading Material

- Zhu et al., "Transformers without Normalization" (CVPR 2025): https://arxiv.org/abs/2503.10622
- Ba et al., "Layer Normalization" (2016): https://arxiv.org/abs/1607.06450
- Touvron et al., "DeiT III: Revenge of the ViT" (2022): https://arxiv.org/abs/2204.07118
- Reference DyT implementation by the authors: https://github.com/jiachenzhu/DyT

---

## Assignment

### Dataset

EuroSAT RGB (same as A08). Reuse your data loading code with 224×224 resize.

### Task

**Part 1 — Implement DyT:** Write `DynamicTanh` as an `nn.Module` with learnable $\alpha$, $\gamma$, $\beta$.

**Part 2 — Replace LayerNorm in DeiT-3:** Write a utility function `replace_layernorm_with_dyt(model)` that traverses the model's module tree and replaces every `nn.LayerNorm` with a `DynamicTanh` of the same feature dimension. Preserve the pretrained $\gamma$ and $\beta$ values from the original LayerNorm.

**Part 3 — Fine-tune and compare:** Fine-tune the DyT-DeiT-3 model on EuroSAT using the same hyperparameters as A08. Compare convergence curves (loss vs epoch) and final test metrics against A08's standard DeiT-3.

**Part 4 — $\alpha$ analysis:** After training, plot the learned $\alpha$ values across all DyT layers (there are 24 LayerNorm layers in a 12-block DeiT-3: 2 per block). Do early layers learn smaller or larger $\alpha$ than later layers? What does this tell you about the compression behaviour at different depths?

### What to implement

1. `dyt.py` — `DynamicTanh` module
2. `replace_norm.py` — `replace_layernorm_with_dyt(model, alpha_init)` utility
3. `train.py` — fine-tuning (same setup as A08, different model)
4. `alpha_plot.py` — visualisation of learned $\alpha$ values per layer

### Deliverables

- [ ] `dyt.py`
- [ ] `replace_norm.py`
- [ ] `train.py`
- [ ] `convergence_comparison.png` — train/val loss curves: DeiT-3 vs DyT-DeiT-3
- [ ] `alpha_values.png` — learned $\alpha$ per layer after training
- [ ] `metrics_table.txt` — F1, ROC-AUC, accuracy for both models
- [ ] `notes.md` — analysis of $\alpha$ values and discussion of when DyT might fail

---

## Starter Code

```python
# dyt.py  — fill in TODO
import torch
import torch.nn as nn

class DynamicTanh(nn.Module):
    def __init__(self, normalised_shape, alpha_init: float = 0.5):
        """
        Drop-in replacement for nn.LayerNorm.
        normalised_shape: int or tuple (same as LayerNorm's normalised_shape arg)
        """
        super().__init__()
        if isinstance(normalised_shape, int):
            normalised_shape = (normalised_shape,)
        self.normalised_shape = normalised_shape

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        # TODO: initialise gamma and beta matching LayerNorm's behaviour
        # gamma should be ones, beta should be zeros, both shape = normalised_shape
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: apply gamma * tanh(alpha * x) + beta
        raise NotImplementedError
```

```python
# replace_norm.py  — fill in TODO
import torch.nn as nn
from dyt import DynamicTanh

def replace_layernorm_with_dyt(model: nn.Module, alpha_init: float = 0.5) -> nn.Module:
    """
    Recursively replace all nn.LayerNorm layers with DynamicTanh.
    Copies gamma and beta values from the original LayerNorm.
    Returns the modified model (in-place modification).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.LayerNorm):
            # TODO: create DynamicTanh with same shape
            # TODO: copy .weight (gamma) and .bias (beta) from LayerNorm to DyT
            # TODO: replace the module using setattr(model, name, dyt_module)
            pass
        else:
            # TODO: recurse into child modules
            replace_layernorm_with_dyt(module, alpha_init)
    return model
```

```python
# verify_replacement.py — sanity check
import torch
import timm
from replace_norm import replace_layernorm_with_dyt

model = timm.create_model('deit3_small_patch16_224', pretrained=False)

# Count LayerNorm layers before
ln_count_before = sum(1 for m in model.modules() if isinstance(m, torch.nn.LayerNorm))
print(f"LayerNorm layers before: {ln_count_before}")

replace_layernorm_with_dyt(model)

from dyt import DynamicTanh
ln_count_after = sum(1 for m in model.modules() if isinstance(m, torch.nn.LayerNorm))
dyt_count      = sum(1 for m in model.modules() if isinstance(m, DynamicTanh))
print(f"LayerNorm layers after: {ln_count_after}")   # Should be 0
print(f"DyT layers after:       {dyt_count}")        # Should equal ln_count_before

# Verify forward pass still works
dummy = torch.randn(2, 3, 224, 224)
out = model(dummy)
print(f"Output shape: {out.shape}")  # Should be (2, 1000) for default head
```

---

## Notes

The `replace_layernorm_with_dyt` function must be recursive — DeiT-3 has LayerNorm layers nested inside `Block` modules inside `Sequential` containers. A single pass over `model.named_children()` is not enough; you need to descend the full module tree.

When copying pretrained $\gamma$ and $\beta$ weights from LayerNorm to DyT, use `.data.copy_()` to avoid autograd issues:
```python
dyt.gamma.data.copy_(layernorm.weight.data)
dyt.beta.data.copy_(layernorm.bias.data)
```

The $\alpha$ value plot is the most theoretically interesting output. If early layers consistently learn smaller $\alpha$ (weaker compression), it suggests that early DeiT-3 layers process more diverse activation scales — which aligns with the observation that early ViT layers are more texture-sensitive, while later layers are more semantic and stable.
