# A08 — Vision Transformer Fine-tuning and Attention Map Visualisation

## Overview

Vision Transformers (ViTs) apply the transformer architecture — built for sequences — to images by treating image patches as tokens. This was considered impractical until Dosovitskiy et al. (2020) showed that with sufficient data (or pretraining), ViTs match or exceed CNNs on image classification.

This assignment fine-tunes DeiT-3-Small on the EuroSAT dataset, visualises what the model attends to via CLS token attention maps, and directly compares these against the Grad-CAM maps you produced in A03. The contrast is instructive: CNNs are biased toward local texture features; ViTs can attend globally.

---

## Theory

### From CNNs to patches

A CNN processes images through a hierarchy of local receptive fields — each neuron only sees a small neighbourhood. Long-range dependencies require many layers of composition. A Vision Transformer replaces this inductive bias with global attention from the very first layer.

An image $I \in \mathbb{R}^{H \times W \times 3}$ is divided into $N = (H/p) \times (W/p)$ non-overlapping patches of size $p \times p$. Each patch is flattened to a vector of dimension $p^2 \cdot 3$ and linearly projected to the model dimension $d$:

$$z_i = E \cdot \text{flatten}(\text{patch}_i) + b$$

where $E \in \mathbb{R}^{d \times p^2 C}$ is the patch embedding matrix. For a $224 \times 224$ image with $p = 16$: $N = 196$ patches.

### The CLS token

A special learnable token $\mathbf{z}_{\text{CLS}} \in \mathbb{R}^d$ is prepended to the patch sequence (borrowed from BERT). After $L$ transformer layers, the output at the CLS position is used as the global image representation, passed through a classification head:

$$\hat{y} = W_{\text{head}} \cdot \mathbf{z}_{\text{CLS}}^{(L)} + b$$

The CLS token has no intrinsic meaning — it learns to aggregate information from all patches through attention. Because it attends to all patches equally (no spatial bias), the model learns a task-adaptive aggregation.

### DeiT-3

DeiT (Data-efficient Image Transformers) was introduced to show that ViTs can be trained effectively on ImageNet alone (without additional massive datasets). DeiT-3 (Touvron et al., 2022) is the third iteration, with improved training recipes including 3-Augment and binary cross-entropy loss for stronger regularisation.

DeiT-3-Small has:
- Patch size: 16, image size: 224×224, $N = 196$ patches
- Depth: 12 transformer blocks
- Heads: 6
- $d_{\text{model}}$: 384
- Parameters: ~22M

It is available in `timm` as `'deit3_small_patch16_224'`.

### Adapting for EuroSAT

EuroSAT images are $64 \times 64$. Two approaches:

1. **Resize to 224×224** (recommended): upscale images. Some texture information is lost but the pretrained patch embeddings remain valid.
2. **Retrain patch embeddings**: keep 64×64 images with smaller patch size (e.g., $p = 8$, giving $N = 64$ patches). The patch embedding matrix must be reinitialised.

Use approach 1 for simplicity. The classification head `model.head` must be replaced with a new `nn.Linear(384, 10)`.

### Attention rollout and CLS attention maps

To understand where the model looks, we can extract the attention weights from the last transformer block. Specifically, the attention from the CLS token ($\text{row } 0$ of the attention matrix) to all patch positions tells us which patches the model weighted most when computing the CLS representation.

For head $h$ in the last layer:

$$\alpha_h = A_h[0, 1:N+1] \in \mathbb{R}^N$$

(index 0 is the CLS token, indices $1..N$ are patch tokens). Average over heads, reshape to a spatial grid, and upsample to the original image size.

A more principled approach is **attention rollout** (Abnar & Zuidema, 2020), which propagates attention weights through all layers by matrix multiplication:

$$\hat{A}^{(l)} = A^{(l)} + I \quad \text{(add identity for residual)}$$
$$\text{Rollout} = \hat{A}^{(1)} \cdot \hat{A}^{(2)} \cdots \hat{A}^{(L)}$$

This accounts for the fact that information flows through residual connections as well as attention.

For this assignment, the last-layer CLS attention approach is sufficient. Rollout is a bonus exploration.

### ViT vs CNN: what do they attend to?

CNNs (Grad-CAM from A03) attend to local regions of high discriminative texture. A Forest classifier attends to the dense tree-canopy texture patches. A Highway classifier attends to the linear road structure.

ViTs can attend globally and to non-contiguous regions simultaneously. For the same images, you will likely see the CLS attention spread over larger regions, sometimes including context that a CNN ignores. Neither is strictly better — it depends on the task structure.

---

## Reading Material

- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020): https://arxiv.org/abs/2010.11929
- Touvron et al., "DeiT III: Revenge of the ViT" (2022): https://arxiv.org/abs/2204.07118
- Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (attention rollout, 2020): https://arxiv.org/abs/2005.00928
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO — beautiful attention maps): https://arxiv.org/abs/2104.14294
- timm documentation: https://huggingface.co/docs/timm/index

---

## Assignment

### Dataset

EuroSAT RGB (same as A02/A03). Resize images to 224×224 for DeiT-3.

### Task

**Part 1 — Fine-tuning:** Load pretrained `deit3_small_patch16_224` from `timm`. Replace the classification head. Fine-tune for 10 epochs with Adam at $lr = 1 \times 10^{-4}$, batch size 32. Use early stopping on validation ROC-AUC. Report macro F1, ROC-AUC, accuracy on test set. Compare against A03's SE-ResNet-18.

**Part 2 — Attention maps (bonus, compulsory):** For each of the 10 EuroSAT classes, extract CLS attention maps from the last transformer block (averaged over heads). Visualise overlaid on the original image. Produce a 2×5 figure. Alongside this, place the corresponding Grad-CAM from A03 for the same images. Discuss the visual differences in your `notes.md`.

### What to implement

1. Data loading with 224×224 resize and ImageNet normalisation
2. DeiT-3 fine-tuning with early stopping
3. `AttentionMapExtractor` using PyTorch forward hooks on the last attention block
4. Overlay visualisation

### Deliverables

- [ ] `train.py` — fine-tuning with early stopping
- [ ] `evaluate.py` — full metrics report
- [ ] `attention_extractor.py` — hook-based attention map extraction
- [ ] `attention_grid.png` — 2×5 CLS attention maps
- [ ] `comparison_grid.png` — side-by-side Grad-CAM (A03) vs ViT attention for same images
- [ ] `notes.md` — qualitative analysis of CNN vs ViT attention differences
- [ ] Test accuracy ≥ 94%

---

## Starter Code

```python
# model.py  — fill in TODOs
import timm
import torch.nn as nn

def build_deit3(num_classes: int = 10, pretrained: bool = True):
    model = timm.create_model(
        'deit3_small_patch16_224',
        pretrained=pretrained,
        num_classes=0  # Remove head — we'll add our own
    )
    # TODO: add a new classification head
    # model.head = nn.Linear(model.num_features, num_classes)
    return model
```

```python
# attention_extractor.py  — fill in TODOs
import torch
import torch.nn as nn
import numpy as np

class AttentionMapExtractor:
    """
    Extracts CLS token attention weights from the last transformer block.
    Uses forward hooks to intercept attention weights without modifying the model.
    """
    def __init__(self, model):
        self.model = model
        self.attention_weights = None
        # TODO: register a hook on the last attention block
        # For DeiT-3 in timm, attention is at model.blocks[-1].attn
        # You need to hook the softmax output inside the attention computation.
        # Hint: hook model.blocks[-1].attn.attn_drop or override the attn module

    def extract(self, x):
        """
        x: (1, 3, 224, 224)
        Returns: attention map of shape (H, W) normalised to [0, 1]
        """
        # TODO: forward pass, extract self.attention_weights
        # attention_weights shape: (1, num_heads, N+1, N+1) where N=196
        # Take row 0 (CLS token), columns 1: (patch tokens)
        # Average over heads, reshape to (14, 14), upsample to (224, 224)
        raise NotImplementedError
```

```python
# verify.py — sanity check before fine-tuning
import torch
from model import build_deit3

model = build_deit3(num_classes=10)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")

dummy = torch.randn(2, 3, 224, 224)
out = model(dummy)
print(f"Output shape: {out.shape}")  # Should be (2, 10)
```

---

## Notes

The timm library's attention modules do not expose attention weights by default. The cleanest approach is to subclass or monkey-patch the last block's attention module to save the softmax output. A simpler hack: register a forward hook on the `attn_drop` layer (identity when dropout=0) which receives the attention weights.

For the comparison figure, use the exact same test images you used for Grad-CAM in A03. This makes the visual comparison fair. You will likely find that Grad-CAM is more localised (picks a specific discriminative patch) while ViT attention is more diffuse — ViT has global context from layer 1, so it does not need to trace gradients back to a specific spatial location.
