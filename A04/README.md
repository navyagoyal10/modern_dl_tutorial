# A04 — Contrastive Learning: When Labels Fail You

## Overview

Supervised learning assumes you have labelled data. In many real-world settings, you have enormous amounts of unlabelled images and a small labelled set. Self-supervised contrastive learning learns visual representations without any labels by solving a pretext task: make two augmented views of the same image similar in embedding space, and push apart embeddings of different images.

This assignment implements SimCLR (Chen et al., 2020), a landmark contrastive learning framework. The key intellectual goal is to understand *when* contrastive representations beat supervised ones — specifically when the labelled data is artificially scarce — and to visualise the geometry of learned representations using UMAP.

---

## Theory

### The representation learning problem

In supervised learning, the model $f_\theta$ learns features that are useful specifically for predicting labels. This is a narrow objective — the features may be entangled with spurious correlations in the training labels.

In self-supervised learning, $f_\theta$ learns features by solving a task defined purely from the data structure itself. The hypothesis is that a model forced to be invariant to "irrelevant" augmentations (crops, colour jitter, flips) while distinguishing different images will learn semantically meaningful features.

### SimCLR

Given a mini-batch of $N$ images, SimCLR creates $2N$ augmented views by applying two independent augmentations to each image. Each view passes through:
1. An encoder $f$ (e.g., ResNet-18 without the final classification head)
2. A projection head $g$: a small 2-layer MLP that maps $\mathbf{h} = f(x)$ to $\mathbf{z} = g(\mathbf{h})$

The normalised embeddings $\mathbf{z}$ are used to compute the NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss.

### NT-Xent Loss

For a pair of views $(i, j)$ originating from the same image (a "positive pair"), the loss is:

$$\ell_{i,j} = -\log \frac{\exp\!\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau\right)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp\!\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau\right)}$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} / (\|\mathbf{u}\| \|\mathbf{v}\|)$ is cosine similarity and $\tau$ is a temperature hyperparameter.

The denominator sums over all $2N - 1$ other views in the batch (including the "negative" views from other images). The loss is minimised when the positive pair has maximum similarity relative to all negatives.

The full loss averages over all $2N$ positive pairs:

$$\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2N} \sum_{k=1}^{N} \left[\ell_{2k-1,\, 2k} + \ell_{2k,\, 2k-1}\right]$$

**Temperature $\tau$:** Controls the concentration of the distribution. Low $\tau$ produces sharper distributions — the model must more precisely identify positives among negatives. Typical value: $\tau = 0.07$ to $0.5$.

**Why large batches?** More negatives per batch = better signal. Each image in the batch serves as a negative for every other image. SimCLR with batch size 256 has 510 negatives per positive pair. With batch size 4096, it has 8190. This is why the original paper used 4096 — you are limited to what fits in your VRAM.

### The projection head trick

A key finding in SimCLR is that the linear probe accuracy is higher when evaluated on the encoder output $\mathbf{h}$ than on the projection head output $\mathbf{z}$. The projection head is discarded after pretraining; only the encoder is kept.

The intuition: the projection head learns to discard information that is not needed for the contrastive loss (e.g., augmentation-specific features). The encoder retains broader information that is useful downstream.

### Linear probe evaluation

After contrastive pretraining, the encoder is frozen. A single linear classifier is trained on top of the frozen features using only a fraction of labelled data. This is the gold-standard evaluation for representation quality: if good features are learned, even a linear classifier should separate the classes.

Formally, we solve:

$$\min_{W, b} \sum_{i \in \mathcal{D}_{\text{labelled}}} \mathcal{L}_{\text{CE}}\!\left(W f(x_i) + b,\, y_i\right)$$

with $f$ frozen. Compare this against training a full supervised model with the same labelled fraction — the gap tells you how much the contrastive pretraining has helped.

### Why contrastive beats supervised at low label fractions

With 10% of labels, a supervised ResNet-18 may overfit — it has 11M parameters but only ~2,700 labelled EuroSAT images. The contrastive encoder was trained on all 27,000 images (without labels), so it has seen far more data. Its features are robust because they were forced to be invariant to crops, colour, and flips — properties that correspond to class-irrelevant variation.

### UMAP visualisation

Uniform Manifold Approximation and Projection (UMAP) is a non-linear dimensionality reduction technique. It models the high-dimensional data as a weighted graph of local neighbourhoods and finds a low-dimensional embedding that preserves this topological structure. It is faster and often better than t-SNE at preserving global structure.

For this assignment, you will embed the 512-dimensional encoder outputs of the test set into 2D and colour by class. A good contrastive representation will show distinct, tight clusters; a bad one will look like a cloud.

---

## Reading Material

- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (2020): https://arxiv.org/abs/2002.05709
- He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo, 2020): https://arxiv.org/abs/1911.05722
- Grill et al., "Bootstrap Your Own Latent" (BYOL — contrastive without negatives, 2020): https://arxiv.org/abs/2006.07733
- McInnes et al., "UMAP: Uniform Manifold Approximation and Projection" (2018): https://arxiv.org/abs/1802.03426

---

## Assignment

### Dataset

EuroSAT RGB (same as A02/A03). Contrastive pretraining uses all 27,000 images without labels. Linear probe uses labelled subsets of 1%, 5%, 10%, and 100%.

### Task

**Part 1 — SimCLR pretraining:** Implement NT-Xent loss and contrastive augmentation pipeline. Pretrain a ResNet-18 encoder (output dimension 512) with a 2-layer projection head (512 → 256 → 128) for 100 epochs. Use batch size 256 (reduce to 128 if VRAM is tight — your loss will be slightly noisier).

**Part 2 — Linear probe evaluation:** For labelled fractions {1%, 5%, 10%, 100%}, train a linear classifier on frozen SimCLR features. Also train a full supervised ResNet-18 on the same fractions. Plot accuracy vs label fraction for both methods.

**Part 3 — UMAP visualisation:** Extract encoder embeddings for the full test set. Compute UMAP projection to 2D. Plot coloured by class, for both your SimCLR encoder and a supervised model trained with 10% labels. The visual difference in cluster quality is the payoff of this entire assignment.

### What to implement

1. Contrastive augmentation pipeline (random crop, colour jitter, grayscale, Gaussian blur)
2. `ProjectionHead` MLP
3. `NTXentLoss` module
4. Pretraining loop (no labels used — just images)
5. Linear probe training loop
6. UMAP plotting utility

### Deliverables

- [ ] `augmentations.py` — SimCLR augmentation pipeline
- [ ] `simclr.py` — `ProjectionHead`, `NTXentLoss`, `SimCLR` model wrapper
- [ ] `pretrain.py` — contrastive pretraining loop, saves encoder checkpoint
- [ ] `linear_probe.py` — trains and evaluates linear classifier on frozen features
- [ ] `umap_plot.py` — generates UMAP figure
- [ ] `label_efficiency_curve.png` — accuracy vs label fraction (SimCLR vs supervised)
- [ ] `umap_simclr.png` and `umap_supervised.png`

---

## Starter Code

```python
# augmentations.py  — fill in TODO
import torchvision.transforms as T

def simclr_augmentation(image_size: int = 64):
    """Returns a composition of augmentations for SimCLR."""
    # TODO: compose the following in order:
    # 1. RandomResizedCrop(image_size, scale=(0.2, 1.0))
    # 2. RandomHorizontalFlip()
    # 3. RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
    # 4. RandomGrayscale(p=0.2)
    # 5. RandomApply([GaussianBlur(kernel_size=int(0.1 * image_size) | 1)], p=0.5)
    # 6. ToTensor()
    # 7. Normalize(mean, std)
    raise NotImplementedError


class TwoViewTransform:
    """Applies the augmentation twice to produce two views."""
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)
```

```python
# simclr.py  — fill in TODOs
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        # TODO: 2-layer MLP with BN and ReLU between layers
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: (N, D) — normalised embeddings from two views
        # TODO: concatenate to (2N, D), compute similarity matrix,
        #       mask diagonal, compute loss
        # Hint: torch.cat([z1, z2], dim=0) gives the 2N embeddings
        raise NotImplementedError
```

```python
# verify_loss.py — sanity check for NT-Xent
import torch
from simclr import NTXentLoss

loss_fn = NTXentLoss(temperature=0.5)
N, D = 8, 128
z = torch.randn(N, D)
z = torch.nn.functional.normalize(z, dim=1)

# When both views are identical, loss should be minimal (not zero but low)
loss_same = loss_fn(z, z)
# When both views are random and independent, loss should be near log(2N-1)
z2 = torch.nn.functional.normalize(torch.randn(N, D), dim=1)
loss_rand = loss_fn(z, z2)

import math
print(f"Loss (identical views): {loss_same.item():.4f}")
print(f"Loss (random views):    {loss_rand.item():.4f}")
print(f"Expected upper bound:   {math.log(2*N - 1):.4f}")
```

---

## Notes

SimCLR is sensitive to batch size. With batch size 128 you have only 254 negatives — the loss signal is weaker and you may need more epochs. If accuracy plateaus early, try increasing batch size first before tuning other hyperparameters.

The UMAP plots are the most instructive output of this entire assignment. If your SimCLR clusters look no better than random, your augmentation pipeline is likely wrong — specifically, colour jitter is essential; without it the model trivially distinguishes views by colour statistics.

Use `umap-learn` library: `pip install umap-learn`. Fit UMAP on the training set embeddings, transform the test set.
