# A02 — Transfer Learning and ResNet Fine-tuning

## Overview

Training deep networks from random initialisation requires enormous amounts of data and compute. Transfer learning sidesteps this: take a model trained on a large dataset (ImageNet), whose weights already encode a rich hierarchy of visual features, and adapt it to your smaller target task by continuing to train on your data.

This assignment teaches you what fine-tuning actually means mechanically — which layers to freeze, which to update, why it works, and what the failure modes look like. The dataset switches from MNIST to EuroSAT, a satellite land-use classification benchmark, to make the domain shift interesting.

---

## Theory

### What does a pretrained network learn?

A deep CNN trained on ImageNet learns a hierarchy of features. Early layers learn low-level patterns (edges, colour blobs, textures). Middle layers learn mid-level structures (corners, curves, simple shapes). Deep layers learn high-level semantic concepts (wheels, faces, fur). This has been verified empirically by visualising what maximally activates each filter.

The critical observation: early-layer features are nearly universal across vision tasks. An edge detector useful for classifying cats is also useful for classifying satellite imagery. Only the final, task-specific layers need to be relearned.

### ResNet-18 architecture

ResNet (He et al., 2016) introduced residual connections to enable training of very deep networks. The core idea is the residual block:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

where $\mathcal{F}$ is a stack of convolutions and $\mathbf{x}$ is the identity shortcut. The gradient can flow directly through the shortcut:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)$$

The $+1$ term ensures gradients cannot vanish to zero even if $\partial \mathcal{F} / \partial \mathbf{x}$ is small. This is why ResNets of 100+ layers train reliably where plain networks fail.

ResNet-18 has 4 residual stages (`layer1` through `layer4`) with channel widths 64, 128, 256, 512, followed by global average pooling and a linear classifier. Total: ~11M parameters.

### Why does ImageNet pretraining help on satellite imagery?

ImageNet contains 1.2M images across 1000 classes. A model trained on it has learned:
- How to detect textures (grass, water, asphalt) — directly useful for land-use
- How to aggregate local evidence into global predictions via pooling
- A good weight initialisation that sits in a favourable region of loss space

EuroSAT has only 27,000 images across 10 classes. Training ResNet-18 from scratch on this would overfit severely. Starting from ImageNet weights, the model already "knows" most of what it needs.

### Freezing vs fine-tuning

**Frozen backbone:** Set `requires_grad=False` on all backbone parameters. Only the new classification head trains. This is fast and low-risk — you cannot destroy the pretrained features. Appropriate when your dataset is very small or very similar to ImageNet.

**Full fine-tuning:** All parameters update. The backbone adjusts to your domain. Better final performance but risks catastrophic forgetting of useful features if your learning rate is too high.

**Layer-wise learning rates:** A practical middle ground — use a lower learning rate for early layers (they need small corrections) and a higher rate for later layers (they need larger domain adaptation). This is not required for this assignment but is worth understanding.

To freeze layers in PyTorch:
```python
for param in model.parameters():
    param.requires_grad = False
# Then unfreeze the head:
for param in model.fc.parameters():
    param.requires_grad = True
```

### Adapting the classifier head

ImageNet has 1000 classes. EuroSAT has 10. The final `model.fc` layer must be replaced:

```python
in_features = model.fc.in_features  # 512 for ResNet-18
model.fc = nn.Linear(in_features, num_classes)
```

The new layer has `requires_grad=True` by default. Its weights are randomly initialised, so it will have high loss initially — that is expected.

### Batch Normalisation behaviour during fine-tuning

ResNet-18 contains Batch Normalisation (BatchNorm) layers. These behave differently in `train()` and `eval()` modes:
- `train()`: statistics (mean, variance) are computed from the current batch and running statistics are updated
- `eval()`: running statistics accumulated during training are used, batch statistics are ignored

When fine-tuning a frozen backbone, a subtle bug is leaving BatchNorm layers in `train()` mode in the frozen layers — the running statistics get corrupted by your new data distribution. Fix: after freezing, call `model.eval()` on the frozen portion, or freeze BatchNorm explicitly.

---

## Reading Material

- He et al., "Deep Residual Learning for Image Recognition" (2016): https://arxiv.org/abs/1512.03385
- Yosinski et al., "How transferable are features in deep neural networks?" (2014): https://arxiv.org/abs/1411.1792
- Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification" (2019): https://arxiv.org/abs/1709.00029
- PyTorch Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

---

## Assignment

### Dataset

EuroSAT RGB — 27,000 images ($64 \times 64$), 10 land-use classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake). This dataset will also be used in A03 and A04.

### Task

Run two experiments and compare them:

**Experiment A — Frozen backbone:** Load pretrained ResNet-18, freeze all layers except `model.fc`, replace `model.fc` with a 10-class head, train for 15 epochs.

**Experiment B — Full fine-tuning:** Same setup, but unfreeze all layers. Use a lower learning rate ($1 \times 10^{-4}$) to avoid destroying pretrained features. Train for 15 epochs.

### Training setup

- Optimiser: Adam
- Loss: `nn.CrossEntropyLoss`
- Batch size: 64
- Split: 70% train / 15% val / 15% test (fix random seed)
- Data augmentation for training: random horizontal flip, random crop with padding

### What to implement

1. Dataset download and split script
2. Data augmentation pipeline using `torchvision.transforms`
3. `FineTuner` class that wraps the model and handles freeze/unfreeze
4. Training loop reusable from A01 (refactor it into a `utils.py`)
5. Evaluation reporting macro F1, accuracy, and per-class accuracy

### Deliverables

- [ ] `dataset.py` — download, split, return `DataLoader`s
- [ ] `model.py` — `FineTuner` class
- [ ] `train.py` — runs both experiments sequentially
- [ ] `evaluate.py` — loads checkpoint, prints full report
- [ ] `results_comparison.txt` — table comparing frozen vs full fine-tune on test set
- [ ] Frozen experiment test accuracy ≥ 85%, full fine-tune ≥ 92%

### Ablation question (written answer in a `notes.md`)

After running both experiments, answer: In Experiment B, which layers moved the most (measure by L2 norm of weight change from initialisation)? Use this to justify why frozen fine-tuning can sometimes match full fine-tuning on small datasets.

---

## Starter Code

```python
# download_data.py
import os
import urllib.request
import zipfile

# EuroSAT RGB download
URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
DEST = "./data/EuroSAT.zip"

os.makedirs("./data", exist_ok=True)
print("Downloading EuroSAT...")
urllib.request.urlretrieve(URL, DEST)
with zipfile.ZipFile(DEST, 'r') as z:
    z.extractall("./data/")
print("Done. Classes:", os.listdir("./data/EuroSAT/2750"))
```

```python
# model.py  — fill in TODOs
import torch.nn as nn
import torchvision.models as models

class FineTuner(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        if freeze_backbone:
            # TODO: freeze all backbone parameters
            pass

        # TODO: replace the final fc layer
        # in_features = self.backbone.fc.in_features
        # self.backbone.fc = ...
        pass

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_all(self):
        # TODO: set requires_grad=True for all parameters
        pass
```

```python
# verify.py — sanity check before training
import torch
from model import FineTuner

model = FineTuner(num_classes=10, freeze_backbone=True)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params (frozen): {trainable:,} / {total:,}")

model2 = FineTuner(num_classes=10, freeze_backbone=False)
trainable2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f"Trainable params (full):   {trainable2:,} / {total:,}")

# Quick forward pass check
dummy = torch.randn(4, 3, 64, 64)
out = model(dummy)
print(f"Output shape: {out.shape}")  # Should be (4, 10)
```

---

## Notes

The gap between frozen and full fine-tuning on EuroSAT is typically 5–8%. If you see frozen outperforming full fine-tuning, your learning rate in Experiment B is too high — the pretrained features are being overwritten. Drop it to $1 \times 10^{-5}$ and re-run.

Normalise your inputs using ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) even though EuroSAT is not ImageNet. The pretrained model expects this range.
