# A03 — Architectural Modifications: SE Blocks, Focal Loss, and Grad-CAM

## Overview

You now know how to train and fine-tune a ResNet. This assignment goes one level deeper: you will modify the architecture itself by inserting Squeeze-and-Excitation (SE) blocks after each residual stage, swap the loss function for Focal Loss to handle class imbalance, and then peer inside the model using Grad-CAM to understand what spatial regions it attends to when making predictions.

This is the first assignment where you must read a research paper and translate it into working code. That skill — reading a methods section, identifying the equations, and implementing them — is more important than any specific architecture you will ever learn.

---

## Theory

### Squeeze-and-Excitation blocks

Standard convolutions treat all feature channels equally. An SE block (Hu et al., 2018) recalibrates channel responses by explicitly modelling inter-channel dependencies.

**Squeeze:** Global average pooling collapses each feature map $U_c \in \mathbb{R}^{H \times W}$ to a scalar descriptor:

$$z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} U_c[i, j]$$

The result $\mathbf{z} \in \mathbb{R}^C$ is a channel-wise summary of the entire feature map.

**Excitation:** A small bottleneck network learns per-channel attention weights:

$$\mathbf{s} = \sigma\!\left(W_2\, \delta\!\left(W_1 \mathbf{z}\right)\right)$$

where $W_1 \in \mathbb{R}^{C/r \times C}$, $W_2 \in \mathbb{R}^{C \times C/r}$, $\delta$ is ReLU, $\sigma$ is sigmoid, and $r$ is the reduction ratio (typically 16). This bottleneck forces the network to learn a compact, global representation of channel importance.

**Scale:** The original feature maps are recalibrated by channel-wise multiplication:

$$\tilde{U}_c = s_c \cdot U_c$$

Channels the network deems uninformative are suppressed ($s_c \approx 0$); informative channels are amplified ($s_c \approx 1$).

**Parameter overhead:** For a stage with $C$ channels and reduction ratio $r$, the SE block adds $2C^2/r$ parameters. For ResNet-18's `layer4` with $C=512$ and $r=16$: $2 \times 512^2 / 16 = 32{,}768$ parameters — less than 0.3% of the backbone's total.

**Intuition:** Imagine your convolutional filters detect "water texture", "vegetation texture", and "road texture". For a given image, most of those are irrelevant. The SE block lets the network say: "for this image, upweight the water and vegetation channels, downweight everything else." This is soft, differentiable, learned channel selection.

### Focal Loss

In imbalanced datasets (or any detection task with many easy negatives), standard cross-entropy has a problem: easy, correctly-classified examples generate many small losses that, when summed, dominate the gradient signal. The network spends most of its learning capacity on examples it already handles well.

Focal Loss (Lin et al., 2017) down-weights well-classified examples by multiplying the cross-entropy loss by a modulating factor:

$$\mathcal{L}_{\text{FL}} = -\alpha_t (1 - p_t)^\gamma \log p_t$$

where $p_t$ is the model's predicted probability for the true class, $\gamma \geq 0$ is the focusing parameter, and $\alpha_t$ is a class-balancing weight.

The term $(1 - p_t)^\gamma$ is the key. For easy examples ($p_t \to 1$), this factor $\to 0$, so their contribution to the loss is heavily suppressed. For hard examples ($p_t \to 0$), the factor $\to 1$, so they receive full loss signal.

Setting $\gamma = 0$ recovers standard cross-entropy. Typical values: $\gamma \in \{0.5, 1, 2, 5\}$, with $\gamma = 2$ being the default in the original paper.

**EuroSAT is not severely imbalanced** (all classes have ~2,700 images), so Focal Loss will not provide a dramatic improvement here. The point is to implement it correctly and understand what $\gamma$ trades off — you will see the difference more clearly in A05 on detection.

### Grad-CAM

Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017) produces a heatmap showing which spatial regions of the input the network attends to for a specific class prediction.

For a target class $c$, Grad-CAM computes the gradient of the class score $y^c$ with respect to the feature maps $A^k$ of the last convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

This $\alpha_k^c$ is the importance weight of feature map $k$ for class $c$. The Grad-CAM heatmap is then:

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

The ReLU keeps only features that positively influence the class score. The result is upsampled to the input image resolution and overlaid.

**Why the last conv layer?** It has the highest semantic content (deepest in the network) while retaining spatial resolution before global pooling collapses everything to a vector.

**Intuition:** Grad-CAM asks: "If I perturb feature map $k$ slightly, how much does the score for class $c$ change?" High-importance feature maps in spatially localised regions produce bright spots on the heatmap.

---

## Reading Material

- Hu et al., "Squeeze-and-Excitation Networks" (2018): https://arxiv.org/abs/1709.01507
- Lin et al., "Focal Loss for Dense Object Detection" (2017): https://arxiv.org/abs/1708.02002
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017): https://arxiv.org/abs/1610.02391
- He et al., "Deep Residual Learning for Image Recognition" (2016): https://arxiv.org/abs/1512.03385

---

## Assignment

### Dataset

EuroSAT RGB (same as A02). Reuse your data loading code.

### Task

**Part 1 — SE-ResNet-18:** Insert SE blocks into ResNet-18 after each of `layer1`, `layer2`, `layer3`, `layer4`. Channel widths: 64, 128, 256, 512. Use reduction ratio $r = 16$. Train from the pretrained ResNet-18 backbone. Report macro F1, ROC-AUC, and accuracy on the test set. Compare against your A02 baseline.

**Part 2 — Focal Loss ablation:** Replace `CrossEntropyLoss` with your Focal Loss implementation. Tune $\gamma \in \{0.5, 1.0, 2.0\}$ on the validation set. Report best test metrics.

**Part 3 — Grad-CAM (bonus, compulsory):** For each of the 10 EuroSAT classes, pick one correctly-classified test image. Compute Grad-CAM using the target class score and the output of `layer4`. Overlay the heatmap on the original image. Save as a $2 \times 5$ grid figure.

### What to implement

1. `SEBlock` as an `nn.Module`
2. `SEResNet18` that wraps a pretrained ResNet-18 and inserts SE blocks in its forward pass (do not redefine the whole ResNet — modify the existing forward)
3. `FocalLoss` as an `nn.Module`
4. `GradCAM` utility class using PyTorch hooks

### Deliverables

- [ ] `se_block.py` — `SEBlock` and `SEResNet18` classes
- [ ] `focal_loss.py` — `FocalLoss` class
- [ ] `grad_cam.py` — `GradCAM` utility
- [ ] `train.py` — trains SE-ResNet-18 with CrossEntropy and with FocalLoss
- [ ] `evaluate.py` — full metrics report
- [ ] `gradcam_grid.png` — 2×5 grid of Grad-CAM overlays (one per class)
- [ ] `ablation_table.txt` — comparing CE vs FL with different $\gamma$
- [ ] SE-ResNet-18 test accuracy ≥ 93%

---

## Starter Code

```python
# se_block.py  — fill in TODOs
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # TODO: define squeeze (global avg pool) and excitation (two linear layers)
        # Hint: reduction bottleneck = channels // reduction
        raise NotImplementedError

    def forward(self, x):
        # x: (B, C, H, W)
        # TODO: squeeze → excitation → scale
        raise NotImplementedError


class SEResNet18(nn.Module):
    def __init__(self, num_classes: int, reduction: int = 16):
        import torchvision.models as models
        super().__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')

        # Keep all ResNet layers
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool

        # TODO: create SE blocks for each stage (channels: 64, 128, 256, 512)
        # self.se1 = SEBlock(64, reduction)
        # ...

        # TODO: replace classifier
        # self.fc = nn.Linear(512, num_classes)
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement forward pass with SE blocks inserted after each layer
        raise NotImplementedError
```

```python
# focal_loss.py  — fill in TODO
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # TODO: compute cross-entropy, then apply (1 - p_t)^gamma modulation
        # Hint: use F.cross_entropy with reduction='none' to get per-sample losses
        raise NotImplementedError
```

```python
# grad_cam.py  — fill in TODO
import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # TODO: register forward and backward hooks on target_layer
        # Hint: use .register_forward_hook and .register_full_backward_hook

    def generate(self, x, class_idx=None):
        # x: (1, C, H, W)
        # TODO: forward pass, backward on target class, compute Grad-CAM map
        # Returns: heatmap of shape (H_orig, W_orig) as numpy array in [0, 1]
        raise NotImplementedError
```

---

## Notes

When modifying ResNet's forward pass, do **not** subclass `ResNet` — just copy the backbone's submodules and write your own `forward`. This is cleaner and avoids fighting torchvision's internal structure.

For Grad-CAM, register hooks before the forward pass. Hooks are the idiomatic PyTorch way to intercept intermediate activations without modifying the model architecture.

The Grad-CAM heatmaps for satellite imagery are particularly interesting because the network should focus on texture patches (water surface, crop rows, road lines) rather than object shapes. If your heatmap lights up randomly, check that you are passing the gradient of the correct class score, not the argmax.
