# A01 — PyTorch Training Loop from Scratch

## Overview

Before you touch a pretrained model, a high-level trainer, or any framework abstraction, you need to know what actually happens during training. This assignment strips everything away. You will build a convolutional neural network and the entire surrounding machinery — data loading, forward pass, loss computation, gradient flow, parameter update, and evaluation — using only raw PyTorch. No PyTorch Lightning, no Hugging Face Trainer, no `model.fit()`.

The task is handwritten digit classification on MNIST. The dataset is intentionally trivial so that debugging is easy and training is fast. The point is not the problem — it is the pipeline.

---

## Theory

### What is a neural network doing?

A neural network is a parameterised function $f_\theta : \mathcal{X} \to \mathcal{Y}$ that maps inputs $x \in \mathcal{X}$ to predictions $\hat{y} \in \mathcal{Y}$. The parameters $\theta$ are the weights and biases of every layer. Training is the process of finding $\theta^*$ such that $f_{\theta^*}$ makes good predictions on unseen data.

We formalise "good predictions" with a loss function $\mathcal{L}(\hat{y}, y)$ that measures how wrong the prediction is relative to the true label $y$. Training minimises the empirical risk:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f_\theta(x_i), y_i)$$

### Convolutions

A 2D convolution applies a small learnable filter $W \in \mathbb{R}^{k \times k}$ to a feature map $X$ by sliding it across spatial positions and computing dot products:

$$(X * W)[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X[i+m,\ j+n] \cdot W[m, n]$$

The key insight is **weight sharing**: the same filter is applied everywhere, so the network can detect the same feature (an edge, a curve) regardless of where it appears. This is translation equivariance, and it is why CNNs are so much more parameter-efficient than fully-connected networks on image data.

For a layer with $C_{out}$ output channels and $C_{in}$ input channels, the total parameter count is $C_{out} \times C_{in} \times k \times k + C_{out}$ (the last term being biases). Compare this to a fully connected layer mapping a $28 \times 28$ image to 128 units: $784 \times 128 = 100{,}352$ parameters. A $3 \times 3$ conv with 1 input and 32 output channels: $32 \times 1 \times 9 = 288$ parameters.

### Activation functions

After each linear operation (conv or fully connected), a non-linearity is applied. Without it, stacking layers is mathematically equivalent to a single linear transformation. The most common choice is ReLU:

$$\text{ReLU}(x) = \max(0, x)$$

It is cheap to compute, does not saturate for positive inputs (avoiding vanishing gradients), and works well in practice. Its derivative is either 0 (for $x < 0$) or 1 (for $x > 0$), which is exactly what makes backpropagation efficient through it.

### Cross-entropy loss

For a $C$-class classification problem, the network outputs a vector of logits $z \in \mathbb{R}^C$. The softmax converts these to probabilities:

$$p_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$$

Cross-entropy loss for a single example with true class $y$ is:

$$\mathcal{L}_{\text{CE}} = -\log p_y = -z_y + \log \sum_{j=1}^{C} e^{z_j}$$

Intuitively: we want $p_y$ to be close to 1, so we maximise $\log p_y$, equivalently minimise $-\log p_y$. The gradient $\partial \mathcal{L} / \partial z_c = p_c - \mathbb{1}[c = y]$ is clean and well-behaved.

In PyTorch, `nn.CrossEntropyLoss` combines softmax and cross-entropy numerically stably. **Do not apply softmax yourself before passing to this loss.**

### Backpropagation

Backpropagation is the chain rule applied recursively through the computation graph. For a composed function $\mathcal{L} = h(g(f(x)))$:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial g} \cdot \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial x}$$

PyTorch builds this graph dynamically as operations execute. Calling `.backward()` on the scalar loss triggers reverse-mode automatic differentiation, populating `.grad` on every tensor that has `requires_grad=True`.

### Stochastic gradient descent and Adam

Vanilla SGD updates parameters as:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

where $\eta$ is the learning rate. In practice, gradients are computed on a mini-batch rather than the full dataset (hence "stochastic").

Adam (Adaptive Moment Estimation) maintains per-parameter running estimates of the first moment (mean gradient) and second moment (uncentred variance):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

with bias-corrected estimates $\hat{m}_t = m_t / (1 - \beta_1^t)$ and $\hat{v}_t = v_t / (1 - \beta_2^t)$. The update is:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

The per-parameter adaptive learning rate makes Adam far more robust to hyperparameter choice than plain SGD, which is why it is the default for most deep learning work.

### Why this works (intuition)

The loss surface of a neural network is high-dimensional and non-convex. Despite this, gradient descent reliably finds good solutions in practice. The prevailing explanation is that in very high dimensions, most critical points are saddle points rather than local minima, and the loss at local minima tends to be close to the global minimum. The stochasticity of mini-batch gradients also helps escape shallow basins.

---

## Reading Material

- LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998) — the original CNN paper on digit recognition: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014): https://arxiv.org/abs/1412.6980
- CS231n Lecture Notes on CNNs: https://cs231n.github.io/convolutional-networks/
- PyTorch documentation on `autograd`: https://pytorch.org/docs/stable/notes/autograd.html

---

## Assignment

### Dataset

MNIST — 60,000 training and 10,000 test grayscale images of handwritten digits ($28 \times 28$, 10 classes). Available directly via `torchvision.datasets.MNIST`.

### Task

Build a CNN classifier for MNIST using only `torch`, `torch.nn`, `torch.optim`, and `torchvision` for data loading. Every other component — the training loop, metric computation, logging — must be written by you.

### Architecture requirements

Your network must contain:
- At least 2 convolutional layers with ReLU activations
- At least 1 max-pooling layer
- At least 1 fully connected layer
- A final linear layer outputting 10 logits

### Training requirements

- Loss: `nn.CrossEntropyLoss`
- Optimiser: Adam, learning rate $1 \times 10^{-3}$
- Batch size: 64
- Epochs: 10
- Track training loss and validation accuracy at each epoch

### What to implement

1. A `Dataset`-compatible wrapper (or use `torchvision` directly, but understand what it does)
2. A `nn.Module` subclass for your CNN
3. A training loop with:
   - `optimizer.zero_grad()` → forward pass → loss → `.backward()` → `optimizer.step()`
   - Per-epoch validation accuracy computation
4. A test-set evaluation function
5. Loss and accuracy curves saved as a figure

### Deliverables

- [ ] `model.py` — CNN definition
- [ ] `train.py` — training loop, saves best model checkpoint
- [ ] `evaluate.py` — loads checkpoint, reports test accuracy
- [ ] `loss_curve.png` — training loss over epochs
- [ ] `accuracy_curve.png` — validation accuracy over epochs
- [ ] Final test accuracy ≥ 98% (this is very achievable; if you are below 95%, something is wrong)

### Sanity checks

- Print parameter count before training: `sum(p.numel() for p in model.parameters())`
- Verify loss at epoch 0 is close to $\ln(10) \approx 2.3$ — random predictions on 10 classes should give this
- Verify gradients are non-None after first `.backward()` call

---

## Starter Code

```python
# download_data.py
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_set = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
print(f"Image shape: {train_set[0][0].shape}")  # Should be torch.Size([1, 28, 28])
```

```python
# model.py  — fill in the TODOs
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define layers
        # self.conv1 = ...
        # self.conv2 = ...
        # self.fc    = ...
        pass

    def forward(self, x):
        # TODO: define forward pass
        # x: (B, 1, 28, 28)
        raise NotImplementedError
```

```python
# train.py  — fill in the TODOs
import torch
from torch.utils.data import DataLoader, random_split

def train(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            # TODO: zero_grad → forward → loss → backward → step
            pass

        # TODO: validation loop
        # TODO: print epoch summary
        pass

if __name__ == '__main__':
    # TODO: build dataset, split, call train()
    pass
```

---

## Notes

The 98% threshold is not arbitrary — MNIST is a solved problem. If you cannot hit it, the bug is almost certainly one of: forgetting `optimizer.zero_grad()`, not calling `model.eval()` during validation (which affects dropout/batchnorm if present), or applying softmax before `CrossEntropyLoss`. Fix those three and you will be fine.

Do not move to A02 until you fully understand what `loss.backward()` is doing and why `optimizer.zero_grad()` must be called before each batch.
