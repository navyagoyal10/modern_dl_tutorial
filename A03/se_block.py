"""SE module and SE-ResNet skeletons for A03."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Input/output shape:
        x: (B, C, H, W)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize SE block components."""
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        # TODO: Define squeeze and excitation submodules.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to feature map."""
        raise NotImplementedError("Implement SE squeeze-excitation forward pass.")


class SEResNet18(nn.Module):
    """ResNet-18 with SE blocks inserted after major stages."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize backbone and placeholders for SE insertion."""
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        # TODO: Create four SE blocks matching layer1..layer4 channel widths.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with TODO-marked SE insertion points."""
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        # TODO: insert SE block after layer1
        x = self.base.layer2(x)
        # TODO: insert SE block after layer2
        x = self.base.layer3(x)
        # TODO: insert SE block after layer3
        x = self.base.layer4(x)
        # TODO: insert SE block after layer4

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        return self.base.fc(x)
