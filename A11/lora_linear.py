"""LoRA linear layer skeleton for A11."""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA wrapper for a frozen linear projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        bias: bool = False,
    ) -> None:
        """Initialize base weight and low-rank adapter matrices.

        Learner task:
            Initialize A and B with LoRA conventions.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.weight = nn.Parameter(torch.empty(out_features, in_features),
                                   requires_grad=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias
            else None
        )
        # TODO: define LoRA A and B trainable matrices.
        raise NotImplementedError("Implement LoRA A/B initialization.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frozen linear layer plus LoRA low-rank update."""
        # TODO: compute base output + scaled LoRA update.
        raise NotImplementedError("Implement LoRALinear.forward.")

    def merge_weights(self) -> None:
        """Merge LoRA update into frozen base weight for inference."""
        # TODO: add delta weight to self.weight and optionally disable adapters.
        raise NotImplementedError("Implement merge_weights.")
