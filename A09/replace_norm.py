"""LayerNorm replacement scaffold for A09."""

from __future__ import annotations

import torch.nn as nn

from dyt import DynamicTanh


def replace_layernorm_with_dyt(model: nn.Module, alpha_init: float) -> nn.Module:
    """Recursively replace LayerNorm modules with DynamicTanh.

    Learner task:
        Walk children recursively, detect LayerNorm, replace with DyT,
        and copy affine parameters where appropriate.
    """
    # TODO: recurse through model.named_children().
    # TODO: when child is nn.LayerNorm, create DynamicTanh replacement.
    # TODO: copy relevant parameters from LayerNorm to DyT.
    # TODO: set replaced module with setattr(parent, name, new_module).
    raise NotImplementedError("Implement recursive LayerNorm replacement.")
