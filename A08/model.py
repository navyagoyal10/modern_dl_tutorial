"""DeiT-3 builder skeleton for A08."""

from __future__ import annotations

import timm


def build_deit3(num_classes: int, pretrained: bool = True):
    """Build DeiT-3 model and replace classifier head.

    Learner task:
        Replace default classification head to output num_classes logits.
    """
    model = timm.create_model("deit3_small_patch16_224", pretrained=pretrained)
    # TODO: replace model head with num_classes classifier.
    raise NotImplementedError("Implement DeiT-3 head replacement.")
