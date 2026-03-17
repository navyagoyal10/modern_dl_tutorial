"""Multimodal classifier skeleton for A12."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from transformers import BertModel


class MultimodalClassifier(nn.Module):
    """Fusion model over DeiT image features and BERT text features."""

    def __init__(
        self,
        num_classes: int = 10,
        image_encoder_dim: int = 384,
        text_encoder_dim: int = 768,
        fusion_hidden_dim: int = 512,
        freeze_encoders: bool = True,
    ) -> None:
        """Initialize encoders and fusion head skeleton."""
        super().__init__()
        self.image_encoder = timm.create_model(
            "deit3_small_patch16_224", pretrained=True, num_classes=0
        )
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # TODO: define fusion MLP from 1152 -> hidden -> num_classes.
        raise NotImplementedError("Implement fusion MLP in __init__.")

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drop_image: bool = False,
        drop_text: bool = False,
    ) -> torch.Tensor:
        """Compute multimodal logits with optional modality ablation."""
        # TODO: encode image and text.
        # TODO: apply modality ablations if requested.
        # TODO: fuse features and classify.
        raise NotImplementedError("Implement MultimodalClassifier.forward.")
