"""BERT token classification skeleton for A10."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTForNER(nn.Module):
    """Token-level NER model built on BERT encoder."""

    def __init__(self, model_name: str, num_labels: int) -> None:
        """Initialize BERT backbone and token classifier head."""
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # TODO: define token-level classification head.
        raise NotImplementedError("Implement token classifier head in __init__.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token logits of shape (B, L, num_labels)."""
        # TODO: run BERT and project each token representation to labels.
        raise NotImplementedError("Implement BERTForNER.forward.")
