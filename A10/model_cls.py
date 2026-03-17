"""BERT classification skeleton for A10."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTForClassification(nn.Module):
    """Sentence classifier using BERT encoder."""

    def __init__(self, model_name: str, num_labels: int = 2) -> None:
        """Initialize encoder and classification head.

        Learner task:
            Define the classifier head on top of CLS representation.
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # TODO: define classifier head using hidden_size -> num_labels.
        raise NotImplementedError("Implement classification head in __init__.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sentence-level logits."""
        # TODO: run BERT, take CLS, and pass through classifier head.
        raise NotImplementedError("Implement BERTForClassification.forward.")
