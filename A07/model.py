"""Decoder-only LM skeleton for A07."""

from __future__ import annotations

import torch
import torch.nn as nn



def safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax implementation."""
    # TODO: implement stable softmax by subtracting max logits.
    raise NotImplementedError("Implement safe_softmax.")


class TransformerBlock(nn.Module):
    """Single decoder transformer block skeleton."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        """Initialize sublayers for self-attention and MLP."""
        super().__init__()
        # TODO: define attention, feed-forward, and normalization layers.

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Run one transformer block forward pass.

        Required learner comment:
            Apply pre-norm -> masked attention -> residual -> MLP -> residual.
        """
        raise NotImplementedError("Implement TransformerBlock.forward.")


class DecoderLM(nn.Module):
    """Decoder-only language model skeleton."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
    ) -> None:
        """Initialize token embeddings, position embeddings, and blocks."""
        super().__init__()
        # TODO: define embeddings, transformer blocks, and output head.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits.

        Args:
            input_ids: (B, L)

        Returns:
            logits: (B, L, vocab_size)
        """
        raise NotImplementedError("Implement DecoderLM.forward.")
