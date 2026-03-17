"""Text dataset for sliding-window LM training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Create fixed-length language modeling chunks from token IDs."""

    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        """Initialize token stream and sequence length."""
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Number of sliding windows in token stream."""
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids) for next-token prediction."""
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
