"""SST-2 dataset and dataloader helpers."""

from __future__ import annotations

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding



def build_sst2_loaders(
    model_name: str,
    batch_size: int,
    max_length: int = 128,
) -> tuple[DataLoader, DataLoader, AutoTokenizer]:
    """Load SST-2 and return train/validation dataloaders."""
    ds = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tok(batch["sentence"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collate = DataCollatorWithPadding(tokenizer=tok)
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(ds["validation"], batch_size=batch_size,
                            shuffle=False, collate_fn=collate)
    return train_loader, val_loader, tok
