"""CoNLL-2003 NER dataset with subword label alignment."""

from __future__ import annotations

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification



def build_conll_loaders(
    model_name: str,
    batch_size: int,
    max_length: int = 256,
) -> tuple[DataLoader, DataLoader, list[str], AutoTokenizer]:
    """Load CoNLL-2003 and return aligned token classification loaders."""
    ds = load_dataset("conll2003")
    label_list = ds["train"].features["ner_tags"].feature.names
    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align(batch):
        enc = tok(
            batch["tokens"],
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
        )
        labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)
            prev = None
            label_ids = []
            for w in word_ids:
                if w is None:
                    label_ids.append(-100)
                elif w != prev:
                    label_ids.append(tags[w])
                else:
                    label_ids.append(-100)
                prev = w
            labels.append(label_ids)
        enc["labels"] = labels
        return enc

    ds = ds.map(tokenize_and_align, batched=True)
    cols = ["input_ids", "attention_mask", "labels"]
    ds.set_format(type="torch", columns=cols)

    collate = DataCollatorForTokenClassification(tokenizer=tok)
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(ds["validation"], batch_size=batch_size,
                            shuffle=False, collate_fn=collate)
    return train_loader, val_loader, label_list, tok
