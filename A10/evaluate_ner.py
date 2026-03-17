"""Evaluate NER model with seqeval entity F1."""

from __future__ import annotations

import argparse

import torch
from seqeval.metrics import f1_score

from dataset_ner import build_conll_loaders
from model_ner import BERTForNER



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate NER model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Compute seqeval entity-level F1 score."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, label_list, _ = build_conll_loaders(args.model_name,
                                                       args.batch_size)
    model = BERTForNER(args.model_name, num_labels=len(label_list)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            label_ids = batch["labels"].cpu().tolist()
            for p_row, l_row in zip(pred_ids, label_ids):
                p_tags, l_tags = [], []
                for p_i, l_i in zip(p_row, l_row):
                    if l_i == -100:
                        continue
                    p_tags.append(label_list[p_i])
                    l_tags.append(label_list[l_i])
                y_pred.append(p_tags)
                y_true.append(l_tags)

    print(f"Entity F1: {f1_score(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main(parse_args())
