"""Two-stage multimodal training script for A12."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from multimodal_model import MultimodalClassifier
from utils import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train multimodal EuroSAT model")
    p.add_argument("--save_dir", type=str, default="./artifacts")
    p.add_argument("--frozen_epochs", type=int, default=10)
    p.add_argument("--finetune_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_full", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_ckpt(path: Path, epoch: int, model, optimizer, metric: float) -> None:
    """Save checkpoint in required standardized format."""
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metric": metric,
        },
        path,
    )


def main(args: argparse.Namespace) -> None:
    """Run frozen-stage then unfrozen-stage training skeleton."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # TODO: build train/val dataloaders with MultimodalEuroSAT dataset.
    raise NotImplementedError("Implement dataloaders and training loop stages.")

    # Stage 1: frozen encoders.
    model = MultimodalClassifier(freeze_encoders=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_head)
    # TODO: train for args.frozen_epochs and compute validation metric.
    save_ckpt(save_dir / "stage1_frozen.pt", args.frozen_epochs, model,
              optimizer, 0.0)

    # Stage 2: unfreeze all parameters and continue fine-tuning.
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_full)
    # TODO: train for args.finetune_epochs and compute validation metric.
    save_ckpt(save_dir / "stage2_unfrozen.pt", args.finetune_epochs, model,
              optimizer, 0.0)


if __name__ == "__main__":
    main(parse_args())
