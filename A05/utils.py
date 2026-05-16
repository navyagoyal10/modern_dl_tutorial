"""Utility functions for A05 — RNN and LSTM from scratch."""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_perplexity(avg_cross_entropy: float) -> float:
    """
    Compute perplexity from average per-token cross-entropy loss.

    PPL = exp(avg_CE)

    A perplexity equal to vocab_size means the model is no better than
    random. A perplexity of 1 means perfect prediction. In practice,
    a well-trained character RNN on Shakespeare reaches PPL ≈ 3–6.

    Args:
        avg_cross_entropy: mean cross-entropy loss per character

    Returns:
        perplexity (scalar float)
    """
    return math.exp(avg_cross_entropy)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:        torch.nn.Module,
    optimizer:    torch.optim.Optimizer,
    epoch:        int,
    val_loss:     float,
    path:         str,
    train_losses: list[float] | None = None,
    val_losses:   list[float] | None = None,
) -> None:
    """
    Save model and optimizer state to a .pt checkpoint file.

    Args:
        model:     the model to save
        optimizer: the optimizer to save
        epoch:     current epoch number
        val_loss:  validation loss at this checkpoint
        path:      file path (e.g. 'checkpoints/rnn_best.pt')
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":        epoch,
            "val_loss":     val_loss,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "train_losses": train_losses or [],
            "val_losses":   val_losses   or [],
            # Save model config so evaluate/generate can reconstruct without CLI flags
            "model_config": {
                "vocab_size": model.vocab_size,
                "embed_dim":  model.embed_dim,
                "hidden_dim": model.hidden_dim,
            },
        },
        path,
    )
    print(f"  Saved checkpoint → {path}  (epoch {epoch}, val_loss={val_loss:.4f})")


def load_checkpoint(
    model:     torch.nn.Module,
    path:      str,
    optimizer: torch.optim.Optimizer | None = None,
    device:    torch.device | None = None,
) -> dict:
    """
    Load a checkpoint into a model (and optionally an optimizer).

    Args:
        model:     model to load weights into
        path:      checkpoint file path
        optimizer: if provided, optimizer state is also restored
        device:    device to map tensors to

    Returns:
        checkpoint dict (contains 'epoch' and 'val_loss')
    """
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return ckpt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_losses: list[float],
    val_losses:   list[float],
    model_name:   str,
    save_path:    str,
) -> None:
    """
    Plot training and validation loss curves for one model.

    Args:
        train_losses: per-epoch training loss
        val_losses:   per-epoch validation loss
        model_name:   label shown in the title (e.g. 'RNN' or 'LSTM')
        save_path:    file path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="train loss")
    axes[0].plot(epochs, val_losses,   label="val loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ppl_train = [compute_perplexity(l) for l in train_losses]
    ppl_val   = [compute_perplexity(l) for l in val_losses]
    axes[1].plot(epochs, ppl_train, label="train PPL")
    axes[1].plot(epochs, ppl_val,   label="val PPL")
    axes[1].set_title(f"{model_name} — Perplexity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve → {save_path}")


def plot_comparison(
    rnn_train:  list[float],
    rnn_val:    list[float],
    lstm_train: list[float],
    lstm_val:   list[float],
    save_path:  str,
) -> None:
    """
    Plot validation perplexity for both models on one figure.

    Args:
        rnn_train, rnn_val:   RNN loss histories
        lstm_train, lstm_val: LSTM loss histories
        save_path:            file path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    rnn_ppl  = [compute_perplexity(l) for l in rnn_val]
    lstm_ppl = [compute_perplexity(l) for l in lstm_val]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(rnn_ppl)  + 1), rnn_ppl,  label="RNN  val PPL",  color="steelblue")
    ax.plot(range(1, len(lstm_ppl) + 1), lstm_ppl, label="LSTM val PPL",  color="darkorange")
    ax.set_title("RNN vs LSTM — Validation Perplexity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot → {save_path}")