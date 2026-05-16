"""
gradient_check.py — Verify that your CharRNN cell implementation is correct.

Uses torch.autograd.gradcheck, which compares autograd-computed gradients
against numerical finite differences at double precision.

The approach: write a pure functional RNN cell (no nn.Module, no .data swaps)
that takes W_xh, W_hh, b_h as explicit tensor arguments. gradcheck can then
perturb each element and verify the autograd gradient matches the numerical one.
This correctly connects the input tensors to the computation graph.

Run this before training. A PASSED result means your RNN cell equations are
correct and the time-step loop is properly wired for autograd.

Usage:
    python gradient_check.py

Expected output:
    Shape check (logits) .......... PASSED
    Shape check (hidden state) .... PASSED
    gradcheck W_xh ................ PASSED
    gradcheck W_hh ................ PASSED
    gradcheck b_h ................. PASSED
    All checks PASSED
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from rnn import CharRNN


# ---------------------------------------------------------------------------
# Shape checks
# ---------------------------------------------------------------------------

def check_shapes(
    vocab_size: int = 20,
    embed_dim:  int = 8,
    hidden_dim: int = 16,
    batch_size: int = 2,
    seq_len:    int = 10,
) -> bool:
    """Instantiate a small CharRNN and verify output tensor shapes."""
    model = CharRNN(vocab_size, embed_dim, hidden_dim)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, h = model(x)

    logits_ok = logits.shape == (batch_size, seq_len, vocab_size)
    h_ok      = h.shape      == (batch_size, hidden_dim)

    def report(name, ok, got, expected):
        status = "PASSED" if ok else f"FAILED — got {got}, expected {expected}"
        print(f"Shape check ({name}) {'.' * (22 - len(name))} {status}")

    report("logits",       logits_ok, tuple(logits.shape), (batch_size, seq_len, vocab_size))
    report("hidden state", h_ok,      tuple(h.shape),      (batch_size, hidden_dim))
    return logits_ok and h_ok


# ---------------------------------------------------------------------------
# Functional RNN cell — no nn.Module, parameters are explicit tensor args
# ---------------------------------------------------------------------------

def _rnn_forward_functional(
    x:          torch.Tensor,   # (batch, seq_len)  long — character indices
    embed_W:    torch.Tensor,   # (vocab_size, embed_dim)  — embedding table
    W_xh:       torch.Tensor,   # (hidden_dim, embed_dim)
    W_hh:       torch.Tensor,   # (hidden_dim, hidden_dim)
    b_h:        torch.Tensor,   # (hidden_dim,)
    W_hy:       torch.Tensor,   # (vocab_size, hidden_dim)
    b_y:        torch.Tensor,   # (vocab_size,)
    hidden_dim: int,
) -> torch.Tensor:
    """
    Pure functional forward pass — identical equations to CharRNN.forward()
    but takes all parameters as explicit float tensors so gradcheck can
    perturb them and track gradients correctly.

    Returns logits summed to a scalar (gradcheck needs a scalar or small output).
    """
    batch, seq_len = x.shape
    # Embedding lookup via matmul with one-hot (keeps computation graph intact)
    one_hot = F.one_hot(x, num_classes=embed_W.shape[0]).to(embed_W.dtype)  # (B, T, V)
    embeds  = one_hot @ embed_W                                               # (B, T, E)

    h = torch.zeros(batch, hidden_dim, dtype=W_xh.dtype, device=W_xh.device)
    logits_all = []
    for t in range(seq_len):
        x_t = embeds[:, t, :]                          # (B, E)
        a_t = x_t @ W_xh.T + h @ W_hh.T + b_h         # (B, H)
        h   = torch.tanh(a_t)                          # (B, H)
        z_t = h @ W_hy.T + b_y                         # (B, V)
        logits_all.append(z_t)

    return torch.stack(logits_all, dim=1).sum()         # scalar


# ---------------------------------------------------------------------------
# Gradient checks
# ---------------------------------------------------------------------------

def check_gradients(
    vocab_size: int = 15,
    embed_dim:  int = 6,
    hidden_dim: int = 8,
    seq_len:    int = 4,
    batch_size: int = 2,
) -> bool:
    """
    Run torch.autograd.gradcheck on W_xh, W_hh, and b_h.

    All tensors are double precision. The embedding table and output weights
    are held fixed (requires_grad=False); only the checked parameter is
    passed with requires_grad=True so gradcheck perturbs only that tensor.
    """
    torch.manual_seed(0)
    dtype = torch.float64

    x = torch.randint(0, vocab_size, (batch_size, seq_len))  # fixed integer input

    # Fixed parameters (not checked)
    embed_W = torch.randn(vocab_size, embed_dim, dtype=dtype)
    W_hy    = torch.randn(vocab_size, hidden_dim, dtype=dtype)
    b_y     = torch.randn(vocab_size,             dtype=dtype)

    # Parameters to check — initialise with small random values for stability
    W_xh_base = torch.randn(hidden_dim, embed_dim,  dtype=dtype) * 0.1
    W_hh_base = torch.randn(hidden_dim, hidden_dim, dtype=dtype) * 0.1
    b_h_base  = torch.randn(hidden_dim,              dtype=dtype) * 0.1

    checks = {
        "W_xh": W_xh_base.clone(),
        "W_hh": W_hh_base.clone(),
        "b_h":  b_h_base.clone(),
    }

    all_passed = True

    for name, param_base in checks.items():
        inp = param_base.clone().requires_grad_(True)

        if name == "W_xh":
            fn = lambda p: _rnn_forward_functional(
                x, embed_W, p, W_hh_base, b_h_base, W_hy, b_y, hidden_dim
            )
        elif name == "W_hh":
            fn = lambda p: _rnn_forward_functional(
                x, embed_W, W_xh_base, p, b_h_base, W_hy, b_y, hidden_dim
            )
        else:  # b_h
            fn = lambda p: _rnn_forward_functional(
                x, embed_W, W_xh_base, W_hh_base, p, W_hy, b_y, hidden_dim
            )

        try:
            result = gradcheck(
                fn, (inp,),
                eps           = 1e-5,
                atol          = 1e-4,
                rtol          = 1e-3,
                raise_on_error = False,
            )
            status = "PASSED" if result else "FAILED"
        except Exception as e:
            status = f"ERROR — {e}"
            result = False

        pad = "." * (22 - len(name))
        print(f"gradcheck {name} {pad} {status}")

        if not result:
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("CharRNN gradient check")
    print("=" * 55)

    shapes_ok = check_shapes()
    if not shapes_ok:
        print("\nFix forward() shapes before running gradient checks.")
        sys.exit(1)

    print()
    grads_ok = check_gradients()

    print()
    if grads_ok:
        print("All checks PASSED — your RNN cell is correct.")
        sys.exit(0)
    else:
        print("Some checks FAILED. Common causes:")
        print("  • W_hh not used: h only depends on x_t, not h_{t-1}")
        print("  • No time loop: forward() only computes the last step")
        print("  • Wrong activation: sigmoid instead of tanh for h_t")
        print("  • W_xh / W_hh as nn.Linear instead of nn.Parameter")
        sys.exit(1)


if __name__ == "__main__":
    main()