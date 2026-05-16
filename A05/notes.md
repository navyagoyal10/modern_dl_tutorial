# A05 — Notes and Analysis

Fill in each section after completing training and generation.
Paste actual samples and outputs from your runs — do not describe hypothetically.

---

## Training configuration used

| Setting | Value |
|---|---|
| `hidden_dim` | |
| `embed_dim` | |
| `seq_len` | |
| `batch_size` | |
| `epochs` | |
| `lr` | |
| `clip_norm` | |

---

## Results summary

| Model | Final Train Loss | Final Val Loss | Test Loss | Test PPL |
|---|---|---|---|---|
| Vanilla RNN | | | | |
| LSTM | | | | |

---

## Question 1 — Temperature and coherence

Generate 200-character samples from your trained **LSTM** at temperatures 0.5, 0.8, and 1.2.
Paste the raw output for each below, then answer the questions.

### Temperature 0.5

```
[paste sample here]
```

### Temperature 0.8

```
[paste sample here]
```

### Temperature 1.2

```
[paste sample here]
```

### Analysis

**At which temperature does the text feel most "Shakespeare-like"?**

[your answer]

**At which temperature does it become incoherent? What does the text look like?**

[your answer]

**Connect this to what temperature does to the softmax distribution mathematically.**
(Hint: what happens to the probability of the top character as T → 0? As T → ∞?)

[your answer]

---

## Question 2 — Long-range structure

Generate 500-character samples from **both models** at temperature 0.8.
Paste both samples below, then answer the questions.

### RNN sample (temperature 0.8, 500 chars)

```
[paste sample here]
```

### LSTM sample (temperature 0.8, 500 chars)

```
[paste sample here]
```

### Analysis

**Find one specific example in your samples where the RNN breaks down but the LSTM sustains long-range structure.**
(Look for: matching quotation marks, consistent speaker names, sustained metre, repeated phrases, open/close brackets.)

[paste the specific excerpt and describe what you observe]

**Explain this observation in terms of gradient paths.**
(Why can the LSTM preserve information that the RNN cannot?
Reference the $f_t$ gradient path vs the $W_{hh}^\top \cdot \tanh'$ path from the theory section.)

[your answer]

---

## Question 3 — Forget gate behaviour

After training your LSTM, run the following snippet to extract and analyse the forget gate bias:

```python
import torch
import matplotlib.pyplot as plt

ckpt = torch.load("outputs/checkpoints/lstm_best.pt", map_location="cpu")

# The forget gate bias is the first hidden_dim slice of self.b
# (order: forget, input, candidate, output — matching the .chunk(4) in forward())
b = ckpt["model"]["b"]
hidden_dim = b.shape[0] // 4
b_f = b[:hidden_dim]

print(f"b_f mean:  {b_f.mean().item():.4f}")
print(f"b_f std:   {b_f.std().item():.4f}")
print(f"b_f min:   {b_f.min().item():.4f}")
print(f"b_f max:   {b_f.max().item():.4f}")

plt.hist(b_f.numpy(), bins=30)
plt.title("Forget gate bias distribution after training")
plt.xlabel("b_f value")
plt.ylabel("Count")
plt.axvline(1.0, color="red", linestyle="--", label="init value (1.0)")
plt.legend()
plt.savefig("outputs/plots/forget_gate_bias.png", dpi=150)
plt.show()
```

### Forget gate bias statistics

| Statistic | Value |
|---|---|
| Mean | |
| Std | |
| Min | |
| Max | |

### Forget gate bias histogram

Save `outputs/plots/forget_gate_bias.png` and confirm it is present in your outputs.

### Analysis

**Did the forget gate bias values move far from their initialisation of 1.0?**

[your answer]

**What does a high forget gate bias value mean for how the LSTM handles that memory dimension?**
(Hint: σ(large positive) ≈ 1. What does f_t ≈ 1 do to c_t?)

[your answer]

**What does a low (or negative) forget gate bias value mean?**
(Hint: σ(large negative) ≈ 0.)

[your answer]

**Is there a pattern in which dimensions have high vs low forget gate bias?**
(You can try printing the top-5 and bottom-5 values by index.)

[your answer]