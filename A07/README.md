# A07 — BPE Tokenizer and Decoder-Only Language Model

## Overview

Language models operate on tokens, not raw text. Before the model sees a single character, a tokenizer breaks the text into subword units and maps them to integer IDs. This assignment builds the full pipeline: implement a Byte Pair Encoding tokenizer from scratch, build a decoder-only transformer (the GPT architecture), and train it on a small Hindi corpus to understand how language modelling actually works.

This is the most complex assignment so far. The transformer block you build here will be reused conceptually in every subsequent assignment.

---

## Theory

### The language modelling objective

A language model assigns a probability to a sequence of tokens $t_1, t_2, \ldots, t_T$. By the chain rule of probability:

$$P(t_1, t_2, \ldots, t_T) = \prod_{i=1}^{T} P(t_i \mid t_1, \ldots, t_{i-1})$$

We train a model to maximise this log-likelihood (equivalently, minimise the negative log-likelihood, i.e., cross-entropy loss):

$$\mathcal{L} = -\frac{1}{T} \sum_{i=1}^{T} \log P(t_i \mid t_1, \ldots, t_{i-1})$$

At each position $i$, the model sees tokens $t_1, \ldots, t_{i-1}$ and predicts a probability distribution over the vocabulary for the next token $t_i$. This requires causal masking (from A06) to prevent the model from seeing future tokens.

### Perplexity

Perplexity is the standard evaluation metric for language models:

$$\text{PPL} = \exp\!\left(\mathcal{L}\right) = \exp\!\left(-\frac{1}{T} \sum_{i=1}^{T} \log P(t_i \mid t_{<i})\right)$$

Intuition: perplexity is the geometric mean of the inverse probability assigned to each correct token. A model with perplexity 50 is, on average, "surprised" by correct tokens roughly as much as if it were choosing uniformly from 50 options. Lower is better. A random model over a vocabulary of $V$ tokens has perplexity $V$.

### Byte Pair Encoding (BPE)

Raw text tokenisation by character produces very long sequences (slow, difficult long-range dependencies). Word-level tokenisation cannot handle unseen words (large vocabulary, sparse for morphologically rich languages like Hindi). BPE is a data-driven middle ground.

**Training algorithm:**
1. Start with a vocabulary of individual characters (plus a special end-of-word symbol)
2. Count the frequency of all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until the desired vocabulary size is reached

**Example:** In a corpus containing "low low low lower lowest", after several merges:
- `l o w` → `lo w` → `low`
- `l o w e r` → `low e r` → `lower`

The merge rules learned on training data are applied greedily (in order) at inference time to tokenise new text.

**For Hindi specifically:** Hindi script (Devanagari) has composite characters. A character-level vocabulary starts with Unicode codepoints. BPE will naturally learn to merge frequent syllable units like `ना`, `की`, `है` into single tokens, which reduces sequence length and captures morphological structure.

### Decoder-only transformer

The decoder-only architecture (GPT-style) is a stack of $N$ identical blocks, each containing:

1. **Layer Normalisation** (pre-norm, applied before each sublayer):
   $$\text{LN}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta$$
   where $\mu, \sigma$ are the mean and standard deviation computed over the feature dimension, and $\gamma, \beta$ are learnable parameters.

2. **Causal Multi-Head Self-Attention** (from A06, with causal mask)

3. **Residual connection:** $x \leftarrow x + \text{Attention}(\text{LN}(x))$

4. **Feed-Forward Network:**
   $$\text{FFN}(x) = W_2\, \text{GELU}(W_1 x + b_1) + b_2$$
   where $W_1 \in \mathbb{R}^{4d \times d}$ and $W_2 \in \mathbb{R}^{d \times 4d}$ (the 4× expansion is standard).

5. **Residual connection:** $x \leftarrow x + \text{FFN}(\text{LN}(x))$

GELU (Gaussian Error Linear Unit) is preferred over ReLU in transformers:
$$\text{GELU}(x) = x \cdot \Phi(x)$$
where $\Phi$ is the Gaussian CDF. Approximately: $\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$.

### Positional encodings

Attention is permutation-equivariant — it does not know the order of tokens. Positional encodings inject order information. The original transformer uses fixed sinusoidal encodings:

$$\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$\text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

GPT-style models typically use learned positional embeddings — a lookup table $E_{\text{pos}} \in \mathbb{R}^{T_{\max} \times d}$ indexed by position.

### Token and positional embeddings

Each token $t_i$ is mapped to a $d$-dimensional vector via a lookup table $E_{\text{tok}} \in \mathbb{R}^{V \times d}$. The input to the transformer is:

$$X = E_{\text{tok}}[t] + E_{\text{pos}}[\text{pos}]$$

The language model head is a linear projection from $d$ to $V$ (often tied to the token embedding matrix, i.e., $W_{\text{head}} = E_{\text{tok}}^\top$, which reduces parameters and improves performance).

---

## Reading Material

- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE for NLP, 2016): https://arxiv.org/abs/1508.07909
- Vaswani et al., "Attention Is All You Need" (2017): https://arxiv.org/abs/1706.03762
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Hendrycks & Gimpel, "Gaussian Error Linear Units" (GELU, 2016): https://arxiv.org/abs/1606.08415
- Karpathy's minGPT (excellent reference implementation): https://github.com/karpathy/minGPT

---

## Assignment

### Dataset

A small Hindi text corpus — we use the Hindi subset of the CC-100 dataset (approx. 50MB, manageable). Download script provided.

### Task

**Part 1 — BPE Tokenizer:** Implement `BPETokenizer` with `train(corpus, vocab_size)`, `encode(text) -> List[int]`, and `decode(ids) -> str`. Verify lossless encode-decode roundtrip on all training sentences.

**Part 2 — Decoder-only Transformer:** Implement `TransformerBlock` and `DecoderLM`. Architecture: 6 layers, 8 heads, $d_{\text{model}} = 256$, max sequence length 256, GELU FFN.

**Part 3 — Training:** Train for 5 epochs on the Hindi corpus. Log perplexity on train and validation sets. Generate 5 text samples by greedy decoding from random prefix tokens.

### Coding standards (auto-grader requirements from the assignment spec)

These naming conventions are required:
- The pair-count dictionary in BPE must be named `digram_stats`
- The merge step must be a private method `_atomic_merge_step(self, word_list, pair, new_token)`
- Add comment `# Begin iterative vocabulary construction.` before the vocab-building loop
- Add comment `# Begin encoding the user given input text.` inside `encode()`
- Add comment `# Begin decoding the user given input text.` inside `decode()`
- Intermediate Q, K, V projections (before splitting into heads) must be named `q_prime`, `k_prime`, `v_prime`
- Wrap softmax in `safe_softmax(tensor, dim)` helper
- The forward method of TransformerBlock must include comment: `# Implementing feed-forward logic with batching.`

### What to implement

1. `tokenizer.py` — `BPETokenizer` class with all required conventions
2. `model.py` — `TransformerBlock`, `DecoderLM`
3. `dataset.py` — text dataset with sliding window chunking
4. `train.py` — training loop with perplexity logging
5. `generate.py` — greedy and top-k sampling generation

### Deliverables

- [ ] `tokenizer.py` — passes encode-decode roundtrip test
- [ ] `model.py` — correct output shapes verified
- [ ] `train.py` — trains without crashing
- [ ] `perplexity_curve.png` — train/val perplexity vs epoch
- [ ] `generated_samples.txt` — 5 Hindi text samples
- [ ] Final validation perplexity < 150 (for the small model size, this is reasonable)

---

## Starter Code

```python
# download_data.py
import os, urllib.request, gzip, shutil

URL  = 'https://data.statmt.org/cc-100/hi.txt.xz'
DEST = './data/hi.txt.xz'
OUT  = './data/hi.txt'

os.makedirs('./data', exist_ok=True)
print("Downloading Hindi CC-100 (this may take a few minutes)...")
urllib.request.urlretrieve(URL, DEST)

import lzma
with lzma.open(DEST, 'rb') as f_in, open(OUT, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

# Use first 200k lines for manageable training
with open(OUT, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()][:200_000]

with open('./data/hi_small.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Saved {len(lines)} lines. Total chars: {sum(len(l) for l in lines):,}")
```

```python
# tokenizer.py  — fill in TODOs
from collections import Counter
from typing import List, Dict, Tuple

class BPETokenizer:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.id_to_token: Dict[int, str] = {}

    def train(self, corpus: str, vocab_size: int):
        # TODO: initialise vocabulary with character-level tokens
        # Begin iterative vocabulary construction.
        # TODO: iteratively count digram_stats and merge most frequent pair
        pass

    def _atomic_merge_step(self, word_list: list, pair: tuple, new_token: str) -> list:
        # TODO: replace all occurrences of `pair` in `word_list` with `new_token`
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        # Begin encoding the user given input text.
        # TODO: apply merge rules greedily, return list of token IDs
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        # Begin decoding the user given input text.
        # TODO: map IDs back to tokens, join, handle end-of-word symbols
        raise NotImplementedError
```

```python
# model.py  — fill in TODOs
import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    # TODO: numerically stable softmax
    raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, max_len: int):
        super().__init__()
        # TODO: define ln1, ln2, attention (your MultiHeadAttention from A06), ffn layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementing feed-forward logic with batching.
        # TODO: pre-norm → attention → residual → pre-norm → FFN → residual
        raise NotImplementedError


class DecoderLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, max_len: int):
        super().__init__()
        # TODO: token embedding, positional embedding, transformer blocks, final LN, head
        raise NotImplementedError

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L)
        # Returns logits: (B, L, vocab_size)
        raise NotImplementedError
```

---

## Notes

Hindi text requires careful UTF-8 handling. Use `text.encode('utf-8').decode('utf-8')` to normalise, and split by characters (not bytes) for initial BPE vocabulary. Devanagari characters are multi-byte in UTF-8 but should be treated as single units.

At the start of training, perplexity should be close to your vocabulary size (random predictions). If it is much higher, your loss computation is wrong — likely you are not ignoring padding tokens correctly or your targets are off by one position.

The targets for language modelling are the inputs shifted by one: if the input sequence is $[t_1, t_2, \ldots, t_L]$, the target is $[t_2, t_3, \ldots, t_{L+1}]$. Use `input_ids[:, 1:]` as targets and `logits[:, :-1]` for the loss.
