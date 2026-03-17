# A10 — BERT: Encoder Models, Bidirectional Attention, and Token-Level Tasks

## Overview

So far you have worked with decoder-only models (causal attention, A07) and classification models (ViT, A08/A09). BERT is neither — it is an encoder model trained with bidirectional attention and a masked language modelling objective. The architectural difference is small (remove the causal mask) but the resulting representations are fundamentally different in character. BERT tokens see the full context in both directions, making them better for tasks that require understanding a full sentence.

This assignment fine-tunes BERT-base on two structurally different tasks: sentence classification and Named Entity Recognition (token-level prediction). Understanding why these tasks require different heads — and only different heads — is the core insight.

---

## Theory

### Masked Language Modelling (MLM)

BERT is pretrained by randomly masking 15% of tokens and predicting the masked tokens from context. For a masked position $i$:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(t_i \mid t_1, \ldots, \mathbf{[MASK]}_i, \ldots, t_L)$$

This differs fundamentally from causal LM: BERT predicts masked positions using both left and right context simultaneously. The result is a bidirectional representation — each token's embedding reflects both what precedes and follows it.

The tradeoff: bidirectional models cannot be used for generation (they need the full sequence at inference time). They excel at understanding tasks: classification, NER, question answering, entailment.

### Bidirectional vs causal attention

Causal (decoder) attention masks out future positions:

$$A_{\text{causal}} = \text{softmax}\!\left(\frac{QK^\top + M}{\sqrt{d_k}}\right)$$

where $M_{ij} = -\infty$ for $j > i$.

BERT's bidirectional (encoder) attention removes the mask entirely:

$$A_{\text{bidi}} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$

Every token attends to every other token. This gives richer contextual embeddings — the word "bank" in "river bank" gets a different representation than in "savings bank" because the surrounding tokens influence it through attention.

### The CLS token for classification

Like ViT (A08), BERT prepends a `[CLS]` token. After $L$ encoder layers, the CLS embedding $\mathbf{h}_{\text{CLS}}^{(L)}$ is passed through a classification head:

$$\hat{y} = \text{softmax}(W \mathbf{h}_{\text{CLS}}^{(L)} + b)$$

For the purpose of fine-tuning, the pretrained BERT encoder provides a rich contextual encoding of the entire sentence via the CLS token. Only $W$ and $b$ are new; the encoder adapts slightly through full fine-tuning.

### Token-level predictions: Named Entity Recognition

NER assigns a label to each token (e.g., PER for person name, ORG for organisation, LOC for location, O for no entity). Instead of using the CLS embedding, we use the per-token embeddings $\mathbf{h}_i^{(L)}$ for each token $i$:

$$\hat{y}_i = \text{softmax}(W \mathbf{h}_i^{(L)} + b)$$

This is structurally identical to classification but applied to every token independently. The loss is summed over all non-padding tokens.

**Subword alignment problem:** BERT's tokeniser splits words into subwords (e.g., "Washington" → ["Wash", "##ington"]). NER labels are typically at the word level, not the subword level. Standard practice: assign the label only to the first subword of each word and ignore the rest (use `-100` as label for ignored subwords, which `CrossEntropyLoss` ignores by default when `ignore_index=-100`).

### Why fine-tune vs train from scratch?

BERT-base has 110M parameters. Training it on a domain-specific NER dataset of a few thousand sentences would overfit immediately. The pretrained weights encode deep linguistic knowledge — BERT already "knows" that "Washington" is often a named entity. Fine-tuning for a few epochs adapts this knowledge to the specific label schema of your dataset with minimal data.

Typical fine-tuning regime:
- Learning rate: $2 \times 10^{-5}$ (much lower than other assignments — BERT is sensitive to large LR)
- Epochs: 3–5
- Warmup: linear learning rate warmup for first 10% of steps, then linear decay

### BERT-base architecture

- 12 encoder blocks
- 12 attention heads
- $d_{\text{model}} = 768$
- FFN hidden dimension: 3072
- Total parameters: ~110M
- Max sequence length: 512 tokens

---

## Reading Material

- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018): https://arxiv.org/abs/1810.04805
- Alammar, "The Illustrated BERT, ELMo, and co." (blog): https://jalammar.github.io/illustrated-bert/
- Conneau & Kiela, "SentEval: An Evaluation Toolkit for Universal Sentence Representations" — for understanding what sentence embeddings capture: https://arxiv.org/abs/1803.05449
- Hugging Face fine-tuning tutorial: https://huggingface.co/docs/transformers/training

---

## Assignment

### Datasets

- **Classification:** SST-2 (Stanford Sentiment Treebank) — binary sentiment classification of movie reviews. Available via HuggingFace `datasets`.
- **NER:** CoNLL-2003 English NER — 4 entity types (PER, ORG, LOC, MISC). Available via HuggingFace `datasets`.

Both are small enough to fine-tune in under an hour on a single GPU.

### Task

**Part 1 — Sentence Classification (SST-2):** Fine-tune `bert-base-uncased` for binary sentiment classification. Use the CLS token representation. Report accuracy and F1 on the test set. Expected: ≥ 92% accuracy.

**Part 2 — NER (CoNLL-2003):** Fine-tune the same `bert-base-uncased` (fresh fine-tune, not from Part 1) for token-level NER. Handle subword alignment correctly. Report entity-level F1 using `seqeval` library. Expected: ≥ 84% entity F1.

**Part 3 — Attention head analysis:** For 5 sentences from each task, visualise the attention weights of a specific head from BERT's last layer as a heatmap. In your `notes.md`, identify at least 2 attention heads that appear to have learned interpretable patterns (e.g., one head might consistently attend to the `[SEP]` token, another might focus on syntactic subjects).

### What to implement

1. Dataset loading and tokenisation for both tasks
2. Subword alignment for NER labels
3. `ClassificationHead` and `NERHead` (both thin wrappers on top of BERT)
4. Training loops for both tasks (separate scripts)
5. Attention heatmap extraction (reuse/adapt `AttentionMapExtractor` from A08)

### Deliverables

- [ ] `dataset_cls.py` — SST-2 loading and tokenisation
- [ ] `dataset_ner.py` — CoNLL-2003 loading, tokenisation, label alignment
- [ ] `model_cls.py` — BERT + classification head
- [ ] `model_ner.py` — BERT + NER head
- [ ] `train_cls.py`
- [ ] `train_ner.py`
- [ ] `evaluate_cls.py` — accuracy, F1
- [ ] `evaluate_ner.py` — seqeval entity F1
- [ ] `attention_cls.png` — attention heatmaps for classification
- [ ] `attention_ner.png` — attention heatmaps for NER
- [ ] `notes.md` — attention head analysis

---

## Starter Code

```python
# dataset_ner.py  — fill in TODOs
from datasets import load_dataset
from transformers import BertTokenizerFast
import torch

def load_conll2003(tokenizer_name='bert-base-uncased', max_len=128):
    dataset = load_dataset('conll2003')
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    label_list = dataset['train'].features['ner_tags'].feature.names
    # e.g. ['O', 'B-PER', 'I-PER', 'B-ORG', ...]

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples['tokens'],
            is_split_into_words=True,
            max_length=max_len,
            padding='max_length',
            truncation=True
        )
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)  # Special tokens: ignore
                elif word_id != prev_word_id:
                    aligned.append(label[word_id])  # First subword: use label
                else:
                    aligned.append(-100)  # Continuation subword: ignore
                prev_word_id = word_id
            labels.append(aligned)
        tokenized['labels'] = labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_and_align, batched=True)
    return tokenized_dataset, label_list
```

```python
# model_ner.py  — fill in TODO
import torch.nn as nn
from transformers import BertModel

class BERTForNER(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # TODO: add a linear classification head mapping 768 → num_labels
        # applied to every token position
        raise NotImplementedError

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state  # (B, L, 768)
        # TODO: apply classification head to every token
        # Returns: logits (B, L, num_labels)
        raise NotImplementedError
```

```python
# evaluate_ner.py  — seqeval-based entity F1
from seqeval.metrics import classification_report, f1_score

def evaluate_ner(model, dataloader, label_list, device):
    model.eval()
    all_preds = []
    all_labels = []
    # TODO: iterate dataloader, get predictions
    # Convert label IDs back to label strings
    # Ignore positions where label == -100
    # all_preds and all_labels should be lists of lists of strings
    # e.g. all_preds[0] = ['O', 'B-PER', 'I-PER', 'O', ...]

    print(classification_report(all_labels, all_preds))
    return f1_score(all_labels, all_preds)
```

---

## Notes

Use `BertTokenizerFast` (not `BertTokenizer`) for NER — the fast version provides `.word_ids()` which is essential for subword alignment. The slow tokenizer does not expose this method.

For the attention heatmaps, BERT's attention weights are accessible at `outputs.attentions` if you pass `output_attentions=True` to the model. This is much cleaner than the hook-based approach used in A08.

The learning rate $2 \times 10^{-5}$ is not arbitrary — large learning rates destabilise BERT fine-tuning. The pretrained weights are in a delicate equilibrium; too large an update destroys the representations. This is why BERT papers universally recommend this range.
