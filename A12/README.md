# A12 — Capstone: Multimodal Classification Pipeline

## Overview

The previous eleven assignments built components in isolation. This one assembles them. A multimodal classifier takes both an image and a text description as input and predicts a label — performing better than either modality alone because different modalities carry complementary information.

This is a practical engineering assignment as much as a research one. You will build a late-fusion pipeline that combines a ViT image encoder (from A08) with a BERT text encoder (from A10), profile the memory and throughput of the combined system, and document design decisions in a written report.

---

## Theory

### Why multimodal?

Unimodal models are limited by what a single modality can convey. For satellite land-use classification:
- An image of a river looks like a highway at low resolution (both are thin linear features)
- A text description "narrow water body between banks" disambiguates trivially
- A text description "paved road with lane markings" does the same

Multimodal models can leverage both signals, reducing error in ambiguous cases.

### Late fusion

There are three broad fusion strategies:

**Early fusion:** Concatenate raw inputs or low-level features, pass through a joint model. Tight coupling — the model learns joint representations from scratch. Requires substantial data to avoid the model ignoring one modality.

**Late fusion:** Run separate unimodal encoders, combine their final representations. The unimodal encoders can be pretrained and frozen. This is robust and easy to implement.

**Intermediate (cross-modal attention):** The two modalities interact through cross-attention at intermediate layers. More powerful but much more complex (e.g., CLIP, FLAVA, Flamingo).

For this capstone, late fusion is the right choice — it lets you reuse your A08 and A10 models directly with minimal new code.

### The fusion layer

Given image embedding $\mathbf{v} \in \mathbb{R}^{d_v}$ (from ViT CLS token) and text embedding $\mathbf{t} \in \mathbb{R}^{d_t}$ (from BERT CLS token):

**Concatenation:**
$$\mathbf{f} = [\mathbf{v};\, \mathbf{t}] \in \mathbb{R}^{d_v + d_t}$$
$$\hat{y} = W_{\text{cls}} \mathbf{f} + b$$

Simple, effective, interpretable. The classifier learns a linear combination of image and text features.

**Gated fusion:** Learn a gate $g = \sigma(W_g [\mathbf{v}; \mathbf{t}])$ that controls the relative contribution of each modality:

$$\mathbf{f} = g \odot \mathbf{v} + (1 - g) \odot \mathbf{t}$$

More expressive — the model learns when to trust vision vs language.

**MLP fusion (recommended for this assignment):**

$$\mathbf{f} = \text{ReLU}(W_1 [\mathbf{v}; \mathbf{t}] + b_1)$$
$$\hat{y} = W_2 \mathbf{f} + b_2$$

A thin MLP on top of the concatenated representation. More flexible than linear but far cheaper than cross-attention.

### Generating text descriptions for EuroSAT

EuroSAT does not come with text descriptions — you will generate them. For each class, write a short template description:

```python
DESCRIPTIONS = {
    'AnnualCrop':     'Aerial view of agricultural fields with annual crop rows',
    'Forest':         'Dense tree canopy covering the ground in an aerial satellite image',
    'HerbaceousVegetation': 'Low vegetation and grass covering open fields',
    'Highway':        'A paved highway or motorway visible from above',
    'Industrial':     'Industrial buildings and factory facilities from aerial view',
    'Pasture':        'Open grassland or pasture used for grazing',
    'PermanentCrop':  'Permanent crops such as orchards or vineyards from above',
    'Residential':    'Residential area with houses and streets in a satellite image',
    'River':          'A river or stream flowing through the landscape',
    'SeaLake':        'A body of water such as a sea, lake or reservoir'
}
```

This is a simplified but valid approach — the text encodes class-level semantic information. In a real multimodal dataset, text would be instance-specific (e.g., image captions or metadata). For a more interesting experiment, you can also use BLIP-2 or CLIP to generate per-image captions automatically.

### Modality dropout for robustness

In real deployments, one modality may be missing. Modality dropout randomly drops either the image or text embedding (replaces with zeros) during training with probability $p_{\text{drop}}$ each. This forces the model to learn useful predictions from either modality alone, making it robust to missing inputs at inference.

### Throughput and memory profiling

A production system needs to process many images quickly. Profile your pipeline:
- Batch inference time (images/second) for the combined model
- Memory footprint (model parameters + activations) during forward pass
- Breakdown: how much time is spent in the ViT encoder vs BERT encoder vs fusion layer?

Use `torch.cuda.memory_allocated()` and `torch.utils.benchmark.Timer` for measurements.

---

## Reading Material

- Radford et al., "Learning Transferable Visual Models from Natural Language Supervision" (CLIP, 2021): https://arxiv.org/abs/2103.00020
- Kiela et al., "Supervised Multimodal Bitransformers for Classifying Images and Text" (2019): https://arxiv.org/abs/1909.02950
- Li et al., "BLIP: Bootstrapping Language-Image Pre-training" (2022): https://arxiv.org/abs/2201.12086
- PyTorch profiler documentation: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

---

## Assignment

### Dataset

EuroSAT RGB (same as A02–A04, A08–A09), augmented with the template text descriptions above.

### Task

**Part 1 — Multimodal pipeline:** Build `MultimodalClassifier` combining:
- Frozen DeiT-3-Small from A08 (image encoder, output dim 384)
- Frozen `bert-base-uncased` from A10 (text encoder, output dim 768)
- MLP fusion layer (input 384+768=1152 → 512 → 10)

Freeze both encoders. Train only the fusion MLP for 10 epochs. Compare accuracy against unimodal image-only and text-only baselines.

**Part 2 — Unfreeze encoder fine-tuning:** After fusion head training converges, unfreeze both encoders with a very low learning rate ($5 \times 10^{-6}$). Train for 5 more epochs. Report improvement.

**Part 3 — Modality ablation:** At test time, run three evaluations: image-only (zero out text embedding), text-only (zero out image embedding), and both. Report accuracy for each. This tells you the marginal contribution of each modality.

**Part 4 — Throughput profiling:** Measure inference throughput (images/second) on your GPU for batch sizes 1, 8, 32, 128. Identify the bottleneck. Produce a throughput vs batch size plot.

**Part 5 — Written design document:** A 2-page `design_doc.md` covering: architecture decisions and their justifications, what you would change if you had 10× more compute, and what the main failure mode of template-based text descriptions is vs per-image captions.

### What to implement

1. `multimodal_dataset.py` — EuroSAT with text description attached to each sample
2. `multimodal_model.py` — `MultimodalClassifier` with fusion MLP
3. `train.py` — two-stage training (frozen encoders → full fine-tune)
4. `evaluate.py` — unimodal and multimodal accuracy comparison
5. `profile.py` — throughput and memory measurements

### Deliverables

- [ ] All source files above
- [ ] `accuracy_comparison.txt` — table: image-only, text-only, multimodal (frozen), multimodal (fine-tuned)
- [ ] `throughput_plot.png`
- [ ] `design_doc.md`
- [ ] Multimodal accuracy ≥ 95% (the text descriptions make some ambiguous classes trivial)

---

## Starter Code

```python
# multimodal_dataset.py  — fill in TODO
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

DESCRIPTIONS = {
    'AnnualCrop':     'Aerial view of agricultural fields with annual crop rows',
    'Forest':         'Dense tree canopy covering the ground in an aerial satellite image',
    'HerbaceousVegetation': 'Low vegetation and grass covering open fields',
    'Highway':        'A paved highway or motorway visible from above',
    'Industrial':     'Industrial buildings and factory facilities from aerial view',
    'Pasture':        'Open grassland or pasture used for grazing',
    'PermanentCrop':  'Permanent crops such as orchards or vineyards from above',
    'Residential':    'Residential area with houses and streets in a satellite image',
    'River':          'A river or stream flowing through the landscape',
    'SeaLake':        'A body of water such as a sea, lake or reservoir'
}

class MultimodalEuroSAT(Dataset):
    def __init__(self, image_paths, labels, class_names, image_transform, tokenizer, max_text_len=64):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = image_transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # TODO: load image, apply transform
        # TODO: get text description for class
        # TODO: tokenize text
        # Return dict: {'image': tensor, 'input_ids': tensor,
        #               'attention_mask': tensor, 'label': int}
        raise NotImplementedError
```

```python
# multimodal_model.py  — fill in TODOs
import torch
import torch.nn as nn
import timm
from transformers import BertModel

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes: int = 10,
                 image_encoder_dim: int = 384,
                 text_encoder_dim: int = 768,
                 fusion_hidden_dim: int = 512,
                 freeze_encoders: bool = True):
        super().__init__()

        # Image encoder: DeiT-3-Small (pretrained, load from A08 checkpoint if available)
        self.image_encoder = timm.create_model(
            'deit3_small_patch16_224', pretrained=True, num_classes=0
        )
        # Text encoder: BERT-base
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

        if freeze_encoders:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # TODO: define fusion MLP
        # input: image_encoder_dim + text_encoder_dim
        # hidden: fusion_hidden_dim
        # output: num_classes
        raise NotImplementedError

    def forward(self, images, input_ids, attention_mask,
                drop_image: bool = False, drop_text: bool = False):
        # TODO: encode image → (B, image_encoder_dim)
        # TODO: encode text  → (B, text_encoder_dim) using CLS token
        # TODO: apply modality dropout if requested
        # TODO: concatenate and pass through fusion MLP
        raise NotImplementedError
```

```python
# profile.py — throughput measurement
import torch
import time
from multimodal_model import MultimodalClassifier

def measure_throughput(model, batch_size: int, device: str, n_runs: int = 20):
    model.eval().to(device)
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    input_ids = torch.ones(batch_size, 64, dtype=torch.long, device=device)
    attn_mask  = torch.ones(batch_size, 64, dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(images, input_ids, attn_mask)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(images, input_ids, attn_mask)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = (n_runs * batch_size) / elapsed
    print(f"Batch size {batch_size:4d}: {throughput:7.1f} images/sec")
    return throughput

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultimodalClassifier()
    for bs in [1, 8, 32, 128]:
        try:
            measure_throughput(model, bs, device)
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {bs}: OOM")
            break
```

---

## Notes

The template text descriptions make EuroSAT almost trivially easy for the text encoder alone — BERT can classify from text description with near-perfect accuracy. The interesting baseline is therefore image-only vs multimodal, not text-only vs multimodal. If your multimodal model underperforms image-only, your fusion layer is not learning correctly.

The design document is not an afterthought. The ability to articulate architectural decisions clearly and identify limitations is a core engineering skill. Treat it seriously — it is worth revisiting after writing all the code.

For a more challenging extension: replace the template descriptions with per-image captions generated by BLIP-2 (`Salesforce/blip2-opt-2.7b` on HuggingFace). This makes the text encoder actually useful for disambiguating individual images rather than just encoding class identity.
