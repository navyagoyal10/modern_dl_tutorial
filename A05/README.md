# A05 — Object Detection: YOLOv8 with a Custom SE-Enhanced Detection Head

## Overview

Classification assigns a single label to an entire image. Detection is harder: find every object in the image, draw a bounding box around it, and classify it. This requires the model to simultaneously solve "where?" and "what?".

This assignment fine-tunes YOLOv8 on a small aerial detection dataset and then replaces its detection head with an SE-enhanced version using the SE block you built in A03. The goal is to connect your architectural knowledge to a real industrial-grade framework, understand what the detection head actually does, and measure whether your modification helps.

---

## Theory

### Object detection fundamentals

An object detector must predict, for each object: a bounding box $(x, y, w, h)$ and a class label $c$. Two broad families exist:

**Two-stage detectors** (Faster R-CNN): First propose candidate regions (Region Proposal Network), then classify and refine each proposal. High accuracy, slower.

**Single-stage detectors** (YOLO, SSD): Directly predict boxes and classes from a feature map grid. Faster, slightly lower accuracy on small objects.

### YOLO architecture overview

YOLO ("You Only Look Once") divides the image into a grid. Each grid cell predicts:
- $B$ bounding boxes, each with $(x, y, w, h, \text{objectness score})$
- $C$ class probabilities

Modern versions (YOLOv5+) use anchor-free detection and a three-scale prediction head. YOLOv8 specifically uses a decoupled head — separate branches for classification and regression — which was shown to improve training stability.

The backbone extracts features at multiple scales. A neck (typically a Path Aggregation Network, PAN) fuses features across scales. The head makes predictions at each scale.

### Detection loss

YOLOv8 uses three loss components:

**Classification loss:** Binary cross-entropy for each class at each predicted box:

$$\mathcal{L}_{\text{cls}} = -\sum_c \left[y_c \log \hat{p}_c + (1 - y_c) \log(1 - \hat{p}_c)\right]$$

**Box regression loss:** IoU-based loss. IoU (Intersection over Union) between predicted box $B_p$ and ground truth $B_g$ is:

$$\text{IoU} = \frac{|B_p \cap B_g|}{|B_p \cup B_g|}$$

A higher IoU means better overlap. DFL (Distribution Focal Loss) used in YOLOv8 regresses a probability distribution over possible edge locations, not a point estimate — this is more numerically stable.

**Total loss:**

$$\mathcal{L} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}$$

### Mean Average Precision (mAP)

The standard detection metric. For a single class:
1. Rank all predictions by confidence score
2. Compute precision and recall at each threshold
3. Average precision (AP) = area under the precision-recall curve

mAP averages AP across all classes. mAP@50 evaluates at IoU threshold 0.5 (a box is "correct" if IoU ≥ 0.5 with the ground truth). mAP@50-95 averages over IoU thresholds from 0.5 to 0.95 in steps of 0.05 — a much stricter metric.

### Where to insert SE blocks

The YOLOv8 detection head consists of convolutional layers that produce class and box predictions. Inserting SE blocks after the final convolutional stage of the head allows the model to recalibrate channel importance specifically for the detection task — separate from the backbone, which was pretrained for classification.

This is architecturally motivated: the detection head sees feature maps that already encode "what is where" information. SE recalibration at this stage allows the head to selectively amplify channels that are most discriminative for the specific classes in your dataset.

To modify the YOLOv8 head, you access `model.model[-1]` (the `Detect` module) and insert your SE block before the final convolutional layer in each of the three scale branches.

### Aerial detection specifics

Aerial images have several detection challenges:
- Objects are small relative to image size (cars, trees, buildings)
- High density of objects (many instances per image)
- Top-down perspective removes occlusion but introduces scale variation

These properties make Focal Loss (from A03) particularly relevant here — there are many background patches (easy negatives) and few object patches.

---

## Reading Material

- Redmon et al., "You Only Look Once" (original YOLO, 2016): https://arxiv.org/abs/1506.02640
- Jocher et al., "Ultralytics YOLOv8" (2023): https://github.com/ultralytics/ultralytics
- Lin et al., "Feature Pyramid Networks for Object Detection" (2017): https://arxiv.org/abs/1612.03144
- Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet, 2017): https://arxiv.org/abs/1708.02002
- Hu et al., "Squeeze-and-Excitation Networks" (2018): https://arxiv.org/abs/1709.01507
- DOTA dataset paper: https://arxiv.org/abs/1711.10398

---

## Assignment

### Dataset

DOTA-v1.0 (small subset) or VisDrone2019 (vehicle detection from drones). VisDrone is recommended — it is smaller, has a clear download script, and is directly relevant (aerial vehicles). We provide a download script for the 10-class subset.

### Task

**Part 1 — Baseline YOLOv8:** Fine-tune `yolov8s` (small variant, ~11M params) on the aerial dataset for 50 epochs. Report mAP@50 and mAP@50-95 on the validation set.

**Part 2 — SE-enhanced head:** Modify the YOLOv8 detection head to include SE blocks (reuse `SEBlock` from A03) in each of the three scale branches. Fine-tune from the same pretrained backbone for 50 epochs. Report metrics and compare against Part 1.

**Part 3 — Analysis:** Compute per-class AP. Identify which classes benefited most from SE enhancement and hypothesise why (consider class size, texture uniformity, confusion pairs).

### What to implement

1. Dataset download and conversion to YOLO format
2. `SEDetectHead` — modified YOLOv8 detection head with SE blocks
3. Fine-tuning script using Ultralytics API
4. Custom model surgery to swap the head
5. Evaluation and per-class AP extraction

### Deliverables

- [ ] `download_data.py` — downloads VisDrone, converts to YOLO format
- [ ] `se_detect_head.py` — modified head definition
- [ ] `train_baseline.py` — vanilla YOLOv8 fine-tuning
- [ ] `train_se.py` — SE-head fine-tuning
- [ ] `evaluate.py` — mAP report + per-class AP table
- [ ] `map_comparison.png` — validation mAP curve over epochs (both models)
- [ ] SE head mAP@50 ≥ baseline mAP@50 − 1% (should be at least comparable)

---

## Starter Code

```python
# download_data.py — VisDrone subset
import os
import urllib.request
import zipfile

URLS = {
    'train': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip',
    'val':   'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip',
}

os.makedirs('./data/visdrone', exist_ok=True)
for split, url in URLS.items():
    dest = f'./data/visdrone/{split}.zip'
    print(f"Downloading {split}...")
    urllib.request.urlretrieve(url, dest)
    with zipfile.ZipFile(dest, 'r') as z:
        z.extractall('./data/visdrone/')
    print(f"  Done: {split}")

# Convert VisDrone annotation format to YOLO format
# VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
# YOLO format: <class> <x_center> <y_center> <width> <height>  (all normalised 0-1)
def convert_visdrone_to_yolo(ann_path, img_w, img_h, out_path):
    # TODO: implement format conversion
    pass
```

```python
# se_detect_head.py  — fill in TODO
# You will need to inspect YOLOv8's Detect module source code first.
# Run: python -c "from ultralytics.nn.modules import Detect; import inspect; print(inspect.getsource(Detect))"

import torch.nn as nn
# from your A03 work:
# from se_block import SEBlock

class SEDetectHead(nn.Module):
    """
    Modified YOLOv8 detection head with SE blocks inserted in each scale branch.
    Inherits structure from Detect but adds SE recalibration before final conv.
    """
    def __init__(self, original_detect_module, reduction: int = 16):
        super().__init__()
        # TODO: copy all submodules from original_detect_module
        # TODO: add SE blocks matching channel widths of each scale branch
        raise NotImplementedError

    def forward(self, x):
        # x: list of 3 feature maps from neck, at 3 scales
        # TODO: apply SE block to each scale, then call original detection logic
        raise NotImplementedError
```

```python
# swap_head.py — utility to replace YOLOv8 head
from ultralytics import YOLO
# from se_detect_head import SEDetectHead

def load_se_yolo(weights: str = 'yolov8s.pt', reduction: int = 16):
    model = YOLO(weights)
    # TODO: access model.model[-1] (the Detect module)
    # TODO: replace with SEDetectHead(original_head, reduction)
    # TODO: verify output shape is unchanged with a dummy forward pass
    return model
```

---

## Notes

The Ultralytics library wraps YOLOv8 with a high-level API. You will need to bypass this slightly for head surgery. The model's internal PyTorch module is at `model.model` (a `DetectionModel`), and the detect head is the last element `model.model.model[-1]`.

After surgery, verify the forward pass with a dummy input before training:
```python
import torch
from ultralytics import YOLO
m = YOLO('yolov8s.pt')
dummy = torch.randn(1, 3, 640, 640)
out = m.model(dummy)
print([o.shape for o in out[0]])  # Should be 3 tensors at different scales
```

If mAP drops significantly after SE insertion, your SE block is likely disrupting weight initialisation. Apply He initialisation to the new SE layers and verify the scale outputs are not zero.
