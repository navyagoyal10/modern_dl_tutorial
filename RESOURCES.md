# Supplementary Resources for All 12 Assignments

A curated, annotated list of blogs, papers, visualisations, and theoretical reads to deepen your understanding beyond what the READMEs cover. Organised by assignment. Where a resource spans multiple assignments, it appears under the most relevant one with a cross-reference note.

---

## A01 — PyTorch Training Loop / CNNs

### Theory and intuition

**Universal Approximation Theorem — IFT 6085 lecture notes (Montreal)**
https://mitliagkas.github.io/ift6085-2020/ift-6085-lecture-10-notes.pdf
The cleanest written derivation of the UAT you will find. Covers width-bounded vs depth-bounded networks, sawtooth functions, and why depth is exponentially more efficient than width for some function classes. Directly answers "why do we need deep networks at all?" Read sections 1–4.

**Universality of Deep CNNs — Ding-Xuan Zhou (2018)**
https://arxiv.org/abs/1805.10769
The paper that formally proved CNNs (not just MLPs) are universal approximators. The key result: a deep CNN can approximate any continuous function when depth is large enough. The parameter count scales polynomially in depth rather than exponentially — this is the formal justification for why depth helps.

**CS231n Convolutional Networks notes — Stanford**
https://cs231n.github.io/convolutional-networks/
Definitive reference for CNN mechanics: receptive fields, pooling, spatial dimensions after convolution, parameter counting. Bookmark this and refer back constantly during A01–A03.

**Feature Visualization — Olah et al., Distill (2017)**
https://distill.pub/2017/feature-visualization/
What do CNN neurons actually detect? This article shows, via activation maximisation, the features learned at each layer. Early layers learn Gabor filters and colour blobs; later layers learn textures, parts, and objects. Directly contextualises *why* transfer learning from ImageNet works (A02).

**Zoom In: An Introduction to Circuits — Distill (2020)**
https://distill.pub/2020/circuits/zoom-in/
Reverse-engineering a CNN at the individual neuron and weight level. Shows that CNNs learn a compositional hierarchy — curve detectors built from edge detectors, object detectors built from curve detectors. Dense, slow reading, but nothing builds intuition for CNN feature hierarchies better.

### Visualisation tools

**CNN Explainer (interactive)**
https://poloclub.github.io/cnn-explainer/
Interactive, in-browser visualisation of a small CNN processing an image. Watch activations propagate layer by layer. Use this alongside A01 to confirm your mental model of what each layer does.

---

## A02 — Transfer Learning and ResNet Fine-tuning

### Theory and intuition

**How transferable are features in deep neural networks? — Yosinski et al. (2014)**
https://arxiv.org/abs/1411.1792
The empirical foundation of transfer learning. Measures how well features from each layer of AlexNet transfer to a different task. Key finding: features become progressively more task-specific with depth; early features are nearly universal. This is the justification for freezing early layers.

**The Building Blocks of Interpretability — Distill (2018)**
https://distill.pub/2018/building-blocks/
What does the pretrained ResNet actually know? This article lets you explore which neurons activate for which concepts in GoogLeNet (closely related architecture). After reading this, "pretrained features" stops being an abstraction.

**Deep Residual Learning — He et al. (2016)**
https://arxiv.org/abs/1512.03385
Read sections 1–3 only. The original ResNet paper. Understand the residual identity shortcut and its gradient-flow motivation formally: $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}(1 + \frac{\partial \mathcal{F}}{\partial x})$ — the +1 prevents vanishing.

**Batch Normalization — Ioffe & Szegedy (2015)**
https://arxiv.org/abs/1502.03167
ResNet relies on BatchNorm. Read Section 2 to understand what it does and why the train/eval mode distinction matters when fine-tuning.

### Practical

**PyTorch Transfer Learning Tutorial (official)**
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
The canonical implementation reference. Cross-reference with your own code during A02.

---

## A03 — SE Blocks, Focal Loss, Grad-CAM

### Theory and intuition

**Squeeze-and-Excitation Networks — Hu et al. (2018)**
https://arxiv.org/abs/1709.01507
Read carefully: sections 2–3 give the full derivation of squeeze, excitation, and the scale operation. Pay attention to Table 2 (ablation study showing reduction ratio $r=16$ as the sweet spot). The SE block adds ~2% extra parameters for 1% top-1 improvement on ImageNet — one of the best efficiency gains in architecture literature.

**Focal Loss for Dense Object Detection — Lin et al. (2017)**
https://arxiv.org/abs/1708.02002
Section 3 derives Focal Loss from scratch. The key insight: the modulating factor $(1-p_t)^\gamma$ down-weights easy examples so that the gradient signal is dominated by hard misclassifications. Read the ablation (Table 1) showing how $\gamma$ affects the contribution of different $p_t$ ranges.

**Grad-CAM — Selvaraju et al. (2017)**
https://arxiv.org/abs/1610.02391
The original paper is clearly written. Section 3 derives the gradient-weighted class activation mapping. The key step is understanding why global average pooling over the gradients gives the correct channel importance weight $\alpha_k^c$. Also see Figure 5 — class-discriminative localisation vs standard CAM.

**Explaining and Harnessing Adversarial Examples — Goodfellow et al. (2014)**
https://arxiv.org/abs/1412.6572
A conceptually related technique: adversarial examples are found by maximising class score w.r.t. input rather than w.r.t. weights. Reading this alongside Grad-CAM sharpens your understanding of what gradients w.r.t. inputs actually mean.

### Visualisation

**Lucid / Feature Visualisation playground**
https://distill.pub/2017/feature-visualization/
See the "Feature Visualization as Optimization" section. This shows what the filters in each layer are *maximally* sensitive to — the complement to Grad-CAM, which shows *which input region* activates a given output.

---

## A04 — Contrastive Learning (SimCLR)

### Theory and intuition

**SimCLR — Chen et al. (2020) [primary paper]**
https://arxiv.org/abs/2002.05709
Read sections 2–4. Section 4 is the ablation that explains *why* each component matters: composition of augmentations is critical; the projection head is essential; batch size matters because more negatives = better signal. The appendix A gives the full augmentation recipe.

**Understanding Contrastive Learning (variational analysis) — OpenReview 2024**
https://openreview.net/forum?id=qjoDJjVZxB
More rigorous than most blogs: proves that the NT-Xent loss alone has degenerate minimisers, and that the training dynamics of neural networks are what make contrastive learning work in practice. Read if you want to understand *why* it works, not just *that* it works.

**Thalles Silva's SimCLR walkthrough (blog)**
https://sthalles.github.io/simple-self-supervised-learning/
The most complete practical blog on implementing SimCLR. Walks through each component with code. The linear probe evaluation section is particularly useful for A04.

**Lightly AI — SimCLR Explained**
https://www.lightly.ai/blog/simclr
Cleanest conceptual overview of the NT-Xent loss and the "instance discrimination" framing. Good starting read before the paper.

**Bootstrap Your Own Latent (BYOL) — Grill et al. (2020)**
https://arxiv.org/abs/2006.07733
Contrastive learning *without* negative pairs. Read after SimCLR to understand what role negatives actually play — BYOL shows they are not strictly necessary, which challenges the simple "push/pull" explanation of why contrastive learning works.

**Understanding UMAP — McInnes et al.**
https://pair-code.github.io/understanding-umap/
Interactive UMAP visualisation tool. Understand the effect of `n_neighbors` and `min_dist` parameters *before* you use UMAP to plot your SimCLR embeddings — otherwise your cluster plots will be misleading.

---

## A05 — YOLOv8 with SE Detection Head

### Theory and intuition

**You Only Look Once — Redmon et al. (2016) [original YOLO]**
https://arxiv.org/abs/1506.02640
Section 2 explains the grid-based detection formulation. Even though you're using YOLOv8 (which is architecturally very different), understanding the original formulation clarifies the conceptual framework. The grid-cell and anchor box intuition carries through all YOLO versions.

**Feature Pyramid Networks — Lin et al. (2017)**
https://arxiv.org/abs/1612.03144
YOLOv8's neck is FPN-based. This paper explains why multi-scale feature fusion is essential for detecting objects at different sizes. The "bottom-up pathway" and "top-down pathway with lateral connections" diagram (Figure 1) should be understood before doing head surgery on YOLO.

**IoU Loss variants — a blog survey**
https://leimao.github.io/blog/IoU-Loss/
Clear derivation of IoU, GIoU, DIoU, CIoU, and the Distribution Focal Loss (DFL) used in YOLOv8. Understand this before reading YOLOv8's loss implementation.

**Ultralytics YOLOv8 architecture docs**
https://docs.ultralytics.com/models/yolov8/
The official documentation. Read the "Architecture" section to understand what `model.model[-1]` (the `Detect` module) contains and where to insert the SE block.

---

## A06 — Attention Mechanism from Scratch

### Theory and intuition — the essential sequence, read in order

**3Blue1Brown — Attention in Transformers (video + written)**
https://www.3blue1brown.com/lessons/attention
The single best visual introduction to attention. Grant Sanderson's animations show Q/K dot products, the attention matrix heatmap, and multi-head splitting more clearly than any written derivation. Watch first, then read the paper.

**The Illustrated Transformer — Jay Alammar**
https://jalammar.github.io/illustrated-transformer/
The written complement to 3B1B. Walks through every matrix multiplication in multi-head attention with annotated diagrams. Bookmark this as your reference while implementing A06 — every time you are unsure of a shape, check here.

**Transformer Explainer (interactive, live GPT-2)**
https://poloclub.github.io/transformer-explainer/
Run a real GPT-2 model in your browser. Watch the attention matrices change as you type. Click on individual heads to see their patterns. Use after implementing A06 to check whether your attention maps look qualitatively similar.

**Attention Is All You Need — Vaswani et al. (2017)**
https://arxiv.org/abs/1706.03762
Read sections 3.1–3.3 carefully. The derivation of scaled dot-product attention and the explanation of why $1/\sqrt{d_k}$ scaling is needed (Section 3.2.1) is the mathematical core of your A06 implementation. The multi-head formulation in eq. 4–5 maps directly to your code.

**Why does the Transformer use $\sqrt{d_k}$ scaling? (formal argument)**
For vectors $\mathbf{q}, \mathbf{k}$ drawn i.i.d. from $\mathcal{N}(0, 1)$, the dot product $\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$ (sum of $d_k$ independent unit-variance terms). Without scaling, large $d_k$ pushes scores into softmax saturation where $\partial \text{softmax} / \partial z \approx 0$. Dividing by $\sqrt{d_k}$ restores unit variance. Vaswani et al. Section 3.2.1 covers this — but working through the variance calculation yourself is more illuminating than just reading it.

**Flash Attention — Dao et al. (2022)**
https://arxiv.org/abs/2205.14135
Read sections 1–2 only (skip the CUDA details). This paper reframes attention as an IO-bound problem, not a compute-bound one, and shows why the naive $O(L^2)$ attention is practically slow even when theoretical FLOPs are manageable. Essential context for understanding why attention at scale is hard.

**Transformer Circuits Thread — Anthropic (ongoing)**
https://transformer-circuits.pub/
What do attention heads actually learn? This research thread reverse-engineers individual transformer layers. The "A Mathematical Framework for Transformer Circuits" paper is the best formal treatment of what residual streams and attention heads are doing algorithmically.

---

## A07 — BPE Tokenizer + Decoder Language Model

### Theory and intuition

**Karpathy — Let's build GPT from scratch (video)**
https://www.youtube.com/watch?v=kCc8FmEb1nY
2 hours. Karpathy builds a complete GPT character-level language model from scratch, live. Watch sections 1 and 4–5 for the decoder architecture. If you watch one video for this entire curriculum, make it this one.

**build-nanoGPT repo (step-by-step commits)**
https://github.com/karpathy/build-nanogpt
Every git commit is a single conceptual addition to the model. Walk through the commit history to see the model built incrementally. The clearest existing reference implementation for A07.

**Neural BPE — Sennrich et al. (2016)**
https://arxiv.org/abs/1508.07909
The original BPE-for-NLP paper. Section 3 gives the algorithm in 8 lines of pseudocode. Understand the greedy merge logic before implementing `digram_stats` and `_atomic_merge_step`.

**The Illustrated GPT-2 — Jay Alammar**
https://jalammar.github.io/illustrated-gpt2/
Walks through GPT-2 forward pass with detailed diagrams of token embeddings, positional encodings, causal masking, and the language model head. Your A07 model is a scaled-down version of this.

**GELU activation — Hendrycks & Gimpel (2016)**
https://arxiv.org/abs/1606.08415
Section 2 derives GELU from a stochastic regularisation perspective ($x \cdot \Phi(x)$ is the expected value of a stochastic unit that is dropped with probability $\Phi(-x)$). Transformer FFNs use GELU because it avoids the "dying ReLU" problem while remaining differentiable.

**What is perplexity? A blog post with worked examples**
https://huggingface.co/docs/transformers/perplexity
HuggingFace's perplexity explainer. Concretely shows how perplexity is computed from cross-entropy loss and why it is the correct metric for language models. Read before you start logging perplexity in A07.

---

## A08 — DeiT-3 Fine-tuning + Attention Maps

### Theory and intuition

**An Image is Worth 16×16 Words — Dosovitskiy et al. (2020)**
https://arxiv.org/abs/2010.11929
The original ViT paper. Read Section 3 for the patch embedding formulation and Section 4 for the ablation showing ViT needs JFT-300M to beat ResNet — and why DeiT fixed this.

**CNN vs ViT — inductive bias analysis**
https://tobiasvanderwerff.com/2024/05/15/cnn-vs-vit.html
A well-structured practitioner comparison. The key figure: ViTs integrate information globally from *layer 1* (measured by average attention distance across patches), while CNNs expand their receptive field gradually. This is why Grad-CAM (CNN) is localised while ViT attention maps are diffuse.

**Emerging Properties in Self-Supervised ViTs (DINO) — Caron et al. (2021)**
https://arxiv.org/abs/2104.14294
DINO's self-supervised ViTs produce attention maps that explicitly segment objects without any segmentation training. Figures 1 and 7 show the most striking ViT attention maps in the literature. Compare with your A08 fine-tuned maps to see how supervision changes what gets attended to.

**Attention Rollout — Abnar & Zuidema (2020)**
https://arxiv.org/abs/2005.00928
Derives the attention rollout formula for propagating attention through layers. Section 3 is the derivation — 2 pages. Read if you want to go beyond last-layer attention maps. The claim is that last-layer CLS attention underestimates the contribution of early-layer attention due to residual connections.

**BertViz — interactive BERT attention visualisation**
https://github.com/jessevig/bertviz
An interactive tool that visualises multi-head attention at neuron, head, and model level. Primarily for BERT (A10) but directly applicable to any transformer. The "head view" shows which heads learn syntactic patterns vs positional patterns — essential reading before A08 and A10.

---

## A09 — Dynamic Tanh (Transformers without Normalisation)

### Theory and intuition

**Transformers without Normalization — Zhu et al. (CVPR 2025)**
https://arxiv.org/abs/2503.10622
The primary paper. Read Section 3.2 (the empirical observation that LayerNorm input-output looks like tanh) and Section 3.3 (the DyT formulation). Figure 2 is the key visualisation: plot LayerNorm's input vs output for a trained model — the S-shaped curve is the central claim.

**Layer Normalization — Ba et al. (2016)**
https://arxiv.org/abs/1607.06450
Section 2 derives LayerNorm from scratch and explains why it stabilises training in recurrent networks. Sections 2–3 give the gradient analysis. Read to understand *what exactly* DyT is replacing and why it is non-trivial that a simple tanh can substitute it.

**Why normalization layers are hard to remove (blog)**
https://sh-tsang.medium.com/review-layer-normalization-ln-for-deep-learning-bd2f37d62aca
A concise summary of why Layer Normalisation was introduced, what problem it solves, and what training looks like without it. Provides empirical context for understanding what the DyT paper claims to have overcome.

**DyT reference implementation — Zhu et al.**
https://github.com/jiachenzhu/DyT
The authors' own PyTorch implementation. Cross-reference your `replace_layernorm_with_dyt` function with their implementation. They provide a `convert_model` utility — compare it against yours.

---

## A10 — BERT, Bidirectional Attention, NER

### Theory and intuition

**BERT — Devlin et al. (2018)**
https://arxiv.org/abs/1810.04805
Read Sections 3.1 (model architecture), 3.2 (pretraining procedure — MLM and NSP), and 4 (fine-tuning procedure). The key insight in Section 3.2: MLM forces bidirectional representations because the model cannot simply "copy" a left-to-right LM — it must use both directions to predict masked tokens.

**The Illustrated BERT — Jay Alammar**
https://jalammar.github.io/illustrated-bert/
The indispensable visual walkthrough. Explains the [CLS] and [SEP] tokens, the fine-tuning head variants for different tasks (classification vs sequence labelling vs span prediction), and how pre-training representations transfer. Read before touching A10 code.

**BertViz — attention visualisation**
https://github.com/jessevig/bertviz
Essential for A10's attention head analysis deliverable. The "neuron view" shows individual Q/K/V values; the "head view" shows which tokens each head attends to; the "model view" shows all heads across all layers. You will find heads that clearly encode positional patterns, [SEP] attention, syntactic dependency.

**A Structural Probe for Finding Syntax in Word Representations — Hewitt & Manning (2019)**
https://arxiv.org/abs/1905.06316
Do BERT's representations encode syntactic structure? This paper probes BERT's hidden states to show that syntactic trees are linearly encoded in the geometry of the embedding space. Context for A10's attention analysis: attention patterns and linear probe geometry are two different windows into BERT's representations.

**seqeval documentation**
https://github.com/chakki-works/seqeval
The library used in A10 for NER evaluation. Read the README to understand the distinction between token-level F1 (which is naive and wrong for NER) and entity-level F1 (which correctly handles B-I-O spans). Your `evaluate_ner.py` must use entity-level F1.

---

## A11 — LoRA and Parameter-Efficient Fine-Tuning

### Theory and intuition

**LoRA — Hu et al. (2021)**
https://arxiv.org/abs/2106.09685
Read Sections 1–5. Section 7 (understanding the low-rank updates) is often skipped but is theoretically important: it shows that $\Delta W$ has much lower intrinsic rank than $W$, that the update direction is mostly orthogonal to the pretrained weights, and that rank $r=1$ is often sufficient.

**Intrinsic Dimensionality — Aghajanyan et al. (2020)**
https://arxiv.org/abs/2012.13255
The theoretical precursor to LoRA. Proves that fine-tuning can be done in a surprisingly small subspace of the full parameter space. The "intrinsic dimension" concept — the minimum dimensionality needed to solve a task — is the mathematical foundation for why LoRA works.

**QLoRA — Dettmers et al. (2023)**
https://arxiv.org/abs/2305.14314
Combines LoRA with 4-bit NormalFloat quantisation. Section 2.3 explains NF4 quantisation: it quantises weights into 16 discrete levels chosen so that each level covers an equal fraction of a standard normal distribution (optimal for weights that are approximately Gaussian). This is what makes 7B fine-tuning possible on a 12GB GPU.

**Fundamentals of LoRA — Nebius blog**
https://nebius.com/blog/posts/fine-tuning/lora-low-rank-adaptation
The best deep-dive blog on LoRA theory. Covers intrinsic dimensionality, the difference between LoRA and adapter layers, and recent variants (DoRA, PiSSA). The DoRA section on direction vs magnitude decomposition is particularly interesting.

**TDS — Understanding LoRA**
https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6
A clean walkthrough of the intrinsic rank hypothesis with diagrams showing the $\Delta W = BA$ decomposition. Useful as a second pass after reading the paper.

**Unsloth documentation**
https://docs.unsloth.ai
Read the "How Unsloth works" section to understand how it achieves 2× memory reduction vs standard PEFT: manual kernel fusion in Triton, removing redundant memory allocations during backward passes, and using flash attention. You do not need to understand the CUDA details — understanding *what* is being optimised is sufficient.

---

## A12 — Multimodal Classification (Capstone)

### Theory and intuition

**CLIP — Radford et al. (2021)**
https://arxiv.org/abs/2103.00020
The landmark paper on vision-language alignment. CLIP trains a joint embedding space by contrasting image-text pairs — the same NT-Xent loss as SimCLR (A04) but across modalities. Read Section 2 for the training objective and Section 3 for the remarkable zero-shot transfer results. Your A12 late-fusion pipeline is a simpler version of the ideas here.

**Supervised Multimodal Bitransformers — Kiela et al. (2019)**
https://arxiv.org/abs/1909.02950
A clean paper on late fusion for image+text classification. Figure 1 shows the architecture you are building in A12. Section 3.2 compares late fusion vs early fusion vs cross-attention — empirically, late fusion is competitive with much more complex approaches when both modalities are independently strong.

**BLIP — Li et al. (2022)**
https://arxiv.org/abs/2201.12086
If you choose the extension of generating per-image captions using BLIP-2 instead of templates: read Section 2 for the captioning pipeline. BLIP-2 (the updated version) is at `Salesforce/blip2-opt-2.7b` on HuggingFace.

**PyTorch Profiler tutorial**
https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
Required reading before implementing A12's `profile.py`. The profiler shows per-operator time and memory allocation, letting you identify whether the bottleneck is the ViT encoder, BERT encoder, or fusion MLP. In practice, BERT (768-dim, 12 layers) dominates compute time over DeiT-3-Small (384-dim).

### JEPA — an alternative multimodal philosophy

**I-JEPA — Assran et al., Meta AI (2023)**
https://arxiv.org/abs/2301.08243
Image Joint Embedding Predictive Architecture. Instead of predicting pixels (MAE) or contrasting augmented views (SimCLR/CLIP), I-JEPA predicts the *representation* of masked image regions from context regions — entirely in embedding space. The key insight: predicting abstract representations forces the model to learn semantic content rather than low-level texture. Figure 2 shows the architecture clearly. I-JEPA beats MAE and DeiT on linear probe benchmarks with less compute. Directly relevant to A12 because it offers a third paradigm for the image encoder — one that requires no text supervision and no augmentation engineering.

**V-JEPA — Bardes et al., Meta AI (2024)**
https://arxiv.org/abs/2404.08471
The video extension of I-JEPA. Predicts future frame representations from past frames in latent space rather than pixel space. The temporal prediction objective forces the model to learn object permanence and motion dynamics — properties that static image encoders lack. The paper's Table 1 shows that V-JEPA features transfer strongly to action recognition and object tracking benchmarks without any fine-tuning. Relevant context: if your A12 pipeline ever extends to video input, JEPA-style encoders are the current best-practice for the image/video modality.

**World Models — LeCun (2022 position paper)**
https://arxiv.org/abs/2206.07682
The theoretical motivation behind the entire JEPA family. LeCun argues that energy-based, joint-embedding, latent-variable models are the path toward human-level AI — and that generative pixel-prediction models (GPT, DALL-E) are the wrong direction. Whether or not you agree with the thesis, Section 4 (the "energy-based model" framing) gives you the conceptual vocabulary to understand why I-JEPA predicts in representation space rather than pixel space. Read this to understand what JEPA is *trying* to be, not just what it does.

---

## Cross-Cutting Resources (relevant to multiple assignments)

### Theoretical foundations

**The Bitter Lesson — Rich Sutton (2019)**
http://www.incompleteideas.net/IncIdeas/BitterLesson.html
One page. The argument that general methods (scale + compute) consistently beat domain-specific engineered methods in AI. Read this after A03 (SE blocks as human-engineered features) and A04 (self-supervised learning vs labelled data). Every architectural decision in this curriculum exists in the context of this tension.

**Neural Networks: Zero to Hero — Karpathy (lecture series)**
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
The lecture series that the A06/A07 assignments are inspired by. Watch the micrograd video (backprop from scratch), the makemore series (n-grams to attention), and the nanoGPT video (full transformer). ~12 hours total. Worth every minute.

**Deep Learning — Goodfellow, Bengio, Courville (book, free online)**
https://www.deeplearningbook.org/
Chapters 6 (MLPs), 9 (CNNs), 10 (sequence models). The standard graduate-level reference. Use as a dictionary: look up specific concepts when the READMEs reference them without full derivation. The Chapter 8 section on Adam is the cleanest formal derivation of the optimiser.

### Visualisation and debugging

**CNN Explainer**
https://poloclub.github.io/cnn-explainer/
Interactive CNN forward pass (A01–A03).

**Transformer Explainer (live GPT-2)**
https://poloclub.github.io/transformer-explainer/
Interactive transformer forward pass (A06–A08).

**Weights & Biases (experiment tracking)**
https://docs.wandb.ai/quickstart
Use `wandb` to log training curves, hyperparameter sweeps, and model checkpoints across A01–A12. The visualisation dashboard makes comparing experiments (frozen vs fine-tuned in A02, CE vs FL in A03, rank sweep in A11) much easier than matplotlib plots.

**TensorBoard embedding projector**
https://projector.tensorflow.org/
Works with PyTorch via `torch.utils.tensorboard`. Use it instead of (or alongside) UMAP in A04 to visualise your SimCLR embeddings in 3D — it supports t-SNE, UMAP, and PCA projection.

### Papers that connect multiple assignments

**A ConvNet for the 2020s — Liu et al. (2022)**
https://arxiv.org/abs/2201.03545
ConvNeXt: takes ViT design decisions (patch embeddings, depthwise separable convolutions, GELU, LayerNorm) and applies them to a pure CNN. The result matches ViT-L performance. Deeply relevant to A08/A09 because it shows the performance gap between CNNs and ViTs is largely about training recipe and design choices, not architectural expressivity. Bridges A03 (CNNs) and A08 (ViTs).

**Masked Autoencoders are Scalable Vision Learners (MAE) — He et al. (2021)**
https://arxiv.org/abs/2111.06377
After SimCLR (A04) and DeiT fine-tuning (A08), read MAE. It shows that masking 75% of image patches and reconstructing them — a much simpler self-supervised task than contrastive learning — learns representations that match or beat SimCLR at scale. Contextualises where the field went after SimCLR.

**Scaling Laws for Neural Language Models — Kaplan et al. (2020)**
https://arxiv.org/abs/2001.08361
After building your small language model in A07, read this. It shows that model loss scales as a power law in parameters, data, and compute — and that these three resource types must be scaled together. This is the paper that justified building GPT-3.

**A Mathematical Framework for Transformer Circuits — Elhage et al. (2021)**
https://transformer-circuits.pub/2021/framework/index.html
After A06 (attention implementation) and A07 (language model training), this is the next level of understanding. It derives, formally, what 1-layer and 2-layer attention-only transformers can compute. The "residual stream" as a communication channel between components is the core conceptual frame.

---

*Total: ~40 resources. Estimated reading time if you follow all primary links: ~45 hours spread across the curriculum. You do not need to read everything before starting — the annotation on each entry tells you exactly when it is relevant.*
