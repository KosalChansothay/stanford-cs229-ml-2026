# CS336 Lecture 17: Alignment — Multimodality

## 0. Quick-Recall Summary
- **Multimodal Tokenization**: Treats non-text inputs (images, audio) as sequences of continuous token embeddings that are projected directly into the LLM's vocabulary space.
- **SigLIP Binary sigmoid Loss**: Replaces CLIP's global softmax loss with a pairwise sigmoid classification loss, decoupling batch size from the objective and enabling efficient parallel scaling.
- **LLaVA Template**: Stitches a frozen CLIP vision encoder to an LLM via a learnable projection matrix $W$, aligning image patches to natural language embedding dimensions.
- **AnyRes (Adaptive Resolution)**: Prevents downsampling information loss by cropping high-resolution images into multiple sub-patches matching the vision encoder's native size, processing them individually.
- **M-RoPE (Multimodal Rotary Embeddings)**: Generalizes RoPE to 3D spaces (Height, Width, Time) by dividing query-key head dimensions into spatial and temporal coordinates.

## 1. Core Paradigm & Systems Overview
- **Objective**: Extend text-only LLM architectures to process, align, and reason across diverse sensory modalities (images, high-resolution document scans, videos).
- **Primary Bottleneck**: Spatial token explosion. High-resolution scans and video frames generate thousands of visual tokens; without cropping and down-sampling, sequence attention matrices overwhelm the quadratic $O(S^2)$ attention window.
- **Builds on**: Positional Embeddings and SFT (Lecture 3 & 15) — leverages 1D Rotary embeddings and SFT alignment frameworks to construct multi-dimensional coordinate spaces.

## 2. Theoretical & Mathematical Primitives

### CLIP Contrastive Loss
Given $N$ normalized image embeddings $I_i$ and text embeddings $T_j$, the cosine similarity matrix is computed as $S\_{i, j} = I_i^T T_j$. The bidirectional loss averages image-to-text and text-to-image cross-entropies:

$$\mathcal{L}\_{\text{CLIP}} = \frac{1}{2} \left( \mathcal{L}\_{I \to T} + \mathcal{L}\_{T \to I} \right)$$

Where:

$$\mathcal{L}\_{I \to T} = -\frac{1}{N} \sum\_{i=1}^N \log \frac{\exp(\tau \cdot S\_{i, i})}{\sum\_{j=1}^N \exp(\tau \cdot S\_{i, j})}$$

### SigLIP Loss
SigLIP converts contrastive matching into independent binary logistic regressions:

$$\mathcal{L}\_{\text{SigLIP}} = -\frac{1}{N} \sum\_{i=1}^N \sum\_{j=1}^N \log \sigma \left( c\_{i, j} (\tau \cdot S\_{i, j} + b) \right)$$

Where:
- $c\_{i, j} = 1$ if $i = j$ (matching positive pair).
- $c\_{i, j} = -1$ if $i \neq j$ (negative pair).
- $b$: A learnable bias parameter.

### Multimodal RoPE (M-RoPE)
Given a 3D visual position $(t, y, x)$ corresponding to temporal, height, and width coordinates, the query vector $q$ is rotated by splitting the query head dimension $D$ into three disjoint parts: $D = D_t + D_y + D_x$.
- The first $D_t$ dimensions are rotated using standard 1D RoPE with coordinate $t$.
- The next $D_y$ dimensions are rotated with coordinate $y$.
- The final $D_x$ dimensions are rotated with coordinate $x$.
- The three rotated sub-vectors are concatenated to construct the final query embedding.

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### LLaVA Alignment and Fine-Tuning Pipeline
1. **Pre-training / Alignment (Stage 1)**: Keep the CLIP vision encoder and the LLM frozen. Train only the adapter projection $W$ on raw caption datasets to map visual feature dimensions to language space.
2. **Instruction Tuning (Stage 2)**: Keep the vision encoder frozen. Fine-tune both the LLM weights and the adapter $W$ on conversational multimodal tasks.
3. **Multimodal RLHF (Stage 3)**: Apply preference optimization (DPO) to refine output style and reduce hallucinations.

### PyTorch/Pythonic Blueprint (Educational AnyRes Patch Cropping)
```python
import torch

def any_res_crop(image, patch_size=336):
    # image shape: (C, H, W)
    _, h, w = image.shape
    
    # 1. Base downsampled image of the entire scene
    base_img = torch.nn.functional.interpolate(
        image.unsqueeze(0), 
        size=(patch_size, patch_size), 
        mode="bilinear"
    ).squeeze(0)
    
    patches = [base_img]
    
    # 2. Extract crops of size patch_size x patch_size
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if y + patch_size <= h and x + patch_size <= w:
                crop = image[:, y:y+patch_size, x:x+patch_size]
                patches.append(crop)
                
    return torch.stack(patches)  # Returns (N_patches, C, patch_size, patch_size)
```

## 4. Hardware Realities & Compute/Memory Accounting
- **The Decoupled Communication Bottleneck**: CLIP's global softmax ties the loss denominator to all ranks in a distributed batch. To calculate gradients, each rank must execute all-to-all collectives across the network. SigLIP's pairwise sigmoid loss decouples ranks: each rank computes its logistic regression locally, eliminating heavy network communications and speeding up training by **2x**.

## 5. Hyperparameter Heuristics

<div id="plotly-cs336-17-siglip-loss" class="plotly-chart" aria-label="Interactive Plotly chart: SigLIP Pairwise Binary Loss"></div>

<p><em>Figure: SigLIP Pairwise Sigmoid Loss decouples batch items, eliminating the global softmax AllGather bottleneck at scale.</em></p>

<div id="plotly-cs336-17-vlm-tokens" class="plotly-chart" aria-label="Interactive Plotly chart: AnyRes Vision Token Scaling"></div>

<p><em>Figure: AnyRes Vision Scaling — High-resolution image slicing scales visual tokens and trades off inference throughput.</em></p>

- **Sichlip Critical Batch Size**: The optimal, stable training batch size for SigLIP contrastive learning scales up to **32,768** pairs before hitting diminishing returns.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **Aspect Ratio Distortions**: Stretching raw non-square images into square $336 \times 336$ inputs destroys spatial geometry, making text lines illegible for subsequent document OCR tasks. This is resolved by AnyRes patch-cropping.

### Conceptual Reflection Questions
1. *Why does Chameleon's unified discrete-token approach suffer from training instability compared to LLaVA's continuous projection?*
   **Answer**: In Chameleon, images are discretized into codebook indices, forcing them to occupy the same vocabulary space as text. However, visual tokens have drastically higher entropy than text (since adjacent pixel values feature immense variation compared to structured language grammar). Shuffling high-entropy visual tokens and low-entropy text tokens through the same shared residual stream triggers gradient norm explosions, necessitating highly delicate QK-normalization and Z-loss regularization.

2. *Why does SigLIP's sigmoid loss remain stable at small batch sizes where CLIP's softmax loss collapses?*
   **Answer**: CLIP's softmax loss requires normalizing similarity scores across the current batch: if the batch size is small (e.g., $B = 64$), the denominator lacks negative contrastive examples, making the loss trivial to satisfy and collapsing the representation space. SigLIP treats alignment as independent binary classifications against a static bias $b$; the classification task remains mathematically well-defined regardless of batch size.
