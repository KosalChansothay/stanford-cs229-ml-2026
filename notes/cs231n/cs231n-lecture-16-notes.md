# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 16: Vision and Language

#### 0. Quick-Recall Summary
*   **Contrastive Alignment Paradigm:** CLIP (OpenAI, 2021) revolutionized multimodal vision by training a symmetric InfoNCE loss over 400M internet-scraped image-text pairs. It bypassed closed-vocabulary limitations, enabling zero-shot classifier generation out-of-the-box.
*   **Feature Grounding Axiom:** Multimodal language models (like LLaVA) extract spatial tokens from the *penultimate* layer of a frozen vision transformer (ViT) rather than the final `[CLS]` token. This preserves dense localized structural coordinate information.
*   **Gated Cross-Attention Routing:** Flamingo (DeepMind, 2022) keeps both visual encoders and LLMs frozen, injecting trainable *Gated Cross-Attention-Dense* layers at every decoder stage. It uses a $\tanh$ gating mechanism to selectively regulate visual information injection.
*   **Segmentation Foundation Paradigm:** Segment Anything Model (SAM) resolves spatial prompt ambiguity by outputting three multi-granular masks simultaneously (part, sub-part, whole), avoiding mathematical loss penalty contradictions during training.
*   **Visual Program Chaining:** VisProg (Visual Programming, 2024) addresses complex, multi-step compositional reasoning by using an LLM to generate executable Python programs that chain specialized, modular computer vision subroutines.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master the design and execution of multimodal foundation models that bridge the semantic and structural representations of computer vision and natural language processing. The lecture details the transition from single-task closed-vocabulary visual classification to unified open-vocabulary grounding, interactive multi-turn dialogue, promptable spatial segmentation, and visual program chaining.
*   **Lecture Category:** (b) Architecture Design and (d) Specific Vision Task (Multimodal Vision-Language Alignment).
*   **Builds on:** Lecture 8 (Transformers & Attention) and Lecture 12 (Self-Supervised Learning, specifically SimCLR and InfoNCE loss).

---

#### 2. Mathematical Foundations

##### Symmetric Multimodal InfoNCE Loss (CLIP)
CLIP trains an image encoder $f_I$ and a text encoder $f_T$. Given a mini-batch of $N$ paired images and text descriptions, let $x_i$ represent the $i$-th image and $y_i$ represent its matching text caption. The normalized projection embeddings are:
$$v_i = \frac{f_I(x_i)}{\|f_I(x_i)\|_2}, \quad u_i = \frac{f_T(y_i)}{\|f_T(y_i)\|_2}$$
The scaled cosine similarity matrix elements $s\_{i, j}$ are defined as:
$$s\_{i, j} = \frac{v_i^T u_j}{\tau}$$
where $\tau$ is a learnable temperature parameter. 

CLIP minimizes the sum of two complementary cross-entropy losses (Image-to-Text and Text-to-Image):
$$L\_{I2T} = -\frac{1}{N} \sum\_{i=1}^N \log \frac{\exp(s\_{i,i})}{\sum\_{j=1}^N \exp(s\_{i,j})}$$
$$L\_{T2I} = -\frac{1}{N} \sum\_{i=1}^N \log \frac{\exp(s\_{i,i})}{\sum\_{j=1}^N \exp(s\_{j,i})}$$
$$L\_{CLIP} = \frac{1}{2} (L\_{I2T} + L\_{T2I})$$

##### Gated Cross-Attention Gating (Flamingo)
Flamingo integrates visual features $h\_{vis}$ into a frozen language model's hidden states $x$ at layer $l$ using a gated cross-attention mechanism. The gate uses a learnable scalar parameter $\alpha_l$, initialized to $0$:
$$x\_{atten} = \text{CrossAttention}(x, h\_{vis})$$
$$x\_{gated} = x + \tanh(\beta_l) \cdot \text{FFN}(x + \tanh(\alpha_l) \cdot x\_{atten})$$
Initializing $\alpha_l = 0$ and $\beta_l = 0$ guarantees that the original behavior of the pre-trained, frozen language model is strictly preserved at the onset of training.

##### SAM Ambiguity Resolution Loss
To solve point-prompt semantic ambiguity, SAM outputs $K=3$ masks (Part, Sub-part, Whole). Let $M_k$ be the predicted mask logits for granularity $k \in \{1, 2, 3\}$, and let $Y$ be the ground-truth mask. The loss is computed only on the mask $k^*$ that achieves the minimum cross-entropy with the ground-truth:
$$k^* = \arg\min\_{k} \mathcal{H}(M_k, Y)$$
$$L\_{SAM} = \mathcal{H}(M\_{k^*}, Y)$$
This prevents conflicting gradients from penalizing valid visual interpretations.

---

#### 3. Architecture / Algorithm Walkthrough

##### Data-Flow Routing & Alignment
```
1. CLIP Contrastive Pre-training:
   Image Batch ──> [Image Encoder (ViT/ResNet)] ──> Project Head ──┐
                                                                   ├──> Symmetric InfoNCE Loss
   Text Batch  ──> [ Text Encoder (Transformer) ] ──> Project Head ──┘

2. LLaVA Dense Visual Grounding:
   Image ──> [Pre-trained ViT] ──> [Penultimate Layer Tokens] ──> [Linear Projection (W)] ──┐
                                                                                          ├──> [Language Model] ──> Autocomplete text
   Text  ───────────────────────────────────────────────────────> [Text Embeddings (Tokens)]  ┘

3. Flamingo Gated Cross-Attention Blocks:
   Text Tokens ──> [Frozen FFN Layer] ──> [Cross-Attention] ──> [tanh Gate] ──> [Layer Norm] ──> Next Layer
                                                ▲
   Image       ──> [Frozen ViT] ──> [Perceiver Sampler] (Downsampled Visual Tokens)
```

##### PyTorch Blueprint (Illustrative Multimodal Alignment)
This minimal PyTorch implementation outlines a unified pipeline featuring a CLIP-style contrastive loss, a frozen LLaVA-style projector, and Flamingo's $\tanh$-gated cross-attention blocks:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricCLIPLoss(nn.Module):
    def __init__(self, init_tau=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_tau))

    def forward(self, image_embeds, text_embeds):
        # Normalize embeddings to the unit hypersphere
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # Compute scaled similarity matrix
        sim_matrix = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        
        # Ground-truth indices represent diagonal matching
        labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
        
        # Compute symmetric cross-entropy
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_t = F.cross_entropy(sim_matrix.t(), labels)
        return (loss_i + loss_t) / 2.0

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))  # Initialized to 0 to preserve LLM weights
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.ffn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, x_vis):
        # x: [Batch, Seq_Len, Embed_Dim]
        # x_vis: [Batch, Fixed_Num_Tokens, Embed_Dim] (Output of Perceiver Sampler)
        attn_out, _ = self.cross_attn(query=x, key=x_vis, value=x_vis)
        x = x + torch.tanh(self.gate) * attn_out
        
        ffn_out = self.ffn(x)
        x = x + torch.tanh(self.ffn_gate) * ffn_out
        return x
```

---

#### 4. Visual Intuition & Interpretability

##### CLIP Alignment Interpretability
*   **The Nearest-Neighbor Heuristic:** CLIP maps classification to a 1-nearest neighbor text-retrieval task on the unit hypersphere. Instead of learning categorical boundaries, it embeds text queries (e.g., "a photo of a plane") and relies on localized cosine proximity.
*   **Zero-Shot Robustness:** CLIP generalizes to out-of-domain natural distributions (such as ObjectNet's rotten or physically rotated bananas) much more robustly than supervised ResNets. This robustness stems from linguistic context (e.g., textures, colors, background shapes) present in text captions rather than flat categorical labels.

##### Grounded Visual Counting (Momo)
*   Momo tracks object categories by explicitly outputting and attending to specific, localized pixel coordinates. For example, when asked to count boats, the model does not hallucinate numbers based on background statistics; instead, it generates coordinates that point directly to each boat and counts the resulting coordinates.

##### SAM Multi-Masking
*   When prompted with a single point on a pair of scissors, SAM generates three distinct, overlapping masks. These correspond to:
    1.  *Whole:* The entire scissor assembly.
    2.  *Part:* The finger loops.
    3.  *Sub-part:* The single metal ring.
*   This visual multi-granularity prevents optimization conflicts during backpropagation.

---


<div id="plotly-cs231n-16-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 16 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)

We propose an interactive **Multimodal Latent Space Hypersphere Visualizer** to study representation alignment and compositionality failures:

*   **Visualization Type:** Interactive 3D Spherical Plot (representing projections on the unit hypersphere).
*   **Data Fields & Encoding:**
    *   **Points:** Text queries (represented by colored triangles) and corresponding image frames (represented by colored circles).
    *   **Color Scale:** Categorical classes (e.g., "Mug in grass" vs. "Grass in mug").
    *   **Vector Links:** Proximity lines displaying cosine distance.
*   **Interactive Controls:**
    *   **Batch Size Slider (2 to 32,768):** Toggles how dense the negative pairs are in the embedding space.
    *   **Compositionality Toggles:** Swaps subject-object orders (e.g., "Mug in grass" $\leftrightarrow$ "Grass in mug"). When activated, it displays how CLIP's embeddings fail to shift significantly under subject-object reversals, illustrating the loss of structural compositionality.

---

#### 6. Empirical Design Heuristics & Benchmark Results

##### Pre-training and Zero-Shot Classification
*   **CLIP Scale Milestones:** Pre-trained on **400 million** image-text pairs using a Vision Transformer (ViT) with **307 million** parameters.
*   **Phrase Prompt Engineering:** Substituting a single class name (e.g., "dog") with a templated phrase (e.g., "a photo of a dog") yields an automatic **1.3% top-1 accuracy boost** on ImageNet. Selecting and averaging multiple templates (e.g., "a photo of a [class]", "a drawing of a [class]") further improves generalization.
*   **KOKA Paradigm Shift:** Adding an image captioning decoder (with cross-attention) alongside the contrastive loss in KOKA yields a **~10% top-1 boost** across ImageNet out-of-distribution benchmarks compared to standard CLIP.

##### Large Multimodal Language Models (LMLMs)
*   **Momo Grounding Efficiency:** While Meta's Llama models rely on **6 billion** uncurated internet pairs, Momo matches GPT and Gemini benchmarks using only **700,000 highly curated, grounded samples**. 
*   **Talking vs. Typing:** Forcing human annotators to *talk* instead of *type* descriptions during training data collection systematically breaks "Gricean maxims of quality/quantity". This yields much denser descriptions of spatial relationships and material details that are typically omitted from written text.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

*   **The Penultimate Layer Trap:** Attempting to extract visual tokens from the final layer of a CLIP encoder for spatial grounding tasks is a silent failure mode. The final layer's tokens discard localized features to satisfy the global image-level contrastive loss. Using the penultimate layer's feature maps preserves spatial coordinate alignment.
*   **Incidental Text Bias:** Scraping uncurated web captions produces "incidental text," where sentences describe subjective emotions rather than physical pixel scenes. Models trained on this data struggle with spatial relationships (e.g., "to the left of").
*   **Hard Negatives Instability:** Manually curating hard negative pairs in CLIP's batch forces the model to learn fine-grained details but causes it to "unlearn" basic visual semantic alignments, degrading overall zero-shot out-of-distribution performance.

##### Graduate-Level Reflection Questions
1.  **Symmetric vs. Asymmetric Information Bottlenecks:** Explain why CLIP's dual symmetric InfoNCE loss yields visual representations with low spatial compositionality. How does KOKA's cross-attention captioning decoder mathematically alleviate this bottleneck?
2.  **The Gating Initialization Limit:** Why does Flamingo initialize the gating parameters ($\alpha_l, \beta_l$) inside its custom cross-attention layers to $0$ rather than using standard random normal initializations? Detail what happens to gradients inside a 10B+ LLM if these gate parameters are initialized randomly.
3.  **Ambiguity in Differentiable Segmentation:** Analyze how SAM's multi-mask output prediction strategy resolves prompt ambiguity mathematically. Why does training a network to predict only one mask for a single point query fail when backpropagating through natural scenes?
