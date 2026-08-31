# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 12: Self-Supervised Learning

#### 0. Quick-Recall Summary
*   **The SSL Paradigm:** Self-Supervised Learning (SSL) leverages unlabeled data by formulating a proxy objective called a **pretext task** (e.g., rotation, jigsaw, inpainting) to automatically generate labels from raw pixel structure, extracting robust features for **downstream tasks** via linear probing or fine-tuning.
*   **Masked Autoencoders (MAE):** An asymmetric ViT-based reconstruction framework utilizing a highly aggressive random masking ratio ($75\%$). The large encoder processes only the remaining $25\%$ unmasked patches, while a lightweight decoder uses learnable mask tokens to reconstruct the pixel values under an MSE loss computed solely on masked patches.
*   **InfoNCE & Mutual Information:** Contrastive learning optimizes the InfoNCE loss to pull positive pairs (random augmentations of the same image) closer while repelling negatives. InfoNCE mathematically lower-bounds the **mutual information** $I(X, X^+)$, where larger numbers of negative samples $K$ tighten the bound.
*   **SimCLR Nonlinear Projection Head:** SimCLR computes InfoNCE in a projected latent space $z = g(h)$ rather than the feature space $h = f(x)$. This non-linear projection prevents the loss of semantic information (e.g., color, texture details) that is irrelevant for contrastive invariance but crucial for downstream generalization.
*   **Momentum Contrast (MoCo):** MoCo decouples dictionary size from the batch size by treating negative samples as a continuous queue. To prevent representation collapse without GPU-prohibitive backpropagation, the key encoder weights $\theta_k$ are updated via an Exponential Moving Average (EMA) of the query encoder weights $\theta_q$: $\theta_k \leftarrow m \theta_k + (1-m) \theta_q$.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To introduce unsupervised representation learning paradigms that bypass the massive cost and scaling limits of manual labeling. The lecture frames the transition from early hand-designed heuristic pretext tasks to high-capacity masked reconstruction (MAE) and contrastive alignment models (SimCLR, MoCo, DINO).
*   **Lecture Category:** (d) Specific Vision Task (Self-Supervised / Unsupervised Representation Learning) blended with (c) Training and Optimization practice.
*   **Builds on:** Stacks on Lecture 5 (Data Augmentation), Lecture 6 (CNN/ViT backbones), and Lecture 8 (Transformers/ViTs), providing the pre-training engine that powers Lecture 16 (Multimodal Foundation Models like CLIP and Momo).

---

#### 2. Mathematical Foundations

##### Pretext Task Formulations
*   **Rotation Prediction Loss:**
    Given a dataset of unlabeled images, each image $X_i$ is rotated by an angle $\theta_c \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$. The network parameters $W$ are optimized via standard 4-way cross-entropy loss:
    $$L\_{rot} = -\frac{1}{N} \sum\_{i=1}^N \sum\_{c=0}^3 \mathbb{I}(y_i = c) \log P(y_i = c | X_i^{\theta_c}; W)$$
    where $\mathbb{I}(\cdot)$ is the indicator function and $P(y_i = c | X_i^{\theta_c}; W)$ represents the predicted probability of rotation class $c$.

*   **Jigsaw Puzzle Classification Loss:**
    Images are divided into a $3\times3$ grid of patches, yielding $9! = 362,880$ possible permutations. To simplify optimization, the label space is restricted to a subset of $C = 64$ highly dissimilar permutations. The model predicts the active permutation index using a 64-way softmax loss:
    $$L\_{jigsaw} = -\log \frac{\exp(s_y)}{\sum\_{j=1}^{64} \exp(s_j)}$$
    where $s_y$ is the logit score for the ground-truth permutation.

*   **Inpainting Reconstruction Loss:**
    A binary mask $M \in \{0, 1\}^{H \times W \times C}$ zeroes out parts of the input image $X$. The encoder-decoder network $f_\theta$ reconstructs the missing pixels. The reconstruction loss is formulated as a masked $L_2$ distance:
    $$L\_{rec} = \|M \odot (X - f_\theta((1 - M) \odot X))\|_2^2$$
    where $\odot$ represents the element-wise (Hadamard) product. This is often combined with an adversarial loss $L\_{adv}$ to reduce blurry outputs.

*   **Split-Brain Autoencoder:**
    An image is split into two disjoint sets of channels, e.g., lightness channel $X_1 = L$ and color channels $X_2 = (A, B)$ in $Lab$ color space. Two independent sub-networks $f_1$ and $f_2$ are trained to cross-predict each other's channels:
    $$L\_{split} = L\_{dist}(f_1(X_1), X_2) + L\_{dist}(f_2(X_2), X_1)$$
    where $L\_{dist}$ represents either an element-wise regression loss or cross-entropy over binned colors.

##### Masked Autoencoders (MAE) Formulation
*   **Asymmetric Data Flow:**
    The original image $X$ is split into non-overlapping patches $\{x_p\}\_{p=1}^P$. A random binary mask selects $75\%$ of the patches to be discarded. The remaining unmasked patches ($25\%$) are embedded and processed by a deep ViT encoder to obtain latents $Z\_{unmasked}$.
    
    Before entering the lightweight ViT decoder, the unmasked latents are aligned with learnable, shared **mask tokens** $e\_{mask} \in \mathbb{R}^D$ and restored to their original sequence positions. Position embeddings $E\_{pos}$ are added to preserve spatial coordinates:
    $$H\_{dec} = [Z\_{unmasked}; \text{placeholder}(e\_{mask})] + E\_{pos}$$
    The loss is a Mean Squared Error (MSE) computed **only** on the reconstructed pixels of the masked patches:
    $$L\_{MAE} = \frac{1}{|M\_{patches}|} \sum\_{i \in M\_{patches}} \|x_i - \hat{x}_i\|_2^2$$
    where $x_i$ and $\hat{x}_i$ are the true and reconstructed pixel values of patch $i$, and $M\_{patches}$ is the set of masked indices.

##### Contrastive Learning & InfoNCE
*   **Cosine Similarity Metric:**
    The alignment between query representation $q$ and key representation $k$ is measured using cosine similarity:
    $$\text{sim}(q, k) = \frac{q^T k}{\|q\|_2 \|k\|_2}$$

*   **InfoNCE Loss Function:**
    For a given query representation $q$, let $k^+$ be the positive key representation (e.g., from an alternate crop of the same image), and $\{k^-_i\}\_{i=1}^K$ be the set of $K$ negative key representations (from other images). The InfoNCE loss is defined as:
    $$L\_{InfoNCE} = -\log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\exp(\text{sim}(q, k^+) / \tau) + \sum\_{i=1}^{K} \exp(\text{sim}(q, k^-_i) / \tau)}$$
    where $\tau$ is a temperature hyperparameter controlling the scaling of similarity scores.

*   **Mutual Information Lower Bound:**
    Minimizing $L\_{InfoNCE}$ maximizes the mutual information $I(X, X^+)$ between similar views $X$ and $X^+$. The mathematical bound is expressed as:
    $$I(X, X^+) \geq \log(K) - L\_{InfoNCE}$$
    where $K$ is the number of negative samples. As $K \rightarrow \infty$, the bound on mutual information becomes tighter, explaining why large batch sizes or memory queues are mathematically necessary.

*   **MoCo Momentum Weight Update:**
    To maintain a continuous dictionary queue $Q$ of negative keys without backpropagating through earlier iterations, MoCo uses a **momentum key encoder** parameterized by $\theta_k$, which is updated via an exponential moving average (EMA) of the active encoder weights $\theta_q$:
    $$\theta_k \leftarrow m \theta_k + (1-m) \theta_q$$
    where $m \in [0.99, 1.0)$ is the momentum coefficient.

---

#### 3. Architecture / Algorithm Walkthrough

##### Algorithmic Logic: SimCLR vs. MoCo
1.  **SimCLR Training Flow (End-to-End, Large Batches):**
    ```
    [Input Image X] ──(Augmentation t)──> [View x_i] ──> [f_q(·)] ──> [h_i] ──> [g_q(·)] ──> [z_i] ──┐
                  └──(Augmentation t')──> [View x_j] ──> [f_q(·)] ──> [h_j] ──> [g_q(·)] ──> [z_j] ──┼─> [InfoNCE Loss]
                                                                                                 │
    [Other Images] ───────────────────────> [Negatives] ──> [f_q(·)] ──> [h_neg] ─> [g_q(·)] ─> [z_neg] ─┘
    ```
    *   SimCLR applies two random augmentations to image $X$.
    *   Both views are mapped to features $h$ and projected to $z$ using the same active encoder $f_q$ and MLP projection head $g_q$.
    *   Gradients propagate through both pathways, requiring immense batch sizes (e.g., $N = 4096$) to collect enough negative samples within the same GPU iteration.

2.  **MoCo Training Flow (Memory Queue & Momentum Key Encoder):**
    ```
    [Query Image X] ──(Aug_1)──> [x_query] ──> [f_q(·) Encoder] ──> [q] (BP active) ───────────┐
                                                                                               ├─> [InfoNCE Loss]
    [Key Image X]   ──(Aug_2)──> [x_key]   ──> [f_k(·) Momentum] ─> [k+] (EMA weights, No BP) ─┤
                                                                                               │
    [Memory Queue Q] ────────────────────────────────────────────> [k_1^-, k_2^-, ..., k_K^-] ─┘
    ```
    *   Query is encoded by active encoder $f_q$ (weights $\theta_q$, optimized via backpropagation).
    *   Positive key is encoded by the momentum key encoder $f_k$ (weights $\theta_k$, updated via EMA).
    *   Negatives are retrieved directly from a FIFO queue $Q$ containing keys from previous batches.
    *   At the end of the step, the current positive key $k^+$ is pushed into the queue, and the oldest key is dequeued.

##### PyTorch Blueprint (SimCLR with Projection Head & Custom InfoNCE)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    # Illustrative blueprint of SimCLR architecture showing the visual feature 
    # representation space 'h' and the non-linear projection space 'z'.
    def __init__(self, backbone, feature_dim=2048, projection_dim=128):
        super(SimCLR, self).__init__()
        self.backbone = backbone  # ResNet or ViT visual encoder
        
        # Non-linear projection head g(h): MLP with ReLU
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim, bias=False)
        )

    def forward(self, x):
        # f(x) -> h (retained for downstream linear probing)
        h = self.backbone(x)
        h = torch.flatten(h, start_dim=1)
        # g(h) -> z (used only during self-supervised pre-training)
        z = self.projection_head(h)
        return h, F.normalize(z, dim=1) # Normalized for cosine similarity

class InfoNCELoss(nn.Module):
    # Symmetric InfoNCE Loss over double-augmented mini-batch elements.
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # z_i, z_j: Normalized projections of two augmented views of shape (B, projection_dim)
        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0) # Shape: (2B, projection_dim)
        
        # Compute pairwise cosine similarity matrix of shape (2B, 2B)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create mask to isolate positive pairs and filter out self-similarities
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0) # Shape: (2B,)
        
        # Calculate denominator mask
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device))
        logits = similarity_matrix[mask].view(2 * batch_size, -1) / self.temperature
        
        # Align positive similarities as target labels
        pos_logits = positives.view(2 * batch_size, 1) / self.temperature
        
        # Combine logits into softmax classification problem
        logits = torch.cat([pos_logits, logits], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        return F.cross_entropy(logits, labels)
```

---

#### 4. Visual Intuition & Interpretability

##### Pretext Task Attention Maps
*   **Rotation Common Sense Map:** Under rotation pre-training, the attention maps of the encoder focus holistically on organic contours (e.g., eye sockets, feet orientation, horizons). This indicates that to predict rotation, the model has to implicitly learn a global structural coordinate system of everyday objects.
*   **Split-Brain Cross-Channel Alignment:** When colorizing gray-scale scenes, the model’s activations track pixel-level boundaries and reflective symmetries. For example, the reflection of a half-dome in a lake is colorized symmetrically with the actual sky, showing that the model is tracking visual correspondences and spatial reflections over time.

##### Contrastive Latent Space Topology
*   **Attract and Repel Forces:**
    ```
    [Augmented View A1] ───(Attract Force)───> [Augmented View A2]
            │                                         │
     (Repel Force)                            (Repel Force)
            ▼                                         ▼
    [Negative Image B1] <───(Attract Force)───> [Negative Image B2]
    ```
    *   Positive pairs (augmented views from the same image source) act under a mutual attraction force, clustering tightly in the latent space.
    *   Negative pairs (views from different sources) act under a mutual repulsive force, distributing clusters isotropically across the unit hypersphere.
*   **t-SNE Dimensionality Reduction:** Visualizing the pre-trained embedding space via t-SNE reveals that even without class labels, semantic clusters (e.g., all dogs, all vehicles, or drawings vs. photos of the same concept) naturally separate into cohesive topological neighborhoods.

---


<div id="plotly-cs231n-12-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 12 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)
To help researchers diagnose representation collapse during contrastive pre-training, we propose an interactive **Isotropic Contrastive Embedding Visualizer**:

*   **Visualization Type:** 3D Scatter Plot with interactive projection tracing.
*   **Data Fields & Encoding:**
    *   **X/Y/Z Axes:** The top 3 principal components (PCA) of the projection vectors $z$.
    *   **Color Scale:** Maps to the raw visual source identity index (each unique source image gets its own color, shared across its augmentations).
    *   **Vector Connectors:** Draws green lines (attraction force vectors) between positive pairs and red lines (repulsive forces) between negative query-key vectors.
*   **Interactive Controls:**
    *   **Epoch Slider:** Animates embedding transitions over training steps. In early epochs, points are randomly distributed. As SimCLR optimizes, positive pairs merge, and semantic clusters separate.
    *   **Temperature ($\tau$) Slider:** Dynamically adjusts the softmax scale hyperparameter in InfoNCE. Toggling $\tau \rightarrow 0$ shows gradients becoming highly sparse (focusing only on immediate hard negatives), while $\tau \rightarrow \infty$ forces uniform repulsive pressure across all negatives.
    *   **Representation Collapse Alert:** Toggles a red warning when all embeddings collapse to a single point on the hypersphere, highlighting failure modes caused by an inactive momentum update or missing batch normalization.

---

#### 6. Empirical Design Heuristics & Benchmark Results

##### Pretext Task Performance Comparison (Linear Probing on ImageNet)
*   **Rotation Heuristic:** Rotation pre-training achieves strong visual representation but is fundamentally limited by class-symmetry bias (e.g., an isotropic sphere looks identical at all rotations).
*   **Masked Autoencoder (MAE) Scaling:**
    *   **Optimal Masking Ratio:** MAE achieves its highest downstream accuracy at an extremely high masking ratio of **$75\%$**. This forces the network to learn holistic structures rather than relying on local pixel correlations.
    *   **Sampling Strategy:** Uniform random masking out-performs grid-based or block-based masking on downstream classification.

##### Contrastive Learning Milestones (ImageNet Evaluation)
*   **SimCLR Performance Bounds:**
    *   Linear probing accuracy scales monotonically with the size of the **non-linear projection head** $g(h)$ and **mini-batch size**.
    *   A projection dimension of $128$ combined with a batch size of $4096$ trained on multiple GPUs matches the performance of fully supervised ResNet-50 baselines.
*   **DINO and DINOv2 Scaling:**
    *   **DINOv1:** Trained on the $1\text{M}$ image subset of ImageNet.
    *   **DINOv2:** Scaled self-supervised learning to a larger uncurated training set of **$142\text{ million}$ images**, yielding high-fidelity, general-purpose feature representations that act as visual foundations without any supervised pre-training.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Failure Modes
*   **The Projection Head Downstream Trap:** During downstream linear probing, **always discard the projection head** $g(h)$ and evaluate on the feature representation $h = f(x)$. Because $g(h)$ is explicitly trained to be invariant to data augmentations, it discards color, pose, and texture details that are crucial for downstream classification.
*   **SimCLR Batch Size Sensitivity:** SimCLR is highly unstable at standard batch sizes (e.g., $64$ or $128$). Without a large pool of negative samples, the contrastive task becomes trivially easy, preventing the network from learning high-frequency discriminative features.
*   **MoCo Momentum Calibration:** Setting the MoCo momentum coefficient $m$ too low (e.g., $m = 0.9$) causes representation collapse. This is because the key encoder weights update too rapidly, causing the key representations in the queue to drift out of sync with the query encoder, leading to divergent gradients.

##### Graduate-Level Reflection Questions
1.  **Semantic vs. Instance Invariance in InfoNCE:** InfoNCE forces the encoder to be invariant to random augmentations. However, if our augmentations include aggressive color jitter and cropping, how does this affect the model's performance on downstream tasks that rely heavily on absolute scale and color consistency (e.g., semantic segmentation of tumors)?
2.  **MAE High Masking Ratio Mechanics:** Why does Masked Autoencoding require an extremely high masking ratio ($75\text{--}80\%$) to learn robust representations, whereas language models (BERT) are pre-trained with only $15\%$ masking? Contrast the spatial redundancy of visual pixels with the high semantic density of discrete text tokens.
3.  **The Information Bottleneck of SimCLR:** Prove why utilizing a non-linear projection head $g(h) = W^{(2)}\text{ReLU}(W^{(1)}h)$ rather than a linear projection head $g(h) = Wh$ preserves more visual information in $h$. Detail this under the lens of the data processing inequality and representation distillation.
