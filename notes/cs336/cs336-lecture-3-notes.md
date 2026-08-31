# CS336 Lecture 3: Architectures

## 0. Quick-Recall Summary
* **Pre-norm is Standard:** All modern dense language models place the normalization layer outside the residual stream (before computation) to ensure a clean gradient highway ($x \to x + \text{Layer}(x)$) and avoid optimization instability.
* **RMSNorm replaces LayerNorm:** Eliminating the mean subtraction and bias terms from standard LayerNorm reduces memory bandwidth traffic, translating to up to 25% runtime speedups despite accounting for <0.2% of mathematical FLOPs.
* **SwiGLU is Dominant:** Replacing vanilla MLPs with Gated Linear Units (GLUs) using the Swish/SiLU activation provides consistent empirical perplexity benefits; the hidden dimension must be scaled by $2/3$ to maintain parameter parity.
* **RoPE Geometry:** Rotary Position Embeddings (RoPE) enforce relative position by splitting the $d$-dimensional query/key vectors into 2D chunks and rotating them by position-dependent angles, ensuring absolute position invariance in inner products.
* **GQA Sweeps Inference:** Grouped-Query Attention (GQA) groups query heads to share key-value heads, serving as an optimal middle-ground between standard Multi-Head Attention and Multi-Query Attention to maximize serving throughput with negligible loss in expressiveness.

---

## 1. Core Paradigm & Systems Overview
* **Objective:** Understand how architectural choices are co-designed around three tightly coupled axes: expressive representation power, training stability at extreme scales, and hardware execution efficiency on modern accelerators.
* **Primary Bottleneck:** While pre-training is traditionally **compute-bound** due to large-matrix tensor contractions (GEMMs), modern deployment and long-context scaling are severely **memory-bandwidth bound**. This bottleneck shifts focus toward minimizing memory accesses (e.g., RMSNorm, GQA, sliding window attention) over pure FLOP reduction.
* **Builds on:** This lecture builds directly on Lecture 2's metrics of **arithmetic intensity** and **resource accounting**. We apply those principles to analyze why tiny, low-FLOP normalization and activation layers consume substantial GPU runtime, and how to optimize them to keep the tensor cores hot.

---

## 2. Theoretical & Mathematical Primitives

### Normalization Operations

#### Standard LayerNorm

$$ y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta $$

where $\mu = \frac{1}{d}\sum\_{i=1}^d x_i$, $\sigma^2 = \frac{1}{d}\sum\_{i=1}^d (x_i - \mu)^2$, and $\gamma, \beta \in \mathbb{R}^d$ are learnable scale and shift parameters.

#### RMSNorm (Root Mean Square Normalization)

$$ \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma = \frac{x}{\sqrt{\frac{1}{d}\sum\_{i=1}^d x_i^2 + \epsilon}} \odot \gamma $$

where the mean subtraction and bias parameters $\beta$ are omitted, reducing memory traffic without hurting representation capacity.

---

### Non-Linearities & Gated Linear Units (GLUs)

#### SwiGLU Activation

$$ \text{SwiGLU}(x) = \text{Swish}(x W_1) \odot (x V) $$

where $\text{Swish}(x) = x \cdot \text{sigmoid}(\beta x)$ (usually $\beta=1$), and $W_1, V \in \mathbb{R}^{d \times d\_{ff}}$. To maintain parameter parity with standard MLPs (which contain two matrices $W_1, W_2$), the intermediate dimension $d\_{ff}$ is scaled down by a factor of $2/3$:

$$ d\_{ff} \approx \frac{2}{3} \times 4 d = \frac{8}{3} d $$

---

### Positional Encodings: Rotary Position Embeddings (RoPE)

We seek an embedding function $f_q(x_i, i)$ and $f_k(x_j, j)$ that preserves relative distance in their inner product:

$$ \langle f_q(x_i, i), f_k(x_j, j) \rangle = g(x_i, x_j, i - j) $$

For a 2D vector $x = (x_1, x_2)^T$ at position $m$, this is accomplished via a rotation matrix:

$$ f(x, m) = R\_{\theta, m} x = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} $$

For a $d$-dimensional vector, RoPE partitions the vector into $d/2$ independent 2D subspaces and rotates each subspace $k \in [1, d/2]$ by frequency $\theta_k = 10000^{-2(k-1)/d}$:

$$ R^d\_{\Theta, m} = \text{diag}\left(R\_{\theta_1, m}, R\_{\theta_2, m}, \dots, R\_{\theta\_{d/2}, m}\right) $$

---

### Scale Optimization & Training Stability

#### Log-Z Loss (Z-loss)
To prevent numerical overflow/underflow in the output softmax normalizer $Z = \sum\_{j} e^{u_j}$, a squaring penalty is added to the training objective:

$$ \mathcal{L}\_{\text{total}} = \mathcal{L}\_{\text{cross-entropy}} + \alpha \log^2 Z $$

This forces the partition function close to 1 ($\log Z \approx 0$), stabilizing pre-training at scale.

#### QK Normalization
To prevent attention logit explosion (which triggers training spikes), queries ($Q$) and keys ($K$) are normalized using RMSNorm or LayerNorm right before the dot-product attention calculation:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\text{Norm}(Q) \text{Norm}(K)^T}{\sqrt{d\_{head}}}\right) V $$

#### Soft Capping
An alternative bounding mechanism that passes attention logits through a scaled hyperbolic tangent function:

$$ \text{capped\_logits} = C \cdot \tanh\left(\frac{\text{logits}}{C}\right) $$

where $C$ is a hard threshold hyperparameter (commonly $10$ or $30$ or $50$).

---

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### Algorithmic Logic: GQA and RoPE

1. **RoPE Rotation Processing:** 
   * Given incoming Query and Key tensors of shape `(B, S, H, D)`.
   * Reshape/split the head dimension $D$ into pairs of size 2.
   * Generate sine and cosine frequencies for position indices up to sequence length $S$.
   * Perform rotation: $x\_{\text{rot}} = [x_1 \cos(m\theta) - x_2 \sin(m\theta), x_1 \sin(m\theta) + x_2 \cos(m\theta)]$.
2. **Grouped Query Attention (GQA) Head Expansion:**
   * Keys and Values are shaped as `(B, S, H_kv, D)`. Queries are shaped as `(B, S, H_q, D)`.
   * For computation, keys and values must be repeated or expanded to match the query head count $H_q$.
   * Let the group size $G = H_q / H_kv$. Each KV head is duplicated $G$ times using `einops` patterns to align with queries before batched matrix multiplication.

### PyTorch/Pythonic Blueprint

```python
import torch
import torch.nn as nn
from einops import rearrange, repeat

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Omits mean subtraction and bias shift for memory bandwidth efficiency.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate root mean square along the last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLUMLP(nn.Module):
    """
    Gated Linear Unit (GLU) with Swish/SiLU activation.
    Applies 2/3 scaling to hidden dimension to maintain parameter parity.
    """
    def __init__(self, d_model: int, d_ff_multiplier: float = 4.0):
        super().__init__()
        # Apply 2/3 adjustment to keep parameter budget equal to a standard MLP
        d_ff = int(2 / 3 * d_model * d_ff_multiplier)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element-wise gated multiplication
        return self.w2(self.act(self.w1(x)) * self.v(x))


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) implemented using 2D chunk rotations.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        # dim must be even
        assert dim % 2 == 0
        self.dim = dim
        
        # Calculate frequencies: theta_k = theta^(-2(k-1)/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin frequencies up to max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqss = torch.outer(t, self.inv_freq)
        # Create a representation for pairs: [cos, cos, sin, sin] pattern
        emb = torch.cat((freqss, freqss), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # Split vector and swap/negate for 2D rotation math
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x shape: (B, S, H, D)
        cos = self.cos_cached[:seq_len, None, :] # Shape: (S, 1, D)
        sin = self.sin_cached[:seq_len, None, :] # Shape: (S, 1, D)
        
        # Apply relative 2D rotation
        return (x * cos) + (self._rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) utilizing einops for clean head replication.
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group_size = n_heads // n_kv_heads
        
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        
        # Project and reshape to (B, S, H, D)
        q = rearrange(self.q_proj(x), "b s (h d) -> b s h d", d=self.d_head)
        k = rearrange(self.k_proj(x), "b s (h d) -> b s h d", d=self.d_head)
        v = rearrange(self.v_proj(x), "b s (h d) -> b s h d", d=self.d_head)
        
        # Replicate KV heads across groups using einops repeat
        # We repeat each of the n_kv_heads by group_size times
        k_expanded = repeat(k, "b s h_kv d -> b s (h_kv g) d", g=self.group_size)
        v_expanded = repeat(v, "b s h_kv d -> b s (h_kv g) d", g=self.group_size)
        
        # Standard Scaled Dot-Product Attention
        scores = torch.einsum("b s h d, b t h d -> b h s t", q, k_expanded) / (self.d_head ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.einsum("b h s t, b t h d -> b s h d", attn, v_expanded)
        out = rearrange(out, "b s h d -> b s (h d)")
        return self.out_proj(out)
```

---

## 4. Hardware Realities & Compute/Memory Accounting

### Memory vs. FLOP Balance of Normalization
* Standard LayerNorm or RMSNorm represents a tiny fraction of the mathematical calculations in a forward pass (~0.17% of total FLOPs).
* However, because these operations are **element-wise**, their arithmetic intensity is extremely low:
  $$ \text{Arithmetic Intensity}\_{\text{Norm}} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} \approx \frac{O(N)}{O(N)} = \text{Constant} $$
* This makes normalization completely **memory-bandwidth bound**. Accelerators sit idle waiting to load activations from High Bandwidth Memory (HBM) to local registers, process them, and write them back.
* On small or shallow networks, LayerNorm can account for up to **25% of actual runtime**. This reality drove the adoption of RMSNorm to shave off mean calculation memory traffic.

### Inference Memory Constraints: The KV Cache Bottleneck
* At generation/inference time, generation occurs auto-regressively, token by token.
* Generating a new token requires loading all weights plus retrieving historical keys and values (the KV cache).
* This turns decoding into a strictly **memory-bandwidth bound** operation. 
* For standard Multi-Head Attention (MHA), memory traffic scales with sequence length $S$ and head count $H$.
* **Multi-Query Attention (MQA)** reduces KV heads to 1, shrinking the KV cache memory footprint drastically but causing representation bottlenecks.
* **Grouped-Query Attention (GQA)** optimizes this: by setting $H\_{kv} \in (1, H_q)$, it matches near-peak multi-head accuracy while retaining multi-query inference throughput speedups.

---

## 5. Visualization Blueprint (GQA KV Cache Compression)

<div id="plotly-cs336-3-gqa-compression" class="plotly-chart" aria-label="Interactive Plotly chart: Grouped-Query Attention KV Cache Compression"></div>

<p><em>Figure: Grouped-Query Attention (GQA) cuts KV cache memory by 8x compared to MHA on a 70B model with negligible loss in quality.</em></p>

<div id="plotly-cs336-3-rope-invariance" class="plotly-chart" aria-label="Interactive Plotly chart: RoPE Relative Distance Attention Score"></div>

<p><em>Figure: RoPE Inner Product Invariance — 2D vector rotation ensures relative token distance decay $\langle R_m q, R_n k \rangle = q^T R\_{n-m} k$.</em></p>


* **Visualization Type:** Dimensional Grid Matrix with Head-Mapping lines.
* **Data Fields & Encoding:**
  * **Y-Axis:** Sequence tokens (Context history).
  * **X-Axis:** Head dimension.
  * **Color Map:** Blue cells indicate active Query Heads ($H_q = 8$). Orange cells represent Key-Value Heads ($H\_{kv} = 2$).
  * **Mappings:** Thin connection lines map groups of Query Heads to their shared Key-Value Head (Group size $G = 4$).
* **Interactive Controls:**
  * **Group-Size Slider ($G$):** Toggles between Multi-Head Attention ($G=1$, $H\_{kv} = H_q$), Grouped-Query Attention ($G=4$, $H\_{kv}=2$), and Multi-Query Attention ($G=8$, $H\_{kv}=1$). Shows how the active memory allocation footprint shrinks dynamically as group size increases.

---

## 6. Empirical Scaling Laws & Hyperparameter Heuristics

While basic scaling behaves gracefully within large forgiving basins, industrial architectures have converged on a set of standardized ratios:

| Hyperparameter | Empirical Rule of Thumb | Standard Values | Notable Exceptions / Bounds |
|:--- |:--- |:--- |:--- |
| **Feedforward Expansion Ratio** | $4 \times d\_{model}$ | $4.0$ (Vanilla MLP) <br> $2.67$ (GLUs / SwiGLU) | Llama 2 uses $3.5$ <br> T5 v1 utilized a bold $64.0$ to optimize GPU utilization, but v1.1 reverted to standard $2.5$. |
| **Aspect Ratio** | $d\_{model} / N\_{layers} \approx 100$ | $\sim 100$ (GPT-3, Llama) | Extreme depth introduces pipeline-parallel complexity; width is much easier to parallelize via tensor-slicing (tensor parallel). |
| **Vocabulary Size** | Scales with multilinguality & model size | $32,000$ (Monolingual English) <br> $100,000 \to 256,000$ (Modern multilingual) | Large vocabulary sizes are highly expressive but consume substantial embedding parameter budget in smaller models. |
| **Regularization** | Weight decay is universally retained. Dropout is largely omitted. | Weight Decay: $0.1$ <br> Dropout: $0.0$ | Weight decay acts as an **optimization catalyst** rather than an overfitting regularizer in single-pass pre-training regimes, interacting favorably with learning rate decay to reach lower minima. |

---

## 7. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas/Common Bugs
* **The "Clean" Residual Stream Principle:** Avoid adding operations (like layer norms) directly inside the residual stream path $x \to x + \text{Layer}(x)$. Keep the highway clear so gradients propagate straight to the shallowest layers.
* **Warm-up Necessity in Post-Norm:** If forced to implement Post-Norm (e.g., matching a legacy Vaswani codebase), a prolonged warm-up learning rate schedule is absolutely mandatory to prevent immediate divergence. Pre-norm allows stable convergence with minimal or zero warm-up.
* **Soft Capping Loss Degradation:** Be warned that applying too tight of a logit soft cap ($C$) can restrict the model's ability to express high-confidence predictions, leading to slight perplexity hits compared to QK-normalization.

### Conceptual Questions

**Q1: Why does a 0.17% FLOP component like RMSNorm sometimes require optimization or custom CUDA kernels, while dense matrix multiplications (GEMMs) representing 99% of FLOPs can be run out-of-the-box using vanilla PyTorch?**
* **Answer:** PyTorch's native out-of-the-box GEMMs are dispatched directly to highly optimized libraries (like cuBLAS or CUTLASS) which execute at near-peak hardware efficiency. RMSNorm is an element-wise reduction operation. If implemented naively as multiple separate PyTorch operators, it launches several independent GPU kernels. Since normalization is memory-bandwidth bound, each launch incurs latency and requires round-trip reads/writes of activations to slow global HBM memory. A custom fused CUDA/Triton kernel reads the activations into fast local SRAM once, computes the variance, performs normalization, and writes the output back in a single round-trip, bypassing the bandwidth bottleneck.

**Q2: Analyze the architectural and parallelization trade-offs of choosing a very deep model (e.g., $N\_{\text{layers}} = 160, d\_{\text{model}} = 4096$) versus a very wide model (e.g., $N\_{\text{layers}} = 40, d\_{\text{model}} = 8192$) of roughly equivalent parameter count.**
* **Answer:** 
  1. *Parallelization/Systems perspective:* The wide model is significantly easier to scale across multi-node systems. Slicing a wide model's projection matrices across GPUs (Tensor Parallelism) is highly efficient and incurs low synchronization latency. The deep model requires slicing across layers (Pipeline Parallelism), which introduces idle bubbles in the computation pipeline and requires complex orchestration.
  2. *Expressive perspective:* Deep models generally possess greater expressive depth (ability to execute more sequential steps of computation). However, empirical sweeps show that depth-to-width ratios exhibit a flat, forgiving basin around $d\_{\text{model}}/N\_{\text{layers}} \approx 100$, indicating that systems ease of parallelization (favoring wider profiles) is often the dominant pragmatic choice.

**Q3: Explain how the Z-loss auxiliary training loss stabilizes output softmax calculations without altering the representational output of the network.**
* **Answer:** Softmax is mathematically translation-invariant: $\text{softmax}(u_j) = \text{softmax}(u_j + C)$. This overparameterization allows the log logits $u_j$ and the corresponding partition function $Z = \sum e^{u_j}$ to drift towards extremely large values during training, causing floating-point overflow/underflow when exponentiating. Adding $\alpha \log^2 Z$ to the cross-entropy loss acts as a soft constraint that forces $Z \approx 1$ (or $\log Z \approx 0$). This grounds the logits $u_j$ in a numerically safe dynamic range without affecting the final probability distribution, preventing catastrophic pre-training loss spikes.
