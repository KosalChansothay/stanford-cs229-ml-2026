# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 8: Attention and Transformers

#### 0. Quick-Recall Summary
*   **The Communication Bottleneck:** Traditional seq2seq RNNs force the entire input sequence representation into a single, fixed-length context vector $C$ (the last hidden state of the encoder), creating an information bottleneck as sequence length $N$ scales.
*   **Dynamic Context Vectors:** Attention bypasses this bottleneck by enabling the decoder to dynamically "look back" at the entire sequence of encoder hidden states on every output tick, computing a custom context vector $c_t$ as a differentiable weighted sum of encoder hidden states.
*   **Scaled Dot-Product Attention:** To generalize attention as a standalone layer, similarity is computed via dot-product $Q K^T$. To prevent vanishing gradients in high dimensions ($d_k$), the dot product is scaled down by $\sqrt{d_k}$ before applying softmax.
*   **Permutation Equivariance:** Self-attention naturally operates on unordered *sets* of vectors rather than sequences. Shuffling input vectors shuffles output vectors identically. Positional embeddings must be explicitly added to inject spatial or sequential order.
*   **The Transformer Paradigm:** Stitching multi-headed self-attention (where vectors compare themselves with all other vectors globally) and position-wise Feed-Forward Networks (where vectors are processed independently) using residual connections and layer normalization creates the state-of-the-art scaling engine of modern AI.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To trace the historical and theoretical transition from sequential sequence modeling (RNNs/LSTMs) to parallel, attention-driven models (Transformers). It isolates "attention" as a standalone, differentiable database-query operator and details how the Transformer block balances global spatiotemporal information mixing with localized, independent projection.
*   **Lecture Category:** (b) Architecture Design and (a) Mathematical Foundations.
*   **Builds on:** Lecture 7 (Recurrent Neural Networks), directly solving the sequential computation trap and the memory-decay constraints of vanilla RNNs and LSTMs.

---

#### 2. Mathematical Foundations

##### RNN Seq2Seq and the Bottleneck
In a standard Recurrent Neural Network (RNN) sequence-to-sequence (seq2seq) architecture, the encoder processes an input sequence of length $T_x$:
$$h_t = f\_{\text{RNN}}(x_t, h\_{t-1}) \quad \text{for } t = 1, \dots, T_x$$
The entire input sequence is compressed into a fixed-length context vector $C$:
$$C = h\_{T_x}$$
The decoder hidden states $s_t$ are then recursively updated utilizing $C$:
$$s_t = f\_{\text{dec}}(y\_{t-1}, s\_{t-1}, C)$$
This formulation forces $C$ to act as a lossy compression bottleneck for long sequences.

##### Attention-Based Context Vectors
To remove this bottleneck, the attention mechanism computes a dynamic context vector $c_t$ for each decoder step $t$:
1.  **Alignment Scores ($e\_{ti}$):** Measures the similarity between the previous decoder state $s\_{t-1}$ and each encoder state $h_i$:
    $$e\_{ti} = f\_{\text{att}}(s\_{t-1}, h_i) = w_a^T \tanh(W_a [s\_{t-1} ; h_i])$$
    where $W_a$ and $w_a$ are learnable linear projection parameters.
2.  **Attention Weights ($a\_{ti}$):** Normalized via softmax to represent a discrete probability distribution over input tokens:
    $$a\_{ti} = \frac{\exp(e\_{ti})}{\sum\_{j=1}^{T_x} \exp(e\_{tj})}$$
3.  **Dynamic Context Vector ($c_t$):** Computed as a weighted sum of encoder hidden states:
    $$c_t = \sum\_{i=1}^{T_x} a\_{ti} h_i$$
This $c_t$ replaces the static context vector $C$ in the decoder update:
$$s_t = f\_{\text{dec}}(y\_{t-1}, s\_{t-1}, c_t)$$

##### Scaled Dot-Product Attention
To divorce attention from RNNs and formulate it as a highly optimized, batched tensor operation, the similarity scoring is simplified to a dot product. 

Let $Q \in \mathbb{R}^{N_q \times d_k}$ be the Query matrix, $K \in \mathbb{R}^{N_k \times d_k}$ be the Key matrix, and $V \in \mathbb{R}^{N_k \times d_v}$ be the Value matrix.
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

*   **Derivation of the Scaling Factor ($\sqrt{d_k}$):** 
    Assume components of query vector $q \in \mathbb{R}^{d_k}$ and key vector $k \in \mathbb{R}^{d_k}$ are independent random variables with mean 0 and variance 1. 
    The dot product is $q \cdot k = \sum\_{i=1}^{d_k} q_i k_i$. 
    The mean of the dot product is $\mathbb{E}[q \cdot k] = 0$.
    The variance of the dot product is:
    $$\text{Var}(q \cdot k) = \sum\_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k \left( \text{Var}(q_i) \text{Var}(k_i) + \mathbb{E}[q_i]^2 \text{Var}(k_i) + \mathbb{E}[k_i]^2 \text{Var}(q_i) \right) = d_k (1 \cdot 1 + 0 + 0) = d_k$$
    As the dimensionality $d_k$ scales to large values, the variance of the dot products grows to $d_k$. This drives the softmax function into regions of extremely small gradients (vanishing gradients). Dividing by $\sqrt{d_k}$ scales the variance of the inputs back to 1, ensuring stable gradient flow during backpropagation.

##### Self-Attention Layer
For a sequence of input vectors $X \in \mathbb{R}^{N \times d\_{\text{in}}}$, we project $X$ into Queries ($Q$), Keys ($K$), and Values ($V$) using learnable projection matrices $W_Q \in \mathbb{R}^{d\_{\text{in}} \times d_k}$, $W_K \in \mathbb{R}^{d\_{\text{in}} \times d_k}$, and $W_V \in \mathbb{R}^{d\_{\text{in}} \times d_v}$:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
$$Y = \text{softmax}\left( \frac{(X W_Q) (X W_K)^T}{\sqrt{d_k}} \right) (X W_V)$$

##### Masked Self-Attention
For auto-regressive decoding (e.g., causal language models), a token at step $t$ must not attend to future tokens $t' > t$. This is enforced by applying a mask matrix $M \in \mathbb{R}^{N \times N}$ to the alignment score matrix $E = \frac{Q K^T}{\sqrt{d_k}}$ before running softmax:
$$M\_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
$$\text{MaskedAttention}(Q, K, V) = \text{softmax}(E + M) V$$
Since $\exp(-\infty) = 0$, future positions receive exactly zero attention weight, preventing data leakage.

##### Multi-Head Self-Attention (MHA)
Rather than performing a single attention function over $d$-dimensional queries, keys, and values, MHA projects them $H$ times with different, learned linear projections to $d_k, d_k, d_v$ dimensions respectively:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O$$
$$\text{where } \text{head}_h = \text{Attention}(Q W_Q^{(h)}, K W_K^{(h)}, V W_V^{(h)})$$
where the projections are parameter matrices $W_Q^{(h)} \in \mathbb{R}^{d\_{\text{model}} \times d_k}$, $W_K^{(h)} \in \mathbb{R}^{d\_{\text{model}} \times d_k}$, $W_V^{(h)} \in \mathbb{R}^{d\_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{H d_v \times d\_{\text{model}}}$.

---

#### 3. Architecture / Algorithm Walkthrough

##### Data Flow in sequence Modeling
The lecture presents the fundamental structural trade-offs between the three key sequence processing paradigms:

```
RNN (Sequential Mixing):
X_1 ──> [Cell] ──> H_1 ──> X_2 ──> [Cell] ──> H_2 ──> ...
          │                          │
          └───(Sequential dependency prevents parallel training)

CNN (Local Sliding Window):
[   X_1   X_2   X_3   X_4   X_5   ]
   \     /     /
    \   /     /
   [  Y_1   Y_2  ] ──> (Highly parallel, but requires deep stacks to build global receptive field)

Transformer (Global Direct Matching):
X_1 ──┐
X_2 ──┼──> [Self-Attention Multi-head Operator] ──> Y_1, Y_2, Y_3, Y_4
X_3 ──┼──> (Highly parallel; every token interacts with every other token in one block)
X_4 ──┘
```

##### PyTorch Blueprint (Illustrative MHA and Transformer Block)
This PyTorch implementation maps the theoretical scaled dot-product mechanics, multi-head parallelization, and block layout directly to clean code:

```python
import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """
    Illustrative implementation of Multi-Head Self-Attention.
    Fuses the projection heads into a single large matrix multiply for hardware efficiency.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Combined projection weights for Queries, Keys, and Values
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None):
        # x shape: [Batch_Size (B), Seq_Len (N), d_model (D)]
        B, N, D = x.shape
        
        # Step 1: Project to Q, K, V simultaneously
        qkv = self.qkv_projection(x)  # Shape: [B, N, 3 * D]
        q, k, v = torch.chunk(qkv, 3, dim=-1) # Split into 3 tensors of shape [B, N, D]
        
        # Step 2: Reshape and transpose to isolate heads
        # Target shape for Q, K, V: [B, num_heads, N, d_k]
        q = q.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 3: Compute Scaled Dot-Product Attention scores
        # Similarity shape: [B, num_heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Mask out forbidden positions with large negative value
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Step 4: Multiply by Values to get context representation
        # Out shape: [B, num_heads, N, d_k]
        out = torch.matmul(attention_weights, v)
        
        # Step 5: Concat heads back and pass through output projection
        # Transpose back to [B, N, num_heads, d_k] and flatten heads to d_model (D)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_projection(out)


class TransformerBlock(nn.Module):
    """
    A standard pre-LN Transformer Block containing:
    Multi-Head Self-Attention -> Residual -> LayerNorm -> MLP -> Residual -> LayerNorm
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
        
        # Position-wise MLP applied independently to each vector
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x, mask=None):
        # Pre-LN Layer Normalization & Multi-Head Attention with Skip Connection
        attention_out = self.self_attention(self.layernorm1(x), mask=mask)
        x = x + attention_out
        
        # Pre-LN MLP Block with Skip Connection
        mlp_out = self.mlp(self.layernorm2(x))
        x = x + mlp_out
        
        return x
```

---

#### 4. Visual Intuition & Interpretability

##### Visual Machine Translation Alignment Maps
*   **Differentiable Alignment:** Visualizing the attention weight matrix $A \in \mathbb{R}^{T_y \times T_x}$ from a translation model reveals the grammatical differences between language domains:
    *   **Diagonal Paths:** A strict 1-to-1 diagonal signifies sequential alignment (e.g. English "we see" mapping directly to French equivalents).
    *   **Backward Anti-Diagonals:** Swapped grammatical rules materialize as backward paths. For instance, English noun-adjective orders (e.g., "European economic area") generate a localized 2D anti-diagonal when translated to French noun-adjective order ("zone économique européenne").
    *   **Unsupervised Grammar Induction:** The network is never given explicit syntactic trees; it discovers grammatical correspondences purely through end-to-end backpropagation on translation dataset pairs.

##### Permutation Equivariance Diagram
*   Without positional information, a self-attention block behaves as a permutation equivariant operator. If the input matrix $X$ is permuted by a permutation matrix $P$, the output is permuted identically:
$$\text{Attention}(PX, PX, PX) = P \cdot \text{Attention}(X, X, X)$$
This demonstrates that self-attention operates over sets rather than ordered vectors.

##### Receptive Field Expansion Comparison
*   **RNN Receptive Fields:** Information must propagate sequentially layer-by-layer, step-by-step. Receptive field propagation is bounded by temporal sequence length, leading to vanishing gradient problems for distant tokens.
*   **CNN Receptive Fields:** Receptive fields grow *linearly* with depth (or *exponentially* if strided). Expanding to global coverage requires deep convolutional stacks.
*   **Transformer Receptive Fields:** Receptive field is *instantaneously global* ($1.0$ coverage) in a single self-attention layer. Every token is capable of directly communicating with any other token regardless of distance.

---


<div id="plotly-cs231n-8-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 8 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Interactive Multi-Head Attention Matcher)
*   **Visualization Type:** Two-dimensional interactive heatmap alignment grid, paired with a bipartite graph node connector.
*   **Data Fields & Encoding:**
    *   **X-axis:** Source sequence tokens (e.g., input English sentence).
    *   **Y-axis:** Target sequence tokens (e.g., output French/Italian sentence).
    *   **Cell Intensity (Opacity):** Magnitude of the scalar attention weight $a\_{ti} \in [0, 1]$.
    *   **Line Color:** Color-coded by active Attention Head index $h \in \{1, \dots, H\}$, demonstrating how different heads specialize in different linguistic relations (e.g., Head 1 tracks verb-noun dependencies, Head 2 tracks noun-adjective order).
*   **Interactive Controls:**
    *   **Head Toggle Checkbox:** Filters the bipartite connector lines to display only selected heads, illustrating head specialization.
    *   **Sequence Length Slider:** Dynamically appends tokens and recalculates the $N \times N$ matrix, demonstrating the quadratic spatial growth of attention computation.

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **Pre-LN vs. Post-LN Stability:** Placing Layer Normalization *before* self-attention and MLP blocks (Pre-LN) creates a stable gradient highway, avoiding the vanishing/exploding gradient problems of deep Post-LN architectures.
*   **Transformer Scale-Up Limits:** The original 2017 Transformer launched at $12$ blocks and $\approx 200$ million parameters. By 2025, architectures of identical block configurations scaled seamlessly across multiple orders of magnitude to hundreds of blocks and trillions of parameters.
*   **ICLR 2025 Test of Time Awards:**
    *   **Bahdanau et al. (2015):** The seminal paper that introduced attention for machine translation ("Neural Machine Translation by Jointly Learning to Align and Translate") won the runner-up Test of Time Award at ICLR 2025.
    *   **Kingma & Ba (2015):** The Adam optimizer paper won the prestigious main Test of Time Award at ICLR 2025.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Silent Failures
*   **Vanishing Gradients in Softmax:** If scaled dot-product normalization ($1/\sqrt{d_k}$) is omitted, high-dimensional query-key vectors yield large dot products. These push the softmax values to extreme distributions (near 1 or 0), causing gradients to vanish and stalling learning.
*   **Quadratic Complexity Barrier ($O(N^2)$):** Because attention computes similarity scores for all token pairs, computing a forward pass on a sequence of length $N$ scales quadratically in both time and memory. For extremely long contexts, standard self-attention quickly triggers GPU Out-of-Memory (OOM) failures, motivating linear-time alternatives (like Mamba or State Space Models).
*   **Order Amnesia:** Forgetting to add positional embeddings to the token vectors makes the Transformer completely blind to sequence layout, treating "The cat ate the mouse" identically to "The mouse ate the cat".

##### Graduate-Level Reflection Questions
1.  **Strict Proof of Softmax Gradient Flattening:**
    Mathematically prove why a high-variance input vector $z \in \mathbb{R}^{d}$ causes the Jacobian of the softmax function $\sigma(z)$ to vanish. Why does dividing $z$ by $\sqrt{d}$ specifically counteract this phenomenon under the assumption that the elements of $z$ are computed as dot products of unit-variance, zero-mean independent random variables?
2.  **Transformer vs. Convolution Representational Capacity:**
    If self-attention is permutation equivariant and lacks translation equivariance, why does a Vision Transformer (ViT) require substantially more training data to achieve high accuracy compared to a traditional Convolutional Neural Network (CNN) on ImageNet? Discuss in terms of *inductive biases* vs. *representational capacity*.
3.  **Memory Analysis of Attention Backpropagation:**
    During the backward pass of a multi-head self-attention layer with sequence length $N$, hidden dimension $d$, and $H$ heads, compute the exact memory footprint required to store intermediate activations for backpropagating gradients through the softmax operator. Explain why this bottleneck limits context length far more severely than the parameter storage cost.
