# CS336 Lecture 4: Attention Alternatives & Mixture of Experts

## 0. Quick-Recall Summary
*   **The Linear Attention Duality**: By dropping the softmax normalizer, attention becomes associative: $(Q K^T) V = Q (K^T V)$. This unlocks a dual-representation paradigm where training is parallelized like a Transformer ($O(N)$ sequence length scaling), while inference operates incrementally as an RNN with constant state memory ($O(1)$ memory).
*   **SSM & Gated Delta Net Recurrences**: Modern state-space models generalize linear attention by introducing input-dependent gates (Mamba 2: $\gamma_t$) and projections (Gated Delta Net: $(I - \beta_t K_t K_t^T)$) to selectively update and erase the hidden state, bridging the representational gap with full attention.
*   **Sparse Attention (DSA)**: Rather than linearizing attention, models like DeepSeek-V3 use a lightweight, lower-dimensional indexer to run top-K selection, performing full softmax attention on a tiny, relevant subset of context to drastically lower KV caching and decode latency.
*   **Mixture of Experts (MoE) Scaling**: MoEs scale model capacity (sparse parameters) without increasing FLOPs per forward pass. Tokens are routed to the top-K of $N$ experts at the token level, specializing feedforward blocks dynamically while keeping compute costs flat.
*   **Load Balancing & Stability**: Sparse top-K routing is non-differentiable and prone to "expert collapse" (where a few experts process all tokens). This is mitigated by adding a load-balancing auxiliary loss ($F_i \cdot P_i$) and stabilizing the router's softmax with FP32 execution and router $z$-loss.

---

## 1. Core Paradigm & Systems Overview
*   **Objective**: The engineering goal of this lecture is to design architectures that bypass the quadratic $O(N^2)$ context-length scaling bottleneck of full attention, and to scale model parameter capacity without a corresponding increase in per-token FLOPS through sparsity.
*   **Primary Bottleneck**: During long-context decoding, retrieval of the KV Cache from global High-Bandwidth Memory (HBM) to processor SRAM scales linearly with context length, bottlenecking execution on **memory-bandwidth limits** rather than raw compute. For Mixture of Experts (MoE), when experts are sharded across separate hardware accelerators, routing tokens across devices over slow networks introduces an intense **communication bottleneck** (`All-to-All` collectives).
*   **Builds on**: This lecture builds on the architecture primitives introduced in Lecture 3 (such as pre-norm, SwiGLU, and positional embeddings) and the arithmetic intensity and memory-access concepts from Lecture 2 to analyze how attention alternatives and mixture-of-experts balance expressive power, training stability, and physical hardware deployment costs.

---

## 2. Theoretical & Mathematical Primitives

### 2.1 The Associativity Shift (Vanilla vs. Linear Attention)
In standard attention, query, key, and value matrices ($Q, K, V \in \mathbb{R}^{N \times D}$) are multiplied as follows:

$$Y = \text{softmax}\left(\frac{Q K^T}{\sqrt{D_k}}\right) V \quad [O(N^2 D_k + N D_k D_v)] \quad \text{}$$

If the softmax normalizer is omitted (or replaced by a feature map $\phi$):

$$Y = (Q K^T) V = Q (K^T V) \quad [O(N D_k D_v)] \quad \text{}$$

By shifting the parentheses via the associative property of matrix multiplication, the time complexity scales linearly with the sequence length $N$ rather than quadratically, provided $D_k, D_v \ll N$.

### 2.2 Recurrent State-Space Representation of Linear Attention
This associativity allows us to rewrite linear attention as an online recurrence (equivalent to an RNN):
*   **Hidden State Update**: 
    $$S_t = S\_{t-1} + K_t V_t^T \in \mathbb{R}^{D_k \times D_v} \quad \text{}$$
*   **Output Projection**: 
    $$Y_t = Q_t S_t \in \mathbb{R}^{D_v} \quad \text{}$$

### 2.3 Mamba 2 Stateful Gated Recurrence
Mamba 2 introduces an input-dependent, non-stateful forget gate $\gamma_t = \sigma(W_\gamma X_t) \in (0, 1)$ to selectively clear the historical state:

$$S_t = \gamma_t S\_{t-1} + K_t V_t^T \quad \text{}$$

$$Y_t = Q_t S_t + D \cdot V_t \quad \text{}$$

Where $D \cdot V_t$ is a parameterized skip connection directly passing the current token value to the output.

### 2.4 Gated Delta Net Projection Recurrence
Gated Delta Net adds a second gating signal $\beta_t = \sigma(W_\beta X_t) \in$ and uses a projection operator to forcefully erase the historical hidden state along the current key's spatial direction:

$$S_t = (I - \beta_t K_t K_t^T) S\_{t-1} + \beta_t K_t V_t^T \quad \text{}$$

$$Y_t = Q_t S_t \quad \text{}$$

Here, the matrix term $(I - \beta_t K_t K_t^T)$ acts as a spatial projector that project out components of the history matching the current key $K_t$, allowing the model to overwrite outdated facts instantly.

### 2.5 Top-K Token Choice routing (MoE)
Given input token representation $x_t \in \mathbb{R}^d$, router weights $W_g \in \mathbb{R}^{N\_{experts} \times d}$:
*   **Routing Logits**: 
    $$H(x_t) = x_t \cdot W_g^T \in \mathbb{R}^{N\_{experts}} \quad \text{}$$
*   **Expert Gate Probabilities**: 
    $$s_t = \text{softmax}(H(x_t)) \in \mathbb{R}^{N\_{experts}} \quad \text{}$$
*   **Sparse Token Routing Output**: 
    $$y_t = \sum\_{i \in \text{top-K}(s_t)} s\_{t, i} E_i(x_t) \quad \text{}$$

### 2.6 Switch Transformer Load Balancing Loss
To prevent "expert collapse" (where SGD updates reinforce a subset of experts, leaving the rest unused), a load-balancing loss $L\_{aux}$ is added over a batch of size $T$:

$$F_i = \frac{1}{T} \sum\_{t=1}^T \mathbb{I}(\text{expert } i \text{ is selected for token } t) \quad \text{}$$

$$P_i = \frac{1}{T} \sum\_{t=1}^T s\_{t, i} \quad \text{}$$

$$L\_{aux} = \alpha \cdot N\_{experts} \sum\_{i=1}^{N\_{experts}} F_i \cdot P_i \quad \text{}$$

During backpropagation, we treat the discrete fraction $F_i$ as a constant parameter, rendering the loss differentiable with respect to the router probability $P_i$:

$$\frac{\partial L\_{aux}}{\partial s\_{t, i}} \propto F_i \quad \text{}$$

This exerts a gradient penalty that forces the router to reduce the probability allocated to expert $i$ if $i$ receives a high fraction of tokens, driving the system back to uniform distribution.

### 2.7 Multi-Head Latent Attention (MLA) Compression
Instead of caching Keys $K_t$ and Values $V_t$ directly, MLA projects them into a low-dimensional compressed latent space $C_t \in \mathbb{R}^{D_c}$ ($D_c \ll D_h$):

$$C_t = W\_{DK} X_t \quad \text{}$$

$$K_t = W\_{UK} C_t, \quad V_t = W\_{UV} C_t \quad \text{}$$

Only the compressed latent vector $C_t$ needs to be cached in HBM, reducing KV cache memory footprint by up to 90% during decoding.

---

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### 3.1 Linear Attention Duality Blueprint
The following class demonstrates both the Parallel form (highly parallelized during pre-training) and the Recurrent form (computationally efficient constant-memory generation during inference) of linear attention:

```python
import torch
import torch.nn as nn
from einops import rearrange

class LinearAttentionDual(nn.Module):
    def __init__(self, d_model, d_head=64):
        super().__init__()
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        
    def _feature_map(self, x):
        # A simple non-negative feature map to replace softmax normalization
        return torch.relu(x) + 1e-6

    def forward_parallel(self, x):
        """
        Parallel Training Pass: O(N) sequence scaling when causal mask is absent.
        """
        B, N, D = x.shape
        q = self._feature_map(self.W_q(x)) # [B, N, D_head]
        k = self._feature_map(self.W_k(x)) # [B, N, D_head]
        v = self.W_v(x)                   # [B, N, D_head]
        
        # Associativity trick: compute (K^T * V) first
        # k: [B, N, D_k], v: [B, N, D_v]
        # k_trans_v: [B, D_k, D_v]
        k_trans_v = torch.einsum('b n k, b n v -> b k v', k, v)
        
        # Multiply with queries: Q * (K^T * V) -> [B, N, D_v]
        out = torch.einsum('b n k, b k v -> b n v', q, k_trans_v)
        return out

    def forward_recurrent(self, x, prev_state=None):
        """
        Incremental Inference step: Constant O(1) memory footprint per decoding token.
        """
        B, D = x.shape # Single token input step
        q = self._feature_map(self.W_q(x.unsqueeze(1))) # [B, 1, D_k]
        k = self._feature_map(self.W_k(x.unsqueeze(1))) # [B, 1, D_k]
        v = self.W_v(x.unsqueeze(1))                   # [B, 1, D_v]
        
        # Initialize zero state if none exists
        if prev_state is None:
            prev_state = torch.zeros(B, self.d_head, self.d_head, device=x.device)
            
        # Update recurrent state: S_t = S_{t-1} + K_t * V_t^T
        # k: [B, 1, D_k], v: [B, 1, D_v] -> update: [B, D_k, D_v]
        state_update = torch.einsum('b l k, b l v -> b k v', k, v)
        current_state = prev_state + state_update
        
        # Output calculation: Y_t = Q_t * S_t
        out = torch.einsum('b l k, b k v -> b l v', q, current_state).squeeze(1)
        return out, current_state
```

### 3.2 Top-K Token Choice Router with Load Balancing Loss
The following PyTorch module implements a standard Top-2 MoE router, computing both routing gates and the differentiable Switch Transformer auxiliary load-balancing loss:

```python
class Top2MoERouter(nn.Module):
    def __init__(self, d_model, n_experts, alpha=0.01):
        super().__init__()
        self.n_experts = n_experts
        self.alpha = alpha
        self.gate_proj = nn.Linear(d_model, n_experts, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, N, d_model]
        Returns:
            topk_idx: Expert indices for top-2 routing [B * N, 2]
            topk_gates: Differentiable gate scalers [B * N, 2]
            aux_loss: Switch Transformer balancing loss scalar
        """
        B, N, D = x.shape
        flat_x = rearrange(x, 'b n d -> (b n) d')
        T = flat_x.shape[0] # Total tokens in batch
        
        # Compute router logits and softmax probabilities in FP32 for numerical stability
        logits = self.gate_proj(flat_x.float()) # [T, n_experts]
        probs = torch.softmax(logits, dim=-1)   # [T, n_experts]
        
        # Hard top-2 selection (non-differentiable indices)
        topk_probs, topk_idx = torch.topk(probs, k=2, dim=-1) # [T, 2]
        
        # Differentiable gates: Renormalize over selected top-k
        topk_gates = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Switch Transformer Load Balancing Loss math:
        # F_i: Fraction of tokens dispatched to expert i
        flat_top1_idx = topk_idx[:, 0]
        F = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            F[i] = (flat_top1_idx == i).float().mean()
            
        # P_i: Average probability mass allocated to expert i
        P = probs.mean(dim=0) # [n_experts]
        
        # Auxiliary loss scalar: alpha * N * sum(F_i * P_i)
        aux_loss = self.alpha * self.n_experts * torch.dot(F, P)
        
        return topk_idx, topk_gates, aux_loss
```

---

## 4. Hardware Realities & Compute/Memory Accounting

### 4.1 Memory Footprint of KV Cache
During autoregressive decoding, keys and values of past tokens must be held in global HBM memory. If storing cache tensors in FP16/BF16 (2 bytes per element), the required memory capacity is:

$$\text{KV Cache Size per Token} = 2 \times N\_{layers} \times N\_{KV\_heads} \times D\_{head} \times 2 \quad (\text{Bytes}) \quad \text{}$$

For a standard **Llama-3-8B** model ($N\_{layers}=32$, $N\_{KV\_heads}=8$ under GQA, $D\_{head}=128$):

$$\text{Llama-3-8B KV Size per Token} = 2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ Bytes} \approx \mathbf{128 \text{ KB}}$$

For a context window of $128,000$ tokens, the KV cache alone demands **16 GB of memory per concurrent batch stream**, drastically bounding serving throughput on high-bandwidth memory (HBM) capacity.

### 4.2 Grouped Query Attention (GQA) Memory Savings
GQA acts as a structural compression factor:
*   **Multi-Head Attention (MHA)**: $N\_{KV\_heads} = N\_{Q\_heads}$ (No memory savings).
*   **Multi-Query Attention (MQA)**: $N\_{KV\_heads} = 1$ (Aggressive memory savings but degrades performance significantly due to query head expressiveness compression).
*   **Grouped-Query Attention (GQA)**: $1 < N\_{KV\_heads} < N\_{Q\_heads}$ (Retrieves nearly all MHA representational quality while capturing $8\times$ HBM reduction if queries are grouped into groups of 8).

### 4.3 Dropless MoE and MegaBlocks
Naive MoE routers route a variable number of tokens to each expert based on data distribution. In parallel GPU implementations, this asymmetry requires:
1.  **Static Padding**: Padding all expert inputs to a fixed maximum "expert capacity". This wastes immense compute on padding zeros, reducing MFU.
2.  **Token Dropping**: Silently dropping tokens that exceed the maximum expert buffer capacity, sending zeros back to the residual stream. This causes severe optimization decay and non-deterministic model behavior during training.

**MegaBlocks** removes this trade-off by representing the dynamic token routing as a large block-diagonal sparse matrix, executing a specialized *block-sparse matrix multiplication* kernel that handles variable token allocations per expert dynamically on silicon without padding or token loss.

---

## 5. Visualization Blueprint: MoE Routing & Load Balancing Dynamics

<div id="plotly-cs336-4-moe-routing" class="plotly-chart" aria-label="Interactive Plotly chart: MoE Expert Load Balancing"></div>

<p><em>Figure: MoE Expert Load Balancing — Auxiliary load-balancing loss prevents routing collapse onto a few popular experts.</em></p>

<div id="plotly-cs336-4-linear-attention" class="plotly-chart" aria-label="Interactive Plotly chart: Linear Attention / SSM Memory Scaling"></div>

<p><em>Figure: Memory Scaling — State-Space Models (SSM) maintain a constant recurrent state size during inference, escaping the $O(T^2)$ attention wall.</em></p>


*   **Visualization Type**: Interactive Matrix-Grid & Expert Load Balance Simulator.
*   **Data Fields & Encoding**:
    *   **X-axis**: Individual input tokens from a sequence ($t \in [1, \dots, T]$).
    *   **Y-axis**: Individual experts ($e \in [1, \dots, N\_{experts}]$).
    *   **Cell Fill Color**: Color intensity maps to the router probability $s\_{t, e}$. The cell is highlighted with a thick border if expert $e$ is in the `top-2` selection for token $t$.
    *   **Right Bar-Chart Margin**: Displays a real-time bar-chart showing the exact load fraction $F_i$ for each expert to visually illustrate balancing.
*   **Interactive Controls**:
    *   **Auxiliary Loss Coefficient ($\alpha$) Slider**: Adjusts $\alpha$ from `0.0` (unregularized) to `1.0`.
    *   **Stochastic Perturbation Noise Slider**: Toggles injecting Gaussian noise to routing logits as suggested by Shazir.
*   **Concepts Demonstrated**:
    *   Set **$\alpha = 0.0$**: The matrix instantly transitions to showing **Expert Collapse**. All tokens route to the same 2 experts (cells light up in intense color on only two horizontal rows), while the remaining experts register a load fraction of zero.
    *   Increase **$\alpha \ge 0.01$**: The matrix immediately spreads out. Active cells distribute uniformly across the grid, demonstrating the balancing forces of Switch Transformer auxiliary penalties.

---

## 6. Empirical Scaling Laws & Hyperparameter Heuristics

### 6.1 Attention Hybrid Layer Alternation
Pure linear attention / state space models struggle to preserve long-range associative retrieval (such as multi-hop reasoning or key-value retrieval) without some full-attention heads. Modern scaling follows strict hybrid heuristics:
*   **Minimax M1**: Combines linear attention and softmax attention at a **7:1 ratio** (seven linear attention layers, one full attention layer).
*   **Qwen 3.5 Hybrid**: Combines Gated Delta Net recurrence with full attention at a **3:1 ratio** (three Gated Delta Net layers, one full attention layer).

### 6.2 Mixture of Experts (MoE) Scaling Laws
*   **The Parameter-to-Compute Efficiency Loop**: As you scale the number of experts $N$ while keeping active forward FLOPs constant, loss consistently decreases across the entire pre-training curve.
*   **Fine-Grained Expert Partitioning**: Traditional systems routed tokens to a few massive experts (e.g., top-1 out of 16 experts). DeepSeek proved that dividing the same parameter pool into highly granular experts (e.g., 64 or 128 smaller experts) dramatically increases specialized learning efficiency.
*   **Shared Experts Always-on**: To prevent redundant learning of general-purpose representations (such as punctuation or basic syntax) across routed experts, modern configurations designate a subset of expert parameters to be "shared experts" which are always executed, bypassing the top-K routing gate completely.

---

## 7. Systems Warnings, Pitfalls, & Reflection Questions

### 7.1 Gotchas & Common Bugs
1.  **Differentiability Trap in Hard Top-K Selection**: Softmax probability calculation is differentiable, but the `torch.topk` indices extraction is a discrete operation. The gradient must pass through the router's soft probability gates: $y = \sum g_i E_i(x)$. If you naively route the tensor without scaling the outputs of the experts by the continuous gating scores $s\_{t,i}$, the router parameters $W_g$ will receive a gradient of exactly zero and fail to learn.
2.  **Unstable Router Softmax (Catastrophic Overflow)**: Router weights can drift, resulting in massive logits. When exponentiated during routing softmax, this causes sudden underflow/overflow (producing `NaN` gates). Always calculate routing logits and softmax in **FP32** precision and include a **Z-loss regularization** term ($10^{-4} \log^2 Z$) to prevent log normalizer drift.
3.  **Fine-tuning Overfitting with MoE Parameters**: MoE models have billions of sparse parameters that quickly overfit on small downstream supervised datasets, creating a massive train-validation performance gap. Mitigate this by freezing the sparse expert weights completely during SFT, updating only the dense attention layers.

### 7.2 Conceptual Graduate-Level Reflection Questions

**Q1: In Gated Delta Net, how does the projector term $(I - \beta_t K_t K_t^T)$ preserve the parallel-recurrent training duality, and why does an LSTM's state-dependent gate fail this test?**
*   **Answer**: The recurrent update of Gated Delta Net is $S_t = (I - \beta_t K_t K_t^T) S\_{t-1} + \beta_t K_t V_t^T$. Notice that the coefficient on $S\_{t-1}$, which is $(I - \beta_t K_t K_t^T)$, depends *only* on the current input key $K_t$ and gating signal $\beta_t$ (which are derived purely from the input $X_t$). Because this forgetting transition matrix is strictly input-dependent and contains no dependencies on the historical state $S\_{t-1}$, we can write out the unrolled recurrence as a series of linear matrix products that can be computed in parallel using a scan prefix or parallel matrix multiplications during training.
*   In contrast, a traditional LSTM forget gate is *state-dependent*: $f_t = \sigma(W_f x_t + U_f h\_{t-1})$. Because $f_t$ depends on $h\_{t-1}$, computing the gate value at step $t$ requires strictly finishing the computation at step $t-1$. This serial dependency prevents parallel training execution, forcing $O(N)$ sequential operations.

**Q2: Analyze the mathematical impact of adding a shared general-purpose expert to a top-2 routed MoE. If we split a dense FFN layer of size $d\_{ffn}$ into $M$ routed experts of size $\frac{d\_{ffn}}{M}$ and 1 shared expert of size $d\_{ffn}$, what is the impact on active FLOPs, parameter count, and HBM memory footprint?**
*   **Answer**: 
    *   **Active FLOPs**: A single forward token pass through a standard top-2 MoE activates exactly 2 routed experts. If a shared expert of size $d\_{ffn}$ is added, the active compute becomes:
        $$\text{FLOPs}\_{active} = \text{Compute}(E\_{shared}) + 2 \times \text{Compute}(E\_{routed}) \propto d\_{ffn} + 2 \times \left(\frac{d\_{ffn}}{M}\right) = d\_{ffn} \left(1 + \frac{2}{M}\right)$$
    *   **Parameter Count**: The total parameters of the MLP blocks equal:
        $$\text{Params}\_{total} = \text{Params}(E\_{shared}) + M \times \text{Params}(E\_{routed}) \propto d\_{ffn} + M \times \left(\frac{d\_{ffn}}{M}\right) = 2 \times d\_{ffn}$$
    *   **HBM Footprint**: To execute the shared expert, its weights must be held in the HBM memory of *every* GPU (all nodes). In contrast, the $M$ sparse experts can be partitioned (expert-parallelism) across $P$ different GPUs, storing only $\frac{M}{P}$ experts on any individual GPU. The shared expert therefore increases local GPU memory consumption, acting as a non-shardable memory footprint constant.

**Q3: Why does standard autoregressive decoding of transformers become bottlenecked on memory bandwidth rather than FLOP capacity, and how does Grouped-Query Attention (GQA) directly shift this roofline location?**
*   **Answer**: During autoregressive decoding, a single token is generated per step. This single token is converted to query, key, and value vectors of size $D_h$. While query is a vector, we must load the *entire* historical KV Cache of size $N\_{tokens} \times D_h$ from HBM to compute attention, and load all the model weight matrices from HBM to project it. Because we perform a matrix-vector multiplication (where we do $2 \times D\_{model}$ operations per element loaded), our **Arithmetic Intensity** is extremely low:
    $$\text{Arithmetic Intensity} \propto \frac{\text{FLOPs}}{\text{Bytes Access}} \propto \frac{1}{\text{Batch Size}}$$
    This places the decoder deeply in the diagonal, memory-bound region of the roofline plot.
*   GQA groups query heads to share a single key-value head. This reduces the number of key-value cache elements that must be read from HBM by a factor of $G$ (the group size). Consequently, the bytes transferred from HBM drop by $G\times$, directly increasing the Arithmetic Intensity of the attention operation, shifting the model's operating point on the roofline curve further to the right toward the compute-saturated flat region.
