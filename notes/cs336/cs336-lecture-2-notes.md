# CS336 Lecture 2: PyTorch & Resource Accounting

## 0. Quick-Recall Summary
* **The $6NT$ Rule:** Training a model with $N$ parameters on $T$ tokens requires approximately $6NT$ FLOPs ($2NT$ for the forward pass, $4NT$ for the backward pass).
* **Mixed Precision Layout:** For stable, memory-efficient pre-training, parameters, activations, and gradients use 16-bit precision (typically BF16, 2 bytes), while optimizer states are held in FP32 (4 bytes).
* **Arithmetic Intensity Boundary:** Defined as $\text{FLOPs} / \text{Bytes Transferred}$. An operation is memory-bound if its intensity is less than the hardware's intensity ceiling ($\approx 295 \text{ FLOPs/byte}$ for NVIDIA H100), and compute-bound if greater.
* **GEMM vs. Element-wise Ops:** Matrix multiplication ($n \times n$) has an arithmetic intensity of $\approx n/3$, making it compute-bound for large $n$. Almost all other operations (ReLU, GELU, dot products, layer norm) are highly memory-bound.
* **Memory Mitigation:** *Gradient Accumulation* handles large batch sizes by accumulating gradients across micro-batches. *Activation Checkpointing* reduces activation memory from $O(L)$ to $O(\sqrt{L})$ by trading redundant forward-pass compute during the backward-pass.

---

## 1. Core Paradigm & Systems Overview
* **Objective:** Establish the foundational principles of low-level resource accounting (FLOPs counting and memory mapping) required to maximize hardware efficiency during LLM training. This lecture bridges the gap between high-level PyTorch tensor manipulations (utilizing `einops` for readable, bug-free implementations) and the hardware-level limits of memory bandwidth and raw compute power.
* **Primary Bottleneck:** Both memory-bound and compute-bound regimes are analyzed. Under-the-hood, almost all non-GEMM operators (activations, normalizations, element-wise transformations) are strictly **memory-bound** because of the massive latency gap between moving data from High Bandwidth Memory (HBM) to the processor vs. executing the calculations on-chip. Deep learning workloads are designed to group operations into massive GEMMs to saturate the compute units and escape the memory bandwidth bottleneck.
* **Builds on:** Lecture 1 introduced the concept of tokenization acting as a compression layer to reduce sequence lengths; Lecture 2 quantifies the compute savings of this compression by analyzing the exact hardware FLOPs and memory cost of processing those tokens through network layers.

---

## 2. Theoretical & Mathematical Primitives

### 2.1 Floating-Point Formats & Precision
Floating-point representations balance dynamic range (controlled by exponent bits) against numerical precision/resolution (controlled by mantissa bits):
* **FP32 (Single Precision):**
  $$\text{Bits} = 32 \quad (1 \text{ sign}, 8 \text{ exponent}, 23 \text{ mantissa}) \quad \text{Size} = 4 \text{ bytes}$$
* **FP16 (Half Precision):**
  $$\text{Bits} = 16 \quad (1 \text{ sign}, 5 \text{ exponent}, 10 \text{ mantissa}) \quad \text{Size} = 2 \text{ bytes}$$
  * *System Pitfall:* Narrow dynamic range causes frequent underflow/overflow (e.g., $1\text{e-}8$ underflows to $0$), triggering training instability and NaNs.
* **BF16 (Brain Floating Point):**
  $$\text{Bits} = 16 \quad (1 \text{ sign}, 8 \text{ exponent}, 7 \text{ mantissa}) \quad \text{Size} = 2 \text{ bytes}$$
  * *Systems Advantage:* Matches the dynamic range of FP32, preventing underflow/overflow issues without requiring complex gradient scaling, at the cost of precision.

### 2.2 Arithmetic Intensity & Hardware Bottlenecks
* **Accelerator Arithmetic Intensity ($\text{Intensity}\_{\text{acc}}$):**
  $$\text{Intensity}\_{\text{acc}} = \frac{\text{Peak FLOPs/sec}}{\text{Memory Bandwidth (Bytes/sec)}} \approx \frac{989.5 \times 10^{12} \text{ FLOPs/sec}}{3.3 \times 10^{12} \text{ Bytes/sec}} \approx 295 \text{ FLOPs/byte} \quad \text{(for NVIDIA H100 dense BF16)}$$
* **Algorithmic Arithmetic Intensity ($\text{Intensity}\_{\text{algo}}$):**
  $$\text{Intensity}\_{\text{algo}} = \frac{\text{FLOPs required by operator}}{\text{Bytes moved to/from HBM}}$$
* **Operational Regimes:**
  $$\text{Memory Bound if: } \text{Intensity}\_{\text{algo}} < \text{Intensity}\_{\text{acc}} \implies \text{Latency dictated by HBM bandwidth}$$
  $$\text{Compute Bound if: } \text{Intensity}\_{\text{algo}} > \text{Intensity}\_{\text{acc}} \implies \text{Latency dictated by raw tensor core speed}$$

### 2.3 Mathematical Derivation of $6NT$ FLOPs Accounting
Consider a forward pass of a linear layer map $Y = XW$ where $X \in \mathbb{R}^{B \times D}$ (batch/sequence dimension $B$, input dimension $D$) and $W \in \mathbb{R}^{D \times K}$ (output dimension $K$):
* **Forward Pass FLOPs:** For each output element, we compute a dot product of length $D$. This involves $D$ multiplications and $D-1$ additions, which we approximate as $2D$ operations. Doing this for all $B \times K$ elements in $Y$ yields:
  $$\text{FLOPs}\_{\text{fwd}} = 2 \cdot B \cdot D \cdot K$$
  If $D = K$, and $N = D^2$ (parameters), this reduces to $2 \cdot B \cdot N$.

* **Backward Pass FLOPs:** To backpropagate, we apply the chain rule to compute two gradients:
  1. **Gradient with respect to inputs ($H\_{1\text{.grad}}$):**
     $$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} W^T \quad \implies \quad X\_{\text{.grad}} \in \mathbb{R}^{B \times D} = Y\_{\text{.grad}} \cdot W^T$$
     This is a matrix multiplication of shapes $(B \times K)$ and $(K \times D)$, requiring:
     $$\text{FLOPs}\_{\text{grad\_input}} = 2 \cdot B \cdot D \cdot K$$
  2. **Gradient with respect to parameters ($W\_{\text{.grad}}$):**
     $$\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Y} \quad \implies \quad W\_{\text{.grad}} \in \mathbb{R}^{D \times K} = X^T \cdot Y\_{\text{.grad}}$$
     This is a matrix multiplication of shapes $(D \times B)$ and $(B \times K)$, requiring:
     $$\text{FLOPs}\_{\text{grad\_param}} = 2 \cdot B \cdot D \cdot K$$

* **Combined Accounting:**
  $$\text{FLOPs}\_{\text{bwd}} = \text{FLOPs}\_{\text{grad\_input}} + \text{FLOPs}\_{\text{grad\_param}} = 4 \cdot B \cdot D \cdot K = 2 \cdot \text{FLOPs}\_{\text{fwd}}$$
  $$\text{Total Training FLOPs} = \text{FLOPs}\_{\text{fwd}} + \text{FLOPs}\_{\text{bwd}} = 6 \cdot B \cdot D \cdot K$$
  Summing across a training run processing $T$ tokens on a model with $N$ parameters yields the classic resource estimation:
  $$\text{Total Run FLOPs} = 6 \cdot N \cdot T$$

---

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### 3.1 Declarative Tensor Manipulations using `einops` and `einsum`
This illustrative blueprint shows how to implement multi-head projection, QK transposition, and manual backward gradients using Einstein notation and `einops`, matching the declarative standard taught in the course.

```python
import torch
import torch.nn as nn
from einops import rearrange, reduce

class CS336MultiHeadAttentionProjection(nn.Module):
    """
    Illustrative implementation mapping multi-head QKV projections and transposes
    to einops/einsum representations, preventing messy transposes.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        
        # Combined projection matrix for Queries, Keys, and Values (QKV)
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=False) # Bias term dropped for system efficiency
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, seq_len, d_model]
        batch, seq_len, _ = x.shape
        
        # Project to QKV
        qkv = self.qkv_projection(x) # [batch, seq_len, 3 * d_model]
        
        # Split into Q, K, V and unflatten heads using einops rearrange
        # We transition from [batch, seq_len, 3 * d_model] -> [3, batch, n_heads, seq_len, d_head]
        q, k, v = rearrange(
            qkv, 
            'b s (three h d) -> three b h s d', 
            three=3, 
            h=self.n_heads, 
            d=self.d_head
        )
        
        # Perform Attention Score compute using einsum: Score = Q K^T / sqrt(d_head)
        # Q shape: [batch, n_heads, seq_len, d_head] -> 'b h i d'
        # K shape: [batch, n_heads, seq_len, d_head] -> 'b h j d'
        # Output Score: [batch, n_heads, seq_len, seq_len] -> 'b h i j'
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) / (self.d_head ** 0.5)
        
        # Softmax over key sequence dimension 'j'
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Weighted value accumulation: Output = Weight V
        # Weight shape: [batch, n_heads, seq_len, seq_len] -> 'b h i j'
        # V shape: [batch, n_heads, seq_len, d_head] -> 'b h j d'
        # Output context: [batch, n_heads, seq_len, d_head] -> 'b h i d'
        context = torch.einsum('b h i j, b h j d -> b h i d', attn_weights, v)
        
        # Rearrange back to contiguous d_model shape: [batch, seq_len, d_model]
        out = rearrange(context, 'b h s d -> b s (h d)')
        return out

def compute_manual_gradients(x: torch.Tensor, w: torch.Tensor, d_loss_dy: torch.Tensor):
    """
    Explicitly illustrates the math behind gradient calculations using Einstein summation,
    demonstrating how backprop matches forward operations in cost.
    
    Forward: Y = XW (b: batch, d: input_dim, k: output_dim)
    """
    # 1. Gradient with respect to input X (d_loss_dx): shape [batch, input_dim]
    # Summing out the output dimension 'k'
    d_loss_dx = torch.einsum('b k, d k -> b d', d_loss_dy, w)
    
    # 2. Gradient with respect to weight W (d_loss_dw): shape [input_dim, output_dim]
    # Summing out the batch dimension 'b'
    d_loss_dw = torch.einsum('b d, b k -> d k', x, d_loss_dy)
    
    return d_loss_dx, d_loss_dw
```

### 3.2 Memory-Efficient Training Patterns: Checkpointing & Accumulation
This blueprint illustrates how to construct the training loop with micro-batches for Gradient Accumulation and utilize Activation Checkpointing.

```python
def gradient_accumulation_step(model, optimizer, batch_x, batch_y, micro_batch_size, accum_steps):
    """
    Executes training across micro-batches to emulate a large batch size 
    without the activation memory footprint.
    """
    optimizer.zero_grad()
    
    # Split the massive batch into micro-batches
    chunks_x = torch.split(batch_x, micro_batch_size)
    chunks_y = torch.split(batch_y, micro_batch_size)
    
    for i, (micro_x, micro_y) in enumerate(zip(chunks_x, chunks_y)):
        # Forward pass & loss calculation
        pred = model(micro_x)
        loss = compute_loss(pred, micro_y)
        
        # Scale loss by accumulation steps to average the gradients properly
        loss_scaled = loss / accum_steps
        
        # Backward pass accumulates gradients on parameters in-place
        loss_scaled.backward()
        
    # After processing all micro-batches, step the optimizer exactly once
    optimizer.step()

# --- Activation Checkpointing Integration ---
from torch.utils.checkpoint import checkpoint

class CheckpointedLayerBlock(nn.Module):
    def __init__(self, linear_dim):
        super().__init__()
        self.linear = nn.Linear(linear_dim, linear_dim, bias=False)
        self.activation = nn.ReLU() # Activations typically saved in forward pass
        
    def _inner_forward(self, x):
        return self.activation(self.linear(x))
        
    def forward(self, x):
        # Wraps the forward pass in torch.utils.checkpoint to free intermediate 
        # activations and recompute them on-demand during the backward pass
        return checkpoint(self._inner_forward, x, use_reentrant=False)
```

---

## 4. Hardware Realities & Compute/Memory Accounting

### 4.1 Memory Footprint of Training States
For mixed precision training using **BF16** for the model and **AdamW** for optimization:
1. **Model Parameters:** $2 \text{ bytes per parameter}$ (BF16).
2. **Gradients:** $2 \text{ bytes per parameter}$ (BF16).
3. **AdamW Optimizer States:** $8 \text{ bytes per parameter}$ (FP32):
   * First-order momentum ($m$): $4 \text{ bytes}$.
   * Second-order variance ($v$): $4 \text{ bytes}$.
4. **Master Parameters:** Often kept in $4 \text{ bytes}$ (FP32) for precise gradient updates.

**Static Training Footprint:** $\approx 12 \text{ to } 16 \text{ bytes}$ per parameter. 
* *Example (H100 constraint):* 8 H100 GPUs provide $8 \times 80\text{ GB} = 640\text{ GB}$ of High Bandwidth Memory (HBM). Allocating $12 \text{ bytes}$ per parameter leaves a maximum theoretical capacity of:
  $$\frac{640 \times 10^9 \text{ bytes}}{12 \text{ bytes/parameter}} \approx 53.3 \text{ Billion Parameters}$$
  *Note:* This calculation represents a strict upper bound that completely excludes dynamic Activation Memory.

### 4.2 Arithmetic Intensity Matrix
The arithmetic intensity of key mathematical operations in deep learning, calculated for half-precision/bfloat16 data ($2 \text{ bytes per element}$), illustrates why operations are memory-bound:

| Operation | Equation | FLOPs | HBM Memory Access (Bytes) | Arithmetic Intensity ($\text{FLOPs/Byte}$) | Bound Type (H100) |
|:--- |:--- |:--- |:--- |:--- |:--- |
| **ReLU (Vector $n$)** | $y_i = \max(x_i, 0)$ | $n$ | $2n \text{ (read)} + 2n \text{ (write)} = 4n$ | $\frac{n}{4n} = \mathbf{0.25}$ | **Memory Bound** |
| **GELU (Vector $n$)** | $y_i = \text{GELU}(x_i)$ | $20n$ | $2n \text{ (read)} + 2n \text{ (write)} = 4n$ | $\frac{20n}{4n} = \mathbf{5.0}$ | **Memory Bound** |
| **Dot Product (Vector $n$)**| $y = u^T v$ | $2n$ | $2n \text{ (read } u) + 2n \text{ (read } v) + 2 \text{ (write)} \approx 4n$ | $\frac{2n}{4n} = \mathbf{0.5}$ | **Memory Bound** |
| **Matrix-Vector (Vector $n$, Matrix $n \times n$)** | $y = Wx$ | $2n^2$ | $2n^2 \text{ (read } W) + 2n \text{ (read } x) + 2n \text{ (write } y) \approx 2n^2$ | $\frac{2n^2}{2n^2} = \mathbf{1.0}$ | **Memory Bound** |
| **Matrix-Matrix (GEMM $n \times n$)** | $Y = WX$ | $2n^3$ | $2n^2 \text{ (read } W) + 2n^2 \text{ (read } X) + 2n^2 \text{ (write } Y) = 6n^2$ | $\frac{2n^3}{6n^2} = \mathbf{\frac{n}{3}}$ | **Compute Bound (if $n > 885$)** |

* **Operator Fusion Concept:** To mitigate the memory-bound limits of element-wise layers (like computing ReLU, then adding, then norming), modern systems use *Operator Fusion*. This merges multiple operations into a single kernel, reading the input once from HBM, executing both calculations sequentially inside the high-speed cache of the accelerator, and writing the final result back to HBM. This keeps the data local and drastically increases the operational arithmetic intensity.

---

## 5. Visualization Blueprint (Roofline Analysis)

<div id="plotly-cs336-2-roofline" class="plotly-chart" aria-label="Interactive Plotly chart: NVIDIA H100 Roofline Analysis"></div>

<p><em>Figure: NVIDIA H100 Roofline Analysis (Dense BF16) — memory-bound operators (LayerNorm, Softmax, GELU) vs compute-bound GEMMs ($N \ge 896$).</em></p>

<div id="plotly-cs336-2-memory-breakdown" class="plotly-chart" aria-label="Interactive Plotly chart: Training Memory Footprint Breakdown"></div>

<p><em>Figure: Training Memory Footprint — Activation Checkpointing reduces activation memory from $O(L)$ to $O(\sqrt{L})$ at scale.</em></p>


This specification outlines how to construct a custom interactive Roofline Model visualization to diagnose hardware-level bottlenecks.

* **Visualization Type:** Double-Logarithmic Line & Scatter Plot (X-axis: Arithmetic Intensity, Y-axis: Achieved FLOPs/sec).
* **Visual Elements & Encodings:**
  * **X-axis (Log Scale):** Arithmetic Intensity ($\text{FLOPs / Byte}$), ranging from $0.1$ to $1000$.
  * **Y-axis (Log Scale):** Achieved Performance ($\text{TFLOPs/s}$), ranging from $1$ to $1000$.
  * **The Roofline Ceiling:** 
    * A diagonal line with slope equal to the system's memory bandwidth ($3.3 \text{ TB/s}$ for H100) representing the **Memory-Bound Ceiling**.
    * A flat horizontal line at the peak dense matrix math performance ($989.5 \text{ TFLOPs/s}$ for H100) representing the **Compute Ceiling**.
    * The point of intersection represents the **Knee Point** (for H100, this sits at $\approx 295 \text{ FLOPs/byte}$).
  * **Scatter Data Points:** Markers representing individual operations (ReLU, GELU, Dot Product, Matrix-Vector, and GEMM at different matrix sizes $n$) placed at their exact arithmetic intensities to visualize whether they land under the diagonal slope (memory-bound) or the horizontal line (compute-bound).
* **Interactive Controls:**
  * **Hardware Model Dropdown:** Toggle between GPUs (e.g., A100, H100, B200) to dynamically recalculate and animate the ceilings and knee points based on their respective bandwidth and FLOP limits.
  * **Matrix Size Slider ($n$):** A slider controlling the dimension $n$ of the GEMM operation. As $n$ increases, the GEMM marker dynamically slides along the line from the memory-bound region over the knee point and into the compute-bound plateau.

---

## 6. Empirical Scaling Laws & Hyperparameter Heuristics
* **Model Flops Utilization (MFU) Target:** Real-world training runs strive for an MFU of $\approx 0.5$ (50% of the theoretical hardware peak). Lower utilization (e.g., $\approx 0.1$) indicates critical bottlenecks in data routing, communication, or non-fused operators that must be profiled.
* **Batch Size Scaling:** Training uses a dynamic heuristic where batch sizes are scaled up over the course of training to improve gradient stability, capped at a "critical batch size". To avoid running out of memory from large activation states at these batch sizes, systems apply *Gradient Accumulation*.

---

## 7. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
* **Asynchronous CUDA Timing Bug:** GPU kernel launches are asynchronous and non-blocking in PyTorch. If you measure time in Python without inserting explicit synchronization barriers (`torch.cuda.synchronize()`), you will profile only the CPU launch latency instead of the actual GPU execution time. Your numbers will look misleadingly fast.
* **FP16 Underflow/Overflow:** Do not train deep networks with FP16 without mixed-precision gradient scaling. The limited 5-bit exponent cannot handle the gradient scales of deep networks, leading to rapid NaN propagation. BF16 should be preferred whenever hardware supports it.
* **Optimizer State Quantization Pitfall:** Naively converting optimizer states to FP16 or BF16 to save memory ruins training stability. Because optimizers track small, accumulated rolling averages and compute squared terms, they require the full 32-bit dynamic range to prevent updates from zeroing out or diverging.

### Conceptual Reflection Questions

1. **Given a model trained on an H100 node with peak dense performance of $989.5 \text{ TFLOPs/s}$ and $3.3 \text{ TB/s}$ memory bandwidth. What is the minimum matrix dimension $n$ for an $n \times n$ matrix-matrix multiplication (GEMM) to theoretically escape the memory-bound regime?**
   * *Answer:* The algorithm is compute-bound if its arithmetic intensity is greater than the hardware intensity ceiling ($\text{Intensity}\_{\text{acc}}$).
     $$\text{Intensity}\_{\text{acc}} = \frac{989.5 \times 10^{12} \text{ FLOPs/s}}{3.3 \times 10^{12} \text{ Bytes/s}} \approx 299.85 \text{ FLOPs/byte}$$
     The arithmetic intensity of GEMM for $n \times n$ matrices in BF16 is $\approx n/3 \text{ FLOPs/byte}$. Set the two equal:
     $$\frac{n}{3} \ge 299.85 \implies n \ge 899.55$$
     Therefore, the matrix dimension must be at least $900 \times 900$ to transition into the compute-bound regime.

2. **Why does activation checkpointing save a massive amount of memory, and what is the exact computational cost of applying it? Explain the square-root rule.**
   * *Answer:* Normally, backpropagation requires keeping the activations of all $L$ layers in memory, scaling as $O(L)$. If we use activation checkpointing and store activations only at intervals of $k$ layers, we only hold $L/k$ checkpointed layers in memory. During the backward pass, we recompute the missing activations within each $k$-layer segment on-the-fly, which requires at most $k$ forward-pass calculations at any time.
     The total memory overhead becomes $O(L/k) + O(k)$. To minimize this memory, we differentiate with respect to $k$ and find the optimal interval is $k = \sqrt{L}$, which reduces the activation memory footprint to $O(\sqrt{L})$. The computational overhead is exactly one extra forward pass for the checkpointed layers, increasing total training compute by $\approx 33\%$ (from $6NT$ to $8NT$, as the forward pass is re-run once).

3. **In mixed-precision training, we use BF16 for parameters and gradients, but FP32 for the optimizer states. What is the exact memory footprint (in bytes) per parameter for this setup under AdamW, and how does it compare to a full FP32 baseline?**
   * *Answer:*
     * *Mixed-Precision AdamW Footprint:* Parameters = $2\text{ bytes}$ (BF16); Gradients = $2\text{ bytes}$ (BF16); AdamW First Moment ($m$) = $4\text{ bytes}$ (FP32); AdamW Second Moment ($v$) = $4\text{ bytes}$ (FP32). Total = $12\text{ bytes/parameter}$ (or $16\text{ bytes}$ if keeping an FP32 master weight parameter copy).
     * *Full FP32 Baseline Footprint:* Parameters = $4\text{ bytes}$; Gradients = $4\text{ bytes}$; AdamW First Moment ($m$) = $4\text{ bytes}$; AdamW Second Moment ($v$) = $4\text{ bytes}$. Total = $16\text{ bytes/parameter}$ (or $20\text{ bytes}$ if keeping an explicit copy).
     * Mixed precision saves $4\text{ bytes per parameter}$ of high-speed device memory, while allowing the tensor cores to run matrix multiplications in 16-bit, which is twice as fast as 32-bit execution.
