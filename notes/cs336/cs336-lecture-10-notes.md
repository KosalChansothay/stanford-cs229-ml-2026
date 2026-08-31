# CS336 Lecture 10: Inference (Execution & Optimization)

## 0. Quick-Recall Summary
- **The Memory Wall of Decoding**: Unlike training which is compute-bound, generative inference is fundamentally **memory-bandwidth bound** due to the sequential generation of tokens, yielding an arithmetic intensity of $\sim 1$.
- **KV Cache Optimization**: Maintaining past key-value representations in HBM avoids redundant $O(T^2)$ prefix computations.
- **GQA & MLA Cache Compression**: Grouped Query Attention (GQA) and Multi-Latent Attention (MLA) compress key-value heads down to low-dimensional projections, massively cutting the KV cache storage footprints.
- **Speculative Decoding**: Employs a cheap draft model to generate candidate tokens in series, then verifies them in parallel in a single compute-dense forward pass of the target model.
- **PagedAttention**: Prevents physical memory fragmentation by partitioning the KV cache into non-contiguous memory blocks (pages), unlocking dynamic prefix sharing across concurrent requests.

## 1. Core Paradigm & Systems Overview
- **Objective**: Optimize auto-regressive decoding performance across critical production dimensions: Time to First Token (TTFT), token generation latency (seconds per token), and system throughput (tokens per second).
- **Primary Bottleneck**: HBM Bandwidth-bound. The GPU is forced to act as a "glorified memory loader," streaming the entire model parameter set from slow global HBM to fast local SRAM just to generate a single token.
- **Builds on**: Triton kernels and Hardware specs (Lecture 5 & 6) — uses memory-access coalescing and cache hierarchies to accelerate decoding under tight bandwidth constraints.

## 2. Theoretical & Mathematical Primitives

### Arithmetic Intensity of Decoding (MatMul)
For a matrix-vector product $y = xW$ (where $x \in \mathbb{R}^{1 \times D}$ and $W \in \mathbb{R}^{D \times F}$ under BF16):
- **Flops**: $2 \times D \times F$
- **Bytes transferred**: $2 \times (D + D \times F + F) \text{ bytes}$
- **Arithmetic Intensity**:
  $$I = \frac{2 \cdot D \cdot F}{2 \cdot (D + D \cdot F + F)} \approx 1 \text{ Flop/byte}$$

Since modern accelerators (like the A100/H100) require $I \ge 150-300$ to saturate compute cores, generation is completely memory-bandwidth bound.

### KV Cache Memory Footprint
For batch size $B$, sequence length $S$, layer count $L$, number of KV heads $k\_{\text{heads}}$, and head dimension $H$ under BF16:

$$\text{Memory}\_{\text{KV}} = 2 \times 2 \times (B \times S \times L \times k\_{\text{heads}} \times H) \text{ bytes}$$

Where the initial factor of 2 accounts for storing both Keys and Values, and the second factor of 2 is for BF16 bytes.

### Latency vs. Throughput Calculations
Given memory bandwidth $BW$ (bytes/sec) and a model with $P$ parameters:
- **Latency (seconds per token)**:
  $$\text{Latency} \approx \frac{2 \cdot P + \text{Memory}\_{\text{KV}}}{BW}$$
- **Throughput (tokens per second)**:
  $$\text{Throughput} \approx \frac{B}{\text{Latency}} = \frac{B \cdot BW}{2 \cdot P + \text{Memory}\_{\text{KV}}}$$

As batch size $B$ increases, parameter memory overhead is amortized, increasing throughput at the expense of generation latency.

### Speculative Decoding Acceptance Criterion
To preserve exact sampling from the target model distribution $q(x)$ given candidate tokens proposed by the draft model $p(x)$:
1. Draw sample $x$ from $p(x)$.
2. Accept $x$ with probability:
   $$\alpha = \min\left(1, \frac{q(x)}{p(x)}\right)$$
3. If rejected, sample $x$ from the residual distribution:
   $$r(x) = \max\left(0, q(x) - p(x)\right)$$

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### Speculative Decoding Engine Logic
1. **Draft Generation**: Generate $K$ tokens in series using the fast, small draft model: $x_1, \dots, x_K \sim p(x)$.
2. **Parallel Validation**: Run the large target model in parallel on the sequence $[x_1, \dots, x_K]$ to compute target probabilities $q(x_i | \dots)$.
3. **Acceptance Phase**: Loop $i$ from 1 to $K$:
   - Generate $u \sim U(0, 1)$.
   - If $u < \min(1, q(x_i)/p(x_i))$, accept $x_i$ and continue.
   - Else, reject $x_i$, sample a replacement token from $r(x_i)$, and discard the remaining candidates.

### PyTorch/Pythonic Blueprint (Educational Speculative Sampling Loop)
```python
import torch

# Educational speculative decoding step
def speculative_decode_step(draft_model, target_model, prompt_ids, K=4):
    # Generates K draft tokens sequentially
    seq = prompt_ids.clone()
    draft_probs = []
    
    for _ in range(K):
        with torch.no_grad():
            logits = draft_model(seq)
            next_prob = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(next_prob, num_samples=1)
            seq = torch.cat([seq, next_token], dim=-1)
            draft_probs.append(next_prob[0, next_token[0, 0]])

    # Parallel validation pass through the target model
    with torch.no_grad():
        target_logits = target_model(seq)
        target_probs = torch.softmax(target_logits[:, -K-1:-1, :], dim=-1)

    # Acceptance loop
    accepted_tokens = []
    for i in range(K):
        token = seq[0, -K + i]
        p_t = draft_probs[i]
        q_t = target_probs[0, i, token]
        
        alpha = min(1.0, (q_t / p_t).item())
        if torch.rand(1).item() < alpha:
            accepted_tokens.append(token)
        else:
            # Rejection: Sample from residual and exit
            residual = torch.clamp(target_probs[0, i, :] - draft_probs[i], min=0.0)
            residual /= residual.sum()
            res_token = torch.multinomial(residual, num_samples=1)
            accepted_tokens.append(res_token[0])
            break
            
    return torch.cat([prompt_ids[0], torch.tensor(accepted_tokens)])
```

## 4. Hardware Realities & Compute/Memory Accounting

### Continuous & Selective Batching
- **Jagged Sequences**: Live requests arrive at varying times and feature highly irregular context lengths. Grouping requests naively forces heavy padding, wasting memory on padding tokens.
- **Selective Batching**: To maintain compute density, the non-attention layers (MLPs, which constitute 2/3 of parameters) concatenate jagged sequences into a single contiguous 1D token vector. This bypasses the need to evaluate redundant padding elements.

### Fragmentation and PagedAttention
- **fragmentation**: Standard memory allocation reserves contiguous space for the maximum possible sequence length. Since most generations terminate early, up to 60-80% of reserved HBM is wasted due to internal and external fragmentation.
- **Paged Memory Layout**: PagedAttention divides the KV cache into fixed-size physical pages in memory, tracking block locations via a page table index. Non-contiguous physical chunks are resolved on-the-fly inside custom CUDA attention kernels, allowing identical prompt prefixes (e.g., system instructions) to be safely shared by multiple users without duplication.

## 5. Hyperparameter Heuristics & Optimizations

<div id="plotly-cs336-10-prefill-decode" class="plotly-chart" aria-label="Interactive Plotly chart: Decoding Latency vs Batch Size"></div>

<p><em>Figure: Autoregressive Decoding is strictly memory-bandwidth bound at small batch sizes; batching increases arithmetic intensity and throughput.</em></p>

<div id="plotly-cs336-10-speculative-decoding" class="plotly-chart" aria-label="Interactive Plotly chart: Speculative Decoding Speedup Factor"></div>

<p><em>Figure: Speculative Decoding achieves 2x–3x serving speedups when the draft model acceptance rate $\alpha$ exceeds 70%.</em></p>

- **MLA (Multi-Latent Attention) Parameters**: DeepSeek-V2 replaces standard multi-head attention with MLA. By projecting the keys and values down to a compressed latent space of $d_c = 512$ (compressing the keys and values from 16,384 dimensions), the physical KV cache size per layer scales with $d_c$ instead of $n \cdot d\_{\text{head}}$, allowing the serving batch size $B$ to scale by over **30x**.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **Underflow in Quantized Attention**: When serving models in FP8 or FP4 formats, attention dot products ($QK^T$) are highly prone to numerical underflow due to the restricted dynamic range of 3-bit or 2-bit mantissas. This is mitigated by isolating the attention softmax and normalizer steps in full FP16 or FP32 precision, quantizing only the dense projection matrices.

### Conceptual Reflection Questions
1. *Why does speculative decoding guarantee mathematically identical sampling outputs compared to standard target model decoding?*
   **Answer**: The acceptance probability $\alpha = \min(1, q(x)/p(x))$ combined with sampling from the residual distribution $r(x) = \max(0, q(x) - p(x))$ on rejection is a form of Rejection Sampling. By scaling the acceptance threshold precisely by the ratio of target-to-draft densities, the probability of any token sequence being output is shown mathematically to converge exactly to the target distribution $q(x)$, ensuring zero quality degradation.

2. *Why is GQA (Grouped Query Attention) highly superior to MQA (Multi-Query Attention) in practice?*
   **Answer**: Multi-Query Attention collapses all key and value heads down to a single head ($k\_{\text{heads}} = 1$), representing an extreme compression ratio. While this yields optimal memory bandwidth savings, it causes severe, irreversible hits to model expressiveness and accuracy. GQA groups multiple query heads into a moderate number of key-value head groups (e.g., 8 groups), establishing an adjustable trade-off that recovers nearly all the performance of MHA while retaining most of the systems gains of MQA.
