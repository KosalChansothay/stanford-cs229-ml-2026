# CS336 Guest Lecture: Systems and Alternative Architectures (Dan Fu)

## 0. Quick-Recall Summary
- **Prefill vs. Decode Partitioning**: Disaggregates serving clusters into dedicated prefill nodes (compute-bound, parallelizable) and decode nodes (memory-bound, sequential).
- **Mega-Kernels**: Fuses multiple sequential operations (e.g., loading KV cache alongside QKV projections) into a single giant GPU kernel to eliminate launch latencies and SM tail idle periods.
- **ThunderKittens**: A hardware-aware low-level CUDA library (alternative to Triton) offering fine-grained register and warp coordination to push GPUs to near speed-of-light HBM bandwidth utilization.
- **Parse Recurrent depth**: A stabilized loop transformer architecture that routes activations through identical block recurrences, trading runtime FLOPS for higher accuracy without increasing parameter size.
- **SSM Stabilization Math**: Controls recurrent loss spikes in loop transformers by constraining recurrence state matrices to negative diagonal structures to enforce a spectral radius $< 1$.

## 1. Core Paradigm & Systems Overview
- **Objective**: Design alternative inference architectures (mega-kernels, looped recurrences) and custom hardware-level libraries to bypass the quadratic context limit and memory-bandwidth walls of standard attention decoding.
- **Primary Bottleneck**: Memory bandwidth and kernel launch latency. Traditional sequential kernel execution causes massive GPU multiprocessor idle phases between operations.
- **Builds on**: Attention Alternatives and FlashAttention (Lecture 4 & 5) — extends algorithmic associativity and online softmax steps to full-layer mega-fusions.

## 2. Theoretical & Mathematical Primitives

### Recurrent State Loop Transformer (Parse)
Standard transformers pass activations sequentially through distinct layers. Parse routes inputs $x$ repeatedly through a shared recurrent block.
Naively looping activations $x_t \leftarrow R(x\_{t-1})$ is highly unstable and triggers catastrophic loss spikes. Parse models the residual progression as a structured dynamic system:

$$x\_{t+1} = A x_t + B \cdot r(x_t)$$

Where:
- $r(x_t)$: The highly non-linear transformer/attention layer block (treated as a bounded perturbation).
- $A, B$: Parameter matrices governing residual feedback and injection scaling.

### SSM Stability Constraint
To guarantee mathematical stability and prevent the activations from blowing up to $10^{19}$ over $k$ iterations, Parse enforces:
1. **The $A$ Matrix**: Constrained to be a negative diagonal matrix:
   $$A = -\text{diag}(\lambda_1, \lambda_2, \dots, \lambda_D) \quad \text{with} \quad \lambda_d > 0$$
   This forces the powers of $A$ to decay asymptotically to zero.
2. **The $B$ Matrix**: Scaled with a strict linear normalization constraint.
This bounds the spectral radius of the system to strictly less than 1, ensuring absolute convergence and stable loss curves under high learning rates.

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### Mega-Kernel Parallel Execution Flow
Standard kernels schedule one operation (e.g., QKV projection) across all SMs, tear down, launch attention, and repeat, creating idle gaps. The **Mega-Kernel** schedules a single thread grid to manage the entire layer:
1. **Overlap Weights & Cache**: While computing the QKV projections, asynchronously pre-load the past KV cache from global HBM into shared registers.
2. **Pipelined Reduction**: As soon as local projection values are ready, stream them directly to the attention registers, completely bypassing HBM write-backs.
3. **Deferred Weight Loading**: Start loading the output projection weights ($W_O$) into local caches *before* the final attention reductions are fully completed.

### PyTorch/Pythonic Blueprint (Educational Parse Recurrent Loop)
```python
import torch

class ParseRecurrenceBlock(torch.nn.Module):
    def __init__(self, dim, layer_block):
        super().__init__()
        self.layer = layer_block
        # Initialize diagonal A with positive lambda values
        self.raw_lambda = torch.nn.Parameter(torch.rand(dim) * 0.1)
        self.B_scale = torch.nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x, num_loops=8):
        # Enforce negative diagonal constraint: A = -diag(exp(raw_lambda))
        A = -torch.exp(self.raw_lambda)
        
        current_state = x
        for _ in range(num_loops):
            # Compute bounded non-linear block perturbation
            residual = self.layer(current_state)
            
            # Apply linear normalization to B injection
            norm_B = torch.tanh(self.B_scale)
            
            # State update step
            current_state = A * current_state + norm_B * residual
            
        return current_state
```

## 4. Hardware Realities & Compute/Memory Accounting
- **Warp-Level ThunderKittens Optimizations**: On modern H100 GPUs, standard memory loading utilizes only 30-40% of the physical bus capacity because kernels execute with fragmented launch gaps. By virtualizing shared memory space across warps and using instruction-level mega-fusions, **ThunderKittens** mega-kernels achieve **72% High Bandwidth Memory (HBM) utilization**, operating at the physical speed of light of the GPU bus.

## 5. Hyperparameter Heuristics

<div id="plotly-cs336-guest-disaggregated" class="plotly-chart" aria-label="Interactive Plotly chart: Disaggregated Serving Cluster Throughput"></div>

<p><em>Figure: Disaggregated Serving splits prefill and decode onto dedicated clusters, multiplying throughput under high concurrency.</em></p>

<div id="plotly-cs336-guest-megakernel" class="plotly-chart" aria-label="Interactive Plotly chart: Mega-Kernel Execution Latency Breakdown"></div>

<p><em>Figure: Mega-Kernel Layer Fusion eliminates kernel launch overheads and keeps GPU shared memory and registers hot.</em></p>

- **Recurrence Scaling Law**: When training loop transformers, as the dataset token size $D$ scales, the optimal hyperparameter recipe dictates that the **number of recurrences $k$ must scale logarithmically** alongside parameters to maximize representation power.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **The Mega-Kernel Labor Cliff**: Mega-kernels are extremely labor-intensive to code, taking talented systems engineers months of assembly tuning. Since they are manually partitioned to fit specific SM and register configurations, a mega-kernel optimized for batch size 16 will fail or crash if batch size is changed to 17, requiring a complete rewrite.

### Conceptual Reflection Questions
1. *Why does separating prefill and decode phases into different physical GPU nodes drastically improve overall cluster efficiency?*
   **Answer**: Prefill is compute-bound (processing thousands of prompt tokens in parallel via dense MatMuls) and achieves maximum FLOP utilization. Decode is memory-bandwidth bound (sequential, single-token generations that idle the Tensor Cores while streaming parameters). Mixing them on the same GPU forces the scheduler to context-switch, starving prefill tasks. Separating them allows specialized hardware allocation (e.g., using GPUs for prefill and ultra-fast LPUs for decode).

2. *How does Parse trade-off inference latency vs. HBM parameter storage compared to standard deep transformers?*
   **Answer**: Parse keeps the total parameter size constant by looping activations through the same physical weights, keeping the HBM footprint tiny. This allows serving engines to allocate 3x more GPU memory to the KV cache, scaling concurrent batch sizes. However, because the activations must pass through the loop multiple times, the total FLOP count per token increases, resulting in higher sequential generation latency.
