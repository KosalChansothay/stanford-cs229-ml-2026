# CS336 Lecture 5: GPUs, TPUs, and FlashAttention

## 0. Quick-Recall Summary
*   **Memory Bandwidth Bottleneck**: Compute throughput scales exponentially while memory bandwidth grows slowly, making memory data transfer (HBM to SRAM) the primary bottleneck in modern LLM systems.
*   **SIMT & Control Divergence**: Threads execute in warps of 32 in lockstep. Branching conditionals (`if/else`) cause serialization (control divergence) as unselected branches must be masked out and run sequentially.
*   **Block-Scaled Quantization**: FP8 and FP4 use shared exponent scaling factors per block of elements (e.g., 1 scaling factor per 32 elements in FP8, per 16 in FP4). Transposition requires storing dual copies to avoid expensive runtime re-quantization.
*   **Tiling & Coalescing**: DRAM reads are burst-based (typically 128 bytes). Coalesced accesses from contiguous layouts prevent redundant memory transactions. Tiling loads submatrices into shared SRAM, reducing global memory accesses by a factor of tile size $T$.
*   **FlashAttention**: Tiling attention is enabled by the online softmax algorithm, which updates running softmax normalizers block-by-block, avoiding the $O(N^2)$ global memory footprint.

---

## 1. Core Paradigm & Systems Overview
*   **Objective**: The engineering goal of this lecture is to demystify the hardware execution model of GPUs and TPUs, analyze the hardware-software contract of parallel accelerators, and introduce critical low-level memory and compute tricks—culminating in the bottom-up design of FlashAttention to maximize Hardware FLOPs utilization.
*   **Primary Bottleneck**: Memory-bound / Communication-bound. While arithmetic logic units (tensor cores) are extremely fast, high bandwidth memory (HBM) is physical distances away on-chip and exhibits 10x higher latency (200-300 cycles) compared to registers and L1/SRAM (20-30 cycles). Thus, moving tensors in and out of global memory is the primary systems bottleneck.
*   **Builds on**: This lecture builds directly on Lecture 2's resource accounting and arithmetic intensity formulas, transitioning from abstract "back-of-the-envelope" FLOP calculations to physical hardware boundaries and custom compiler/kernel behaviors.

---

## 2. Theoretical & Mathematical Primitives

### Arithmetic Intensity & Roofline Boundary
The arithmetic intensity $I$ of an algorithm is defined as:

$$I = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$

If $I < I\_{\text{accelerator}} = \frac{\text{Peak FLOPs/sec}}{\text{Memory Bandwidth (Bytes/sec)}}$, the kernel is memory-bound; otherwise, it is compute-bound. For an NVIDIA H100, the accelerator intensity is:

$$I\_{\text{H100}} = \frac{1979 \times 10^{12} / 2 \text{ FLOPs/sec}}{3.3 \times 10^{12} \text{ Bytes/sec}} \approx 295 \text{ FLOPs/Byte} \quad [\sim104:00]$$

### Tiling Memory Access Reduction
In a naive $N \times N$ matrix multiplication, each input element is read from global memory $N$ times. With a square tile size $T$ loaded into shared SRAM, each element is read only $\frac{N}{T}$ times from global memory, achieving a $T$-times reduction:

$$\text{Global memory reads reduction factor} = T \quad [\sim371:00]$$

### Online Softmax Algorithm (FlashAttention Foundation)
Instead of a global softmax which requires materializing the entire matrix $x$:

$$m = \max_i x_i, \quad d = \sum_i e^{x_i - m}, \quad a_i = \frac{e^{x_i - m}}{d}$$

The online softmax updates running maximums and denominators block-by-block. For two blocks $A$ and $B$, let the local maximum of block $A$ be $m^{(A)}$ and normalizer sum be $d^{(A)}$, and block $B$ be $m^{(B)}$ and normalizer sum be $d^{(B)}$. The merged state is:

$$m^{\text{new}} = \max(m^{(A)}, m^{(B)}) \quad [\sim385:00, \sim386:00]$$

$$d^{\text{new}} = d^{(A)} \cdot e^{m^{(A)} - m^{\text{new}}} + d^{(B)} \cdot e^{m^{(B)} - m^{\text{new}}} \quad [\sim385:00, \sim386:00]$$

This mathematical reformulation allows computing softmax block-by-block without storing intermediate $N \times N$ matrices in HBM.

---

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### Algorithmic Logic
The tiled online softmax attention loop (FlashAttention style) operates as follows:
1. Divide the sequence indices of Query (Q), Key (K), and Value (V) into block chunks.
2. Load a block of Q, K, V into fast SRAM/shared memory.
3. Compute local query-key dot products $S = Q K^T$.
4. Compute local row maximums and update the scaling factor dynamically.
5. Adjust the running denominator sum and partial output matrix on the fly.
6. Repeat across tiles and write the final normalized output to HBM.

### PyTorch/Pythonic Blueprint
```python
# Illustrative Pythonic Blueprint for Online Softmax Accumulation
import torch

def online_softmax_step(prev_m, prev_d, prev_out, new_block):
    """
    Simulates the block-by-block state updates of online softmax.
    All inputs represent state for a single row across sequential tiles.
    """
    # 1. Compute local statistics for the incoming block
    local_m = torch.max(new_block)
    local_exp = torch.exp(new_block - local_m)
    local_d = torch.sum(local_exp)
    
    # 2. Merge states using the mathematical transformation
    new_m = torch.max(prev_m, local_m)
    
    # Scale previous sum and local sum to the new common maximum
    scaled_prev_d = prev_d * torch.exp(prev_m - new_m)
    scaled_local_d = local_d * torch.exp(local_m - new_m)
    new_d = scaled_prev_d + scaled_local_d
    
    # 3. Rescale the running output matrix to maintain mathematical equivalence
    rescale_factor = torch.exp(prev_m - new_m)
    new_out = prev_out * (scaled_prev_d / new_d) + (local_exp / new_d) # Simplified representation
    
    return new_m, new_d, new_out
```

---

## 4. Hardware Realities & Compute/Memory Accounting

### Memory Overhead & Bandwidth Hierarchy
*   **Registers**: Fastest, local to thread/SM. B200 has 65k registers per SM (total 256KB per SM).
*   **L1/Shared Memory**: Fast, on-SM. (20-30 cycles latency).
*   **L2 Cache**: Chip-wide, larger, slower than L1.
*   **HBM/Global Memory**: Slowest (200-300 cycles latency, 10x slower than L1). B200 has up to 192GB and 8 TB/sec memory bandwidth.

### Hardware Efficiency & MFU
*   **Model FLOPs Utilization (MFU)**: Actual FLOPs realized divided by peak promised FLOPs. A well-optimized transformer training run targets around 0.5 MFU (50% utilization).
*   **Operator Fusion**: Combines multiple pointwise operations (like GLU, GeLU, LayerNorm) into a single kernel to minimize HBM round-trips. Reduces runtime overhead from up to 25% to negligible figures.
*   **Wave Quantization**: Happens when the total number of blocks/tiles does not divide the number of physical SMs on the chip. For example, an NVIDIA A100 has 108 SMs. A matrix multiplication with 98 tiles fits in 1 wave (98 SMs busy, 10 idle). Adding 1 dimension increases the tile count to 120, requiring 2 waves of execution, leaving 96 SMs completely idle in the second wave, dropping performance dramatically.

---

## 5. Visualization Blueprint (conceptual spec)

<div id="plotly-cs336-5-flash-io" class="plotly-chart" aria-label="Interactive Plotly chart: FlashAttention SRAM Tiling HBM IO Complexity"></div>

<p><em>Figure: FlashAttention SRAM Tiling cuts HBM reads/writes from $O(N^2)$ to $O(N)$, overcoming the memory bandwidth bottleneck.</em></p>

<div id="plotly-cs336-5-quant-throughput" class="plotly-chart" aria-label="Interactive Plotly chart: Quantization Precision vs Tensor Core Compute Throughput"></div>

<p><em>Figure: Quantization Scaling — FP8 and FP4 double and quadruple Tensor Core throughput while slashing memory footprint.</em></p>


*   **Visualization type**: 2D Line Chart (Matrix size vs. realized throughput).
*   **Data fields & encoding**:
    *   **X-axis**: Matrix Dimension $N$ (linear scale, e.g., 1700 to 1850).
    *   **Y-axis**: Throughput in TeraFLOPs/sec.
    *   **Color-encoding**: Divisibility of dimension (e.g., Purple for divisible by 32, Red for 16, Blue for odd).
    *   **Highlights**: Annotations showing the dramatic Cliff-edge at $N=1793$ (wave quantization on A100's 108 SMs).
*   **Interactive controls**: Dropdowns to select different GPU architectures (A100: 108 SMs, H100: 132 SMs, B200: 160 SMs) and Sliders to adjust Tile size $T$ to show the shift in the cliff locations.

---

## 6. Empirical Scaling Laws & Hyperparameter Heuristics
*   **Divisibility/Padding**: Always pad matrices and vocabulary sizes to multiples of 16, 32, or 64. This ensures alignment with DRAM burst boundaries (128-byte cache lines) and allows coalesced memory reads, preventing high-penalty uncoalesced memory fetches. Divisibility also ensures efficient thread block scheduling without partial tile waste.

---

## 7. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas / Common Bugs
*   **Asynchronous CUDA Profiling Error**: Timing PyTorch operations without calling `torch.cuda.synchronize()` before and after measurement. PyTorch executes asynchronously, leading to artificially low (almost instant) timings.
*   **OOM via Activation Accumulation**: Standard attention stores an $N \times N$ attention matrix per layer per head, scaling quadratically. Using flash attention or activation checkpointing avoids this footprint.

### Conceptual Questions
1.  *Why does transposing a block-scaled FP8 or FP4 matrix require copying the matrix twice rather than simple index swapping?*
    *   **Answer**: In block-scaled formats like MXFP8, scale factors are shared across a contiguous block of elements (e.g., 1 scaling factor per 32 elements). If you transpose the matrix, the elements that originally shared a block are now scattered across different rows/columns, breaking the contiguous layout required by the scale factor blocks. To avoid the high overhead of dynamically re-quantizing the matrix during the backward pass, systems store two copies of the matrix: one quantized row-wise and one quantized column-wise.
2.  *Explain why a simple element-wise LayerNorm layer, representing less than 0.2% of a model's theoretical FLOPs, can consume up to 25% of the total wall-clock runtime.*
    *   **Answer**: LayerNorm is highly memory-bandwidth bound. It has a very low arithmetic intensity (few FLOPs per byte transferred). The GPU spending most of its time waiting for high bandwidth memory (HBM) transfers rather than executing calculations makes it run slowly. Since LayerNorm must read the entire input, subtract means, calculate variances, and write back, it saturates memory bandwidth, leading to low Model FLOPs Utilization (MFU).
3.  *Given a GPU with 108 SMs and a tile layout that yields 109 tiles, how much worse is the wave quantization penalty compared to a layout that yields 108 tiles?*
    *   **Answer**: With 108 tiles, all 108 SMs are perfectly utilized in a single wave, completing the work in 1 step. With 109 tiles, the GPU must schedule a second wave to process the single remaining tile. During this second wave, only 1 SM is active while the other 107 SMs sit completely idle, effectively doubling the runtime of the computation.
