# CS336 Lecture 6: Kernels, Triton, XLA

## 0. Quick-Recall Summary
- **Thread Block as Scheduling Unit**: Threads are grouped into thread blocks (Concurrent Thread Arrays/CTAs) scheduled on a single Streaming Multiprocessor (SM), where they share SRAM memory to minimize slow HBM accesses.
- **The Warp and Lock-Step Execution**: GPUs execute threads in warps of 32 in strict lock-step; branching introduces control divergence where active paths execute sequentially while others mask and sit idle.
- **Memory Coalescing & Bank Conflicts**: DRAM is accessed in 128-byte cache line bursts; coalescing groups warp reads contiguously to maximize memory throughput, while bank conflicts serialize shared memory reads when multiple threads hit the same 4-byte wide bank.
- **Operator Fusion & Triton's Role**: Native PyTorch launches independent kernels for element-wise graphs (forcing costly HBM round-trips); `torch.compile` uses Triton to automatically fuse operations into a single kernel that reads and writes to HBM only once.
- **Tiled Matrix Multiplication (MatMul)**: To prevent a memory-bound bottleneck from $O(MKN)$ redundant global HBM reads, tiled MatMul schedules a C-tile per thread block, streaming sub-tiles of A and B into shared memory to accumulate dot products in registers.

---

## 1. Core Paradigm & Systems Overview
- **Objective**: The engineering goal of this lecture is to move from high-level PyTorch abstractions down to hardware-aware GPU programming. It details the physical layout of the GPU memory hierarchy and explains how writing custom kernels using the Triton programming language can minimize global data movement and drastically improve execution throughput.
- **Primary Bottleneck**: **Memory-bound execution**. While GPU floating-point operations (FLOPs) have scaled super-exponentially due to specialized hardware like Tensor Cores, memory bandwidth (DRAM/HBM to SRAM) is growing much slower, making data movement the primary latency bottleneck in deep learning systems.
- **Builds on**: This lecture directly builds on **Lecture 2 (Resource Accounting)** and **Lecture 5 (GPUs, TPUs, and FlashAttention)** by moving from theoretical roofline calculations into the mechanical and programmatic realities of writing, compiling, and profiling custom GPU kernels.

---

## 2. Theoretical & Mathematical Primitives

### GPU Memory Hierarchy & Latency Characteristics
GPU memory exhibits an inverse relationship between capacity and speed. Managing these levels programmatically is the core of performance engineering:
- **Registers**: Fastest and closest to execution. B200 has 65,536 registers per SM (total 256 KB per SM). Latency: ~10–20 cycles.
- **L1 Cache / Shared Memory (SRAM)**: Local to each SM, programmable (shared memory) or hardware-managed (L1). Latency: ~20–30 cycles.
- **L2 Cache**: Shared across the entire chip. Latency: ~100–200 cycles.
- **High Bandwidth Memory (HBM/DRAM)**: Off-chip global memory. B200 provides 8 TB/s of bandwidth. Latency: ~1000+ cycles (10x slower than L1/shared memory).

### Arithmetic Intensity and Operations Accounting
The **Arithmetic Intensity** ($I$) of an algorithm measures the ratio of floating-point operations performed to bytes of data moved:

$$I = \frac{\text{Flops}}{\text{Bytes Transferred}}$$

An accelerator has an native intensity threshold ($I\_{acc}$), calculated from its specifications:

$$I\_{acc} = \frac{\text{Peak Flops/sec}}{\text{Memory Bandwidth (Bytes/sec)}}$$

For an H100, $I\_{acc} \approx 295$ FLOPs/byte. If $I\_{algo} < I\_{acc}$, the operation is **memory-bound**; if $I\_{algo} > I\_{acc}$, it is **compute-bound**.

#### 1. Element-Wise ReLU
For a vector of size $N$ in 16-bit precision (2 bytes per element):
- **Bytes Transferred**: Read $X$ ($2N$ bytes) + Write $Y$ ($2N$ bytes) = $4N$ bytes.
- **Flops**: $N$ comparisons.
- **Arithmetic Intensity**:
  $$I\_{\text{ReLU}} = \frac{N}{4N} = 0.25 \text{ FLOPs/byte}$$
  Since $0.25 \ll 295$, ReLU is heavily memory-bound.

#### 2. Element-Wise Gated Linear Unit (GLU)
- **Bytes Transferred**: Read inputs ($2N$) + Write output ($2N$) = $4N$ bytes.
- **Flops**: Highly complex operations (sigmoid, tanh, multiplications), $\approx 20N$ flops.
- **Arithmetic Intensity**:
  $$I\_{\text{GLU}} = \frac{20N}{4N} = 5.0 \text{ FLOPs/byte}$$
  Despite doing 20x more compute than ReLU, $5.0 \ll 295$, meaning GLU remains bottlenecked by the same memory transport speed.

#### 3. Vector Dot Product
For two vectors of size $N$:
- **Bytes Transferred**: Read $X$ ($2N$) + Read $W$ ($2N$) + Write scalar output ($2$) $\approx 4N$ bytes.
- **Flops**: $N$ multiplications + $(N-1)$ additions = $2N - 1$ flops.
- **Arithmetic Intensity**:
  $$I\_{\text{Dot}} = \frac{2N}{4N} = 0.5 \text{ FLOPs/byte}$$
  Memory-bound.

#### 4. Matrix-Vector Multiplication
For $Y = W X$ where $W \in \mathbb{R}^{N \times N}$ and $X \in \mathbb{R}^N$:
- **Bytes Transferred**: Read $X$ ($2N$) + Read $W$ ($2N^2$) + Write $Y$ ($2N$) $\approx 2N^2$ bytes.
- **Flops**: $N$ dot products of size $N \approx 2N^2$ flops.
- **Arithmetic Intensity**:
  $$I\_{\text{Mat-Vec}} = \frac{2N^2}{2N^2} = 1.0 \text{ FLOPs/byte}$$
  Memory-bound (explains why decoding in LLM inference is memory-bandwidth bottlenecked).

#### 5. Matrix-Matrix Multiplication (MatMul)

<div id="plotly-cs336-6-matmul-intensity" class="plotly-chart" aria-label="Interactive Plotly chart: MatMul Arithmetic Intensity Boundary"></div>

<p><em>Figure: MatMul Arithmetic Intensity ($N/3$) crosses the H100 hardware boundary into the compute-bound regime at $N \approx 896$.</em></p>

<div id="plotly-cs336-6-triton-fusion" class="plotly-chart" aria-label="Interactive Plotly chart: Triton Operator Fusion Execution Latency"></div>

<p><em>Figure: Triton Block Fusion eliminates intermediate DRAM round-trips, achieving up to 4x lower latency than unfused PyTorch kernels.</em></p>

For square matrices $A, B \in \mathbb{R}^{N \times N}$:
- **Bytes Transferred (Naive)**: Read $A$ ($2N^2$) + Read $B$ ($2N^2$) + Write $C$ ($2N^2$) = $6N^2$ bytes.
- **Flops**: $N^2$ dot products of size $N = 2N^3$ flops.
- **Arithmetic Intensity (Idealized)**:
  $$I\_{\text{MatMul}} = \frac{2N^3}{6N^2} = \frac{N}{3} \text{ FLOPs/byte}$$
  As $N$ scales beyond $\approx 1000$, arithmetic intensity surpasses $I\_{acc}$, transitioning the workload into the **compute-bound** regime.

---

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### 1. Element-Wise GLU Activation Kernel
```python
import triton
import triton.language as tl

@triton.jit
def triton_glu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Determine block index along the 1D grid
    pid = tl.program_id(axis=0)
    
    # Calculate starting pointer offset for this block
    start_offset = pid * BLOCK_SIZE
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary bounds
    mask = offsets < num_elements
    
    # Assuming input is split into left (a) and right (b) halves for gating
    # e.g., GLU(x) = x_left * sigmoid(x_right)
    half_elements = num_elements // 2
    
    a_ptrs = x_ptr + offsets
    b_ptrs = x_ptr + half_elements + offsets
    
    # Load contiguous blocks from HBM into registers/SRAM
    x_a = tl.load(a_ptrs, mask=(offsets < half_elements))
    x_b = tl.load(b_ptrs, mask=(offsets < half_elements))
    
    # Compute fused element-wise gated activation
    sigmoid_b = 1.0 / (1.0 + tl.exp(-x_b.to(tl.float32)))
    y = x_a * sigmoid_b
    
    # Store back to HBM
    tl.store(y_ptr + offsets, y, mask=(offsets < half_elements))
```

### 2. Triton Softmax (Row Fits Block)
```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, n_cols, row_stride, BLOCK_SIZE: tl.constexpr):
    # Each thread block processes exactly one row
    row_idx = tl.program_id(axis=0)
    row_start_ptr = x_ptr + row_idx * row_stride
    
    # Create 1D column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load the entire row into shared memory / registers
    # Out-of-bounds columns are masked out with negative infinity
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Numerically stable softmax
    x_max = tl.max(x, axis=0)
    x_safe = x - x_max
    numerator = tl.exp(x_safe)
    denominator = tl.sum(numerator, axis=0)
    
    y = numerator / denominator
    
    # Write the output row back to HBM
    y_row_start_ptr = y_ptr + row_idx * row_stride
    tl.store(y_row_start_ptr + col_offsets, y, mask=mask)
```

### 3. Triton Row Sum (Row Does Not Fit Block)
```python
@triton.jit
def triton_row_sum_kernel(x_ptr, y_ptr, n_cols, row_stride, BLOCK_SIZE: tl.constexpr):
    # Thread block handles one row, but iterates over tiles of BLOCK_SIZE
    row_idx = tl.program_id(axis=0)
    row_start_ptr = x_ptr + row_idx * row_stride
    
    # Initialize a 1D vector accumulator in registers/SRAM
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over the column elements in chunks of BLOCK_SIZE
    for tile_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load sub-tile from HBM
        tile = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        accumulator += tile
        
    # Perform intra-block reduction across the threads
    row_sum = tl.sum(accumulator, axis=0)
    
    # Write the single scalar sum for this row to HBM
    tl.store(y_ptr + row_idx, row_sum)
```

### 4. Triton Tiled Matrix Multiplication (Conceptual Spec)
```python
@triton.jit
def triton_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    # Grid coordinates identify which C-tile this block handles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Initialize 2D index matrices for thread blocks
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    
    # Set up memory pointers for current blocks
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Register accumulator for block's outputs
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    
    # Loop through the contracting dimension K in steps of BK
    for k in range(0, K, BK):
        # Load tiles with boundary conditions masked
        a_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        b_tile = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        
        # Core fused hardware tensor core matrix dot product
        accumulator += tl.dot(a_tile, b_tile)
        
        # Advance pointers down contracting dimension
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
        
    # Fused element-wise activation (ReLU)
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)
    
    # Map final 2D pointers of C and store back to HBM
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
```

---

## 4. Hardware Realities & Compute/Memory Accounting

### Warp Occupancy and Register Accounting
A Streaming Multiprocessor (SM) has strict resource constraints. When registers per thread exceed boundaries, warp occupancy drops.
- **Example Workload**: A thread block has 128 threads. Each thread uses 160 registers.
- **Hardware Constraints (NVIDIA B200)**: Maximum registers per SM = 65,536. Maximum resident warps per SM = 64.
- **Analysis**:
  - Registers per block = $128 \text{ threads} \times 160 \text{ registers} = 20,480 \text{ registers}$.
  - Max concurrent blocks on SM = $\lfloor 65,536 / 20,480 \rfloor = 3 \text{ blocks}$.
  - Active threads running concurrently = $3 \text{ blocks} \times 128 \text{ threads/block} = 384 \text{ threads}$.
  - Equivalent active warps = $384 / 32 = 12 \text{ warps}$.
  - **Warp Occupancy Ratio**:
    $$\text{Occupancy} = \frac{12 \text{ warps}}{64 \text{ max warps}} = 18.75\% \text{ occupancy}$$
    *Note: Low warp occupancy limits the scheduler's ability to hide long memory latency.*

### Bank Conflicts in Shared Memory (SRAM)
Shared memory on an SM is divided into **32 independent banks**, each 4 bytes wide.
- **Hardware Constraint**: Each bank can only service *one* memory address access per clock cycle.
- **The Conflict**: If multiple threads in a warp attempt to access different addresses that map to the *same* bank, the hardware serializes these reads.
- **Worst-Case (32-Way Conflict)**: When 32 threads in a warp access a single vertical column of a matrix layout (where every row is spaced by exactly 32 banks), execution speed drops up to 32x.
- **Mitigation (Swizzling)**: Shuffling shared memory coordinates dynamically so that row indices and column indices map to staggered banks.

### Wave Quantization & The Tail Effect
Grid executions are scheduled in successive "waves" across the physical SMs of a chip.
- **Example Scenario**: NVIDIA A100 GPU has exactly **108 SMs**.
- **Tiling Dimension**: We run a MatMul using a tile size of $256 \times 128$.
- **Case 1 ($N = 1792$)**:
  - Tiles required: $\frac{1792}{256} \times \frac{1792}{128} = 7 \times 14 = 98$ tiles.
  - **Wave Scheduling**: All 98 tiles run simultaneously in a single wave across the 108 SMs. 10 SMs sit idle. Utilization is extremely high, and the run finishes in 1 cycle.
- **Case 2 ($N = 1793$)**:
  - Tiles required (rounded up): $\lceil \frac{1793}{256} \rceil \times \lceil \frac{1793}{128} \rceil = 8 \times 15 = 120$ tiles.
  - **Wave Scheduling**:
    - *Wave 1*: 108 tiles run concurrently on the 108 SMs.
    - *Wave 2*: The remaining **12 tiles** are scheduled. Only 12 SMs are active, while **96 SMs sit completely idle**.
  - **The Step Cost**: Bumping the matrix dimension by exactly *one* value ($1792 \to 1793$) results in a sudden ~2x increase in execution time due to the second fractional wave.

---

## 5. Visualization Blueprint

### Wave Quantization Step-Function (SM Idle Analysis)
- **Visualization Type**: Segmented Grid Heatmap / Stacked Timeline.
- **Data Fields & Encoding**:
  - **Y-axis**: SM Index (0 to 107 for A100).
  - **X-axis**: Execution Time (Wave 1 duration vs. Wave 2 duration).
  - **Color Gradient**: SM status (Green = Active Compute, Grey = Idle/Wasted silicon).
- **Interactive Controls**:
  - **Matrix Size Slider ($N$)**: Adjusts $N$ from $1024 \to 2048$. Bumping past multiples of tile sizes (e.g. $1792 \to 1793$) instantly renders Wave 2 on the timeline, highlighting the idle SM blocks and showing the sudden drop in Model FLOPs Utilization (MFU).

---

## 6. Empirical Scaling Laws & Hyperparameter Heuristics

### Matrix Sizing & Memory Coalescing Alignments
NVIDIA DRAM architecture loads data in contiguous chunks of **128 bytes** (one cache line).
- **Heuristic**: Matrix dimensions (especially inner contracting dimensions like hidden dimension $d$, and batch configurations) must be aligned with cache boundary math to prevent strided, uncoalesced memory calls.
- **Rule of Thumb**: Dimensions must always be divisible by **16 or 32**. Matrices that violate this alignment force double the DRAM transactions to fetch the same data, diluting effective memory bandwidth.

---

## 7. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas/Common Bugs
1. **The Asynchronous Benchmarking Illusion**: Benches timing PyTorch calls on GPU without executing a `torch.cuda.synchronize()` barrier will return artificially low, near-instant latency measurements. Because the CPU submits instructions asynchronously to the GPU queue without waiting for completion, timing calls measure queue insertion speed, not physical execution time.
2. **Naive Element-Wise Round-Trips**: Standard multi-operation lines in PyTorch (e.g., non-fused GELU/GLU) instantiate distinct intermediate tensors. Each operation forces a slow global HBM read/write round-trip. Fusing these into a single Triton/CUDA block-level operation prevents this bottleneck.

### Conceptual Reflection Questions
1. **In a sequence reduction operation (like Softmax), why does setting each row to a single Thread Block become unviable when row length $V$ exceeds the maximum registers/SRAM capacity? Describe the algorithmic shift required.**
   - *Answer*: When row length $V$ is larger than the SM's physical shared memory/register limits, we can no longer load the entire row atomically to perform thread reductions. We must shift to a **hierarchical block-tiling scheme**. Threads must loop iteratively over row sub-tiles, maintaining a local "running max" and "running accumulator" in registers. Finally, threads execute an intra-block tree reduction to combine the sub-tile aggregates, resolving global normalization dynamically.

2. **Recompute saves activation memory by recalculating forward states on the backward pass. For a chain of element-wise operators, under what physical hardware ratio does recomputation become a net latency win rather than a deficit?**
   - *Answer*: Recomputation is a latency win when the time to recalculate the operation's math is less than the time saved by avoiding global HBM memory bandwidth operations. If the hardware's accelerator intensity $I\_{acc}$ is very high (compute is extremely fast relative to memory transport), we are memory-bandwidth bound. Recalculating an element-wise activation costs almost zero raw execution cycles because the arithmetic registers can compute it faster than DRAM can transport the saved activation forward states.

3. **How does the physical difference in networking topology between TPUs (Toroidal Mesh) and GPUs (Fat Tree) impact their optimal sharding strategies for Mixture of Experts (MoE) vs. Dense Transformers?**
   - *Answer*: Toroidal Mesh networks (TPUs) excel at predictable, structured, neighbor-to-neighbor communication patterns, making them highly efficient for static tensor and pipeline parallel workloads. However, MoE routing requires dynamically dispatching tokens to distant experts, resulting in sparse, unstructured, all-to-all communication. A Fat Tree network (GPUs) provides robust, high-bandwidth all-to-all paths through spine switches, making them far better suited to handle the unpredictable routing congestion of expert parallelism.