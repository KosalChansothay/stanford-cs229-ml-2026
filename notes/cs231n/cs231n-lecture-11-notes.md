# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 11: Large Scale Distributed Training

#### 0. Quick-Recall Summary
*   **The Hardware-Communication Gap:** While single-device GPU mixed-precision throughput has surged by $1000\times$ over the last decade (from 5 TFLOPs on K40 to 5000 TFLOPs on B200), inter-device communication scaling has lagged behind, creating a critical bottleneck in scaling distributed training.
*   **Fully Sharded Data Parallelism (FSDP):** To bypass single-GPU memory limits, FSDP shards model weights, gradients, and optimizer states across devices. It uses transient, all-gather broadcasts on-the-fly during the forward pass and reduces-scatters gradients during the backward pass, deleting parameters immediately after execution to keep memory footprints constant.
*   **Optimal Activation Checkpointing:** Retaining all activation tensors in memory scales linearly ($O(N)$), causing out-of-memory errors for deep nets. Recomputing activations during backpropagation reduces memory to $O(1)$ at the cost of quadratic ($O(N^2)$) computation. Selecting checkpoint spacing at $C = \sqrt{N}$ yields the optimal compromise: $O(\sqrt{N})$ memory with $O(N\sqrt{N})$ compute.
*   **The Megatron-LM Block Multiplex:** Tensor Parallelism (TP) shards individual weight matrices inside a Transformer's 2-layer MLP block. By sharding the first layer column-wise and the second layer row-wise, the spatial block matrix multiplications naturally align, requiring only a single All-Reduce operation at the very end of the block.
*   **Model FLOPs Utilization (MFU):** MFU is the ultimate guiding metric for training efficiency, measuring the fraction of peak theoretical hardware FLOPs actively spent on forward/backward math rather than idling in networking, caching, or data loading bottlenecks.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master the algorithmic paradigms and hardware constraints underpinning large-scale neural network training. It details how to partition massive parameters, activations, and sequences across multi-node clusters of tens of thousands of GPUs, transitioning from single-device data parallel execution to highly complex 4D hybrid parallel topologies.
*   **Lecture Category:** (e) Systems / Scaling.
*   **Builds on:** Lecture 6 (CNN Architectures and GPU training memory bottlenecks) and Lecture 8 (Attention and Transformers, illustrating sequence dimensions and MLP blocks).

---

#### 2. Mathematical Foundations

##### Gradient Linearity & Synchronous DP Updates
In synchronous Data Parallelism (DP), the total loss $L$ over a macro-batch composed of $M$ GPUs with a local batch size of $N$ is the average of individual sample losses:
$$L = \frac{1}{M \cdot N} \sum\_{g=1}^{M} \sum\_{i=1}^{N} \mathcal{L}\left(f(X\_{i,g}; W), y\_{i,g}\right)$$

By the linearity of the gradient operator, the global weight update gradient is computed independently in local blocks and averaged:
$$\nabla_W L = \frac{1}{M} \sum\_{g=1}^{M} \left( \frac{1}{N} \sum\_{i=1}^{N} \nabla_W \mathcal{L}\left(f(X\_{i,g}; W), y\_{i,g}\right) \right)$$
This proves that sharded data parallel training is mathematically equivalent to single-device training on the full macro-batch, without approximation.

##### Sharded Memory Footprint Math (Standard Adam Optimizer)
For a model with $P$ parameters trained using mixed-precision FP16/BF16 weights and standard FP32 Adam optimizer states, the memory footprint is calculated as follows:
*   **Parameters ($W$):** $2P$ bytes (FP16/BF16 representation)
*   **Gradients ($\nabla W$):** $2P$ bytes (FP16/BF16 representation)
*   **Optimizer States (Adam):** 
    *   *Master Weights (FP32):* $4P$ bytes (for numerical update stability)
    *   *First Moment / Momentum vector $m$ (FP32):* $4P$ bytes
    *   *Second Moment / Variance vector $v$ (FP32):* $4P$ bytes

$$\text{Total Memory per Parameter} = 2P + 2P + (4P + 4P + 4P) = 16P \text{ bytes}$$
Under a model sharding scheme (FSDP) distributed across $M$ GPUs, the local device memory overhead scales as:
$$\text{Local Memory}\_{\text{FSDP}} = \frac{16P}{M} \text{ bytes}$$

##### Mathematical Optimization of Activation Checkpointing
Let $N$ be the number of layers in a neural network.
1.  **Standard Backpropagation:**
    $$\text{Memory} = O(N) \quad \text{Compute} = O(N)$$
2.  **Activation Recomputation (Checkpointing every $C$ layers):**
    For a block size of $C$, we retain exactly one activation checkpoint every $C$ layers. To perform backward passes inside a block of size $C$, we run local forward passes on-demand:
    $$\text{Memory}(N, C) = \underbrace{O\left(\frac{N}{C}\right)}\_{\text{Stored Checkpoints}} + \underbrace{O(C)}\_{\text{Intra-block Forward Activations}}$$
    $$\text{Compute}(N, C) = \underbrace{O(N)}\_{\text{Initial Forward}} + \underbrace{O\left(\frac{N^2}{C}\right)}\_{\text{Quadratic Block-wise Recomputation}}$$
    To find the optimal checkpoint interval $C^*$ that minimizes memory, we solve:
    $$\frac{d}{dC} \left( \frac{N}{C} + C \right) = 0 \implies -\frac{N}{C^2} + 1 = 0 \implies C^* = \sqrt{N}$$
    Substituting $C^* = \sqrt{N}$ back into the memory and compute bounds yields:
    $$\text{Optimal Memory} = O(\sqrt{N}) \quad \text{Compute} = O(N\sqrt{N})$$

##### Megatron-LM Block Matrix Parallelism (Tensor Parallelism)
Inside a Transformer's multi-layer perceptron (MLP) block, the data tensor $X$ undergoes two successive linear projection layers with weights $W_1$ and $W_2$ and activation function $\sigma$:
$$\text{Output} = \sigma(X W_1) W_2$$

To parallelize this across $M$ GPUs without communication between layers:
1.  **First Layer (Column-Parallel sharding of $W_1$):**
    We split $W_1$ vertically into column slices: $W_1 = [W\_{1,1}, W\_{1,2}, \dots, W\_{1,M}]$. Each GPU $i$ holds slice $W\_{1,i}$ and computes:
    $$Y_i = \sigma(X W\_{1,i})$$
2.  **Second Layer (Row-Parallel sharding of $W_2$):**
    We split $W_2$ horizontally into row slices: $W_2 = [W\_{2,1}^T, W\_{2,2}^T, \dots, W\_{2,M}^T]^T$. Each GPU $i$ holds $W\_{2,i}$ and computes a partial block matrix product locally:
    $$Z_i = Y_i W\_{2,i}$$
3.  **Fusing Output (Mathematical Identity):**
    Using the algebraic identity of block matrix multiplication:
    $$\text{Output} = \sum\_{i=1}^{M} Z_i = \sum\_{i=1}^{M} \sigma(X W\_{1,i}) W\_{2,i}$$
    Thus, GPUs only perform an **All-Reduce sum** at the very end of the second linear layer, completely bypassing any synchronization step in between.

---

#### 3. Architecture / Algorithm Walkthrough

##### Algorithmic Logic of FSDP Weight-Sharding
Unlike standard data parallelism where every GPU holds a full replica of the weights, FSDP dynamically coordinates weight state broadcast and deletion:

```
[GPU 1 (Owner W1)] ──(Broadcast W1)──> [GPU 2] ──> (Both Compute Layer 1 Forward)
                                                     │
                                                     ▼
                                      [GPU 2 Deletes W1 Memory Replica]
                                                     │
                                                     ▼
[GPU 2 (Owner W2)] ──(Broadcast W2)──> [GPU 1] ──> (Both Compute Layer 2 Forward)
```

##### PyTorch Blueprint (Illustrative Column/Row-Sharded MLP)
This minimal skeleton represents the Megatron-LM tensor parallelism approach for sharding Linear layers:

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ShardedMLPBlock(nn.Module):
    """
    Illustrative blueprint modeling Megatron-LM Tensor Parallelism.
    Layer 1 is sharded column-wise; Layer 2 is sharded row-wise.
    """
    def __init__(self, in_features, hidden_features, out_features, rank, world_size):
        super(ShardedMLPBlock, self).__init__()
        self.rank = rank
        self.world_size = world_size
        
        # Column sharding: Each GPU gets a slice of hidden_features output columns
        assert hidden_features % world_size == 0
        self.sharded_hidden_dim = hidden_features // world_size
        
        # Local linear weights for layer 1 (Column-sharded)
        # Input shape: [Batch, Sequence, In_Features]
        # Output shape: [Batch, Sequence, Sharded_Hidden_Dim]
        self.col_linear = nn.Linear(in_features, self.sharded_hidden_dim, bias=False)
        self.activation = nn.ReLU()
        
        # Row sharding: Each GPU gets a slice of hidden_features input rows
        # Input shape: [Batch, Sequence, Sharded_Hidden_Dim]
        # Output shape: [Batch, Sequence, Out_Features]
        self.row_linear = nn.Linear(self.sharded_hidden_dim, out_features, bias=False)
        
    def forward(self, x):
        # 1. Forward Layer 1 (Trivially parallel over Columns; no comms)
        hidden_states = self.activation(self.col_linear(x))
        
        # 2. Forward Layer 2 (Local matrix multiplication over Rows; no comms)
        partial_output = self.row_linear(hidden_states)
        
        # 3. Synchronize outputs via All-Reduce sum to yield complete output block
        # Fuses all local sharded matrices back into the final representation
        dist.all_reduce(partial_output, op=dist.ReduceOp.SUM)
        
        return partial_output
```

---

#### 4. Visual Intuition & Interpretability

##### GPU Memory Hierarchy
*   **The Latency-Capacity Trade-off:** Standard training loops stall when moving data across physical barriers. 
    *   *High Bandwidth Memory (HBM)* holds the massive parameters ($80\text{ GB}$ at $\approx 3\text{ TB/s}$ bandwidth) but sits far from the compute units.
    *   *L2 Cache* ($50\text{ MB}$ at intermediate bandwidth) buffers tensors between computational steps.
    *   *L1 Cache & Registers* ($256\text{ KB}$) provide maximum execution throughput but are severely capacity-constrained.
    *   *Software Rule:* High-performance custom CUDA kernels maximize data reuse in L1 cache, avoiding round-trips to high-latency HBM.

##### Pipeline Parallelism (PP) "Bubble" Dynamics
*   **Idle GPU Bubbles:** Distributing layers sequentially across devices (e.g. GPU 1 gets Layers 1-10, GPU 2 gets Layers 11-20) introduces severe stall delays because GPU $K+1$ must wait for GPU $K$ to complete its forward pass.
*   **Micro-batch Interleaving:** Injecting multiple independent micro-batches concurrently into the pipeline (the 1F1B schedule) minimizes this idle "bubble" time. Each GPU alternates between computing a forward step for micro-batch $B_i$ and a backward step for micro-batch $B\_{i-k}$, forcing high overall cluster utilization.

```
Naive PP (No micro-batches):
GPU 4:                [F4][B4]
GPU 3:            [F3]    [B3]
GPU 2:        [F2]            [B2]
GPU 1:    [F1]                    [B1]
        |<────────── Idle Bubble ─────────>|

1F1B PP (Interleaved micro-batches 1, 2, 3, 4):
GPU 4:             [F1][F2][F3][F4][B1][B2][B3][B4]
GPU 3:         [F1][F2][F3][B1][F4][B2][B3][B4]
GPU 2:     [F1][F2][B1][F3][B2][F4][B3][B4]
GPU 1: [F1][B1][F2][B2][F3][B3][F4][B4]
        (Idle bubble space is significantly minimized)
```

---


<div id="plotly-cs231n-11-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 11 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)
To guide distributed system engineering, we propose an interactive **distributed Parallelism and Bottleneck Analyzer**:

*   **Visualization Type:** 2D Parameter-Activation Grid with dynamic communication network edge rendering.
*   **Data Fields & Encoding:**
    *   **Node Representation:** Individual GPUs (represented as circles, color-coded by memory usage: Green = Low, Red = Out-of-Memory threshold).
    *   **Edge Representation:** Inter-GPU communication lanes (NVLink intra-server vs. InfiniBand inter-server), where edge width maps to bandwidth speed (e.g., $900\text{ GB/s}$ vs $50\text{ GB/s}$) and edge animation speed maps to communication frequency.
*   **Interactive Controls:**
    *   **Parallelism Slider Matrix:** Controls the allocation of resources across the 4D parallelism axes: Data Parallelism (DP/FSDP), Tensor Parallelism (TP), Pipeline Parallelism (PP), and Context Parallelism (CP).
    *   **Activation Checkpointing Toggle:** Instantly recalculates and renders the change in active activation memory vs. peak computation FLOPs, visualizing the $O(\sqrt{N})$ scaling trade-off.
    *   **Sequence Length Dial:** Scales the input sequence length, showing the quadratic memory explosion in the attention matrix and the exact transition point where Context Parallelism must be enabled.

---

#### 6. Empirical Design Heuristics & Benchmark Results

##### H100 GPU Hardware Thresholds & Matrix Sizes
*   **Matrix Dimensions for Maximum Tensor Core Throughput:** Tensor cores achieve maximum hardware flops utilization (HFU) of $\approx 80\%$ on an H100 only when matrix shapes are sufficiently large (e.g., $8000 \times 8000$ dimensions or higher) to hide L2-to-HBM memory access latencies. Under small matrix sizes (e.g., $<512$), throughput drops precipitously as the compute units are starved for data.

##### State-of-the-Art MFU Baselines
*   **Benchmark Standards:** Model FLOPs Utilization (MFU) measures the efficiency of the full distributed stack.
    *   *MFU < 30%:* Indicates severe training bottlenecks (e.g., slow host-to-device communication, unoverlapped pipeline bubbles, or CPU data-loading bottlenecks).
    *   *MFU > 30%:* Standard target for highly optimized distributed code.
    *   *MFU > 40%:* Excellent/State-of-the-Art efficiency.
    *   *Llama 3 405B:* Reached $\approx 38\text{--}43\%$ MFU on $16,000$ H100 GPUs using a highly optimized 4D parallel layout.

##### Hybrid 4D Parallelism Topology Recipes
As models scale, researchers chain multiple sharding strategies to match cluster physical layouts:
1.  **Data Parallelism (DP):** Scalable up to 128 GPUs / 1B parameters.
2.  **FSDP + Activation Checkpointing:** Crucial for models between 1B and 10B parameters.
3.  **HSDP (FSDP + DP):** Essential above 256/512 GPUs to group sharding within high-bandwidth servers and use standard data parallelism across slower network rack routers.
4.  **4D Hybrid Parallelism:** For extreme models ($>100\text{B}$ parameters, sequence length $>100,000$), combine all four axes simultaneously (e.g., Llama 3 405B configuration):
    $$\text{Total GPUs } (16,384) = \underbrace{8}\_{\text{Tensor Parallel}} \times \underbrace{16}\_{\text{Context Parallel}} \times \underbrace{16}\_{\text{Pipeline Parallel}} \times \underbrace{8}\_{\text{Data Parallel}}$$

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas
*   **The Redundant Data Loader Pitfall:** A common distributed bug is failing to shard the data loader across GPUs. If every GPU accidentally loads the exact same mini-batch of data, the model processes identical samples on different devices, wasting compute and skewing batch normalization statistics.
*   **The FP32 Fallback Slowdown:** If a PyTorch module fails to cast weight or input variables into mixed-precision FP16/BF16 types, the tensors are routed to standard FP32 cores instead of the dedicated tensor cores. This results in an immediate, silent $20\times$ drop in computational throughput.
*   **The "Unsharding" Memory Spike:** During the FSDP forward pass, weight matrices are broadcast temporarily. If layer sizes are chosen arbitrarily (leading to uneven division among GPUs), memory allocations spike unevenly, triggering unpredictable out-of-memory crashes on individual nodes.

##### Graduate-Level Reflection Questions
1.  **The Compute-to-Communication Paradox:** When transitioning from A100 to H100 hardware, peak theoretical tensor core compute throughput increased by $\approx 3\times$ while HBM memory bandwidth only increased by $\approx 2\times$. Why does this hardware asymmetry lead to lower Model FLOPs Utilization (MFU) on newer GPUs, and what software strategies can counteract this communication-to-compute gap?
2.  **Evaluating the Limits of Asynchronous SGD:** Asynchronous Data Parallelism allows multiple GPU nodes to compute gradients independently and update weights without waiting for cluster-wide synchronization. Why does this approach degrade optimization convergence bounds in very large language and vision models, and why does the ML community predominantly favor synchronous execution?
3.  **Decoupling the 4D Parallelism Matrix:** Under a hybrid sharding scheme (e.g., Llama 3 405B), why must Tensor Parallelism (TP) be mapped strictly within the same physical node (utilizing NVLink) while Pipeline Parallelism (PP) can be mapped across separate racks? Trace the communication complexity of an attention weight-sharding layer vs. a layer-sharding boundary to justify your answer.
