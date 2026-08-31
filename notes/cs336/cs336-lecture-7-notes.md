# CS336 Lecture 7: Parallelism (Foundations)

## 0. Quick-Recall Summary
- **Collective Communication Primitives**: Standardized distributed templates (All-Reduce, All-Gather, Reduce-Scatter) form the math-hardware interface of distributed training, avoiding complex manual point-to-point routing.
- **DDP (Distributed Data Parallel)**: Datasets are sharded while parameters are fully replicated; gradients are synchronized after the backward pass using a single, monolithic, parameter-wise `all_reduce`.
- **Tensor Parallelism (Megatron-LM)**: Layers are sharded intra-node along weight dimensions. Column-parallel linear layers split weights along columns (all-gathering activations in the forward pass), while row-parallel linear layers split along rows (all-reducing partial results).
- **Pipeline Parallelism**: Model layers are sharded sequentially across devices. Micro-batches are used to overlap sequential dependencies and minimize the idle time of pipeline bubbles.
- **Hardware Interconnect Hierarchy**: Intra-node communication over NVLink/NVSwitch (~1.8 TB/s) is orders of magnitude faster than inter-node PCIe, InfiniBand, or Ethernet (~100 GB/s), dictating which parallelism is mapping to which physical link.

## 1. Core Paradigm & Systems Overview
- **Objective**: Distribute massive model computations and memory footprints across hundreds or thousands of GPUs (ranks) by sharding parameters, gradients, activations, and optimizer states, maximizing hardware utilization (MFU).
- **Primary Bottleneck**: Communication-bound. Distributed training is bottlenecked by the slow physical interconnect bandwidth when copying activations or synchronizing gradients across separate nodes.
- **Builds on**: Single-GPU kernel optimization (Lecture 5 & 6) — once individual device operations are highly fused and tiled, the global bottleneck shifts from local execution to distributed network communications.

## 2. Theoretical & Mathematical Primitives

### Collective Communication Duality
An All-Reduce operation is mathematically and systems-wise equivalent to a sequential composition of a Reduce-Scatter and an All-Gather:

$$\text{All-Reduce}(X) = \text{All-Gather}(\text{Reduce-Scatter}(X))$$

### Column & Row Tensor Parallelism
For a standard fully connected layer $Y = XW$, we can shard the weight matrix $W$ by column or by row across $M$ devices (ranks).

1. **Column Parallel**:
   - Split weight $W = [W_1, W_2, \dots, W_M]$ along columns.
   - Each rank $i$ computes a slice of the activation:
     $$Y_i = X W_i$$
   - To reconstruct the full $Y$, we execute an **All-Gather** primitive on the activations:
     $$Y = \text{All-Gather}(Y_1, Y_2, \dots, Y_M)$$

2. **Row Parallel**:
   - Split weight $W$ along rows: $W = [W_1^T, W_2^T, \dots, W_M^T]^T$ and shard input $X = [X_1, X_2, \dots, X_M]$.
   - Each rank computes a partial product:
     $$Z_i = X_i W_i$$
   - The full output $Y$ is reconstructed by an **All-Reduce (Sum)** primitive:
     $$Y = \text{All-Reduce}(\sum\_{i=1}^M Z_i)$$

### Duality of Forward and Backward Passes
In tensor parallelism, the forward and backward communication operators are duals:
- **Column Parallel**: Forward = $\text{All-Gather}(Y_i)$; Backward = $\text{Reduce-Scatter}(\nabla\_{Y_i})$
- **Row Parallel**: Forward = $\text{All-Reduce}(Z_i)$; Backward = $\text{Identity}$ (split gradients)

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### Distributed Data Parallel (DDP) Logic
1. Shard input batch along the batch dimension: each rank $i$ gets a local slice $X\_{\text{local}, i}$.
2. Perform forward and backward passes locally to compute parameters and their gradients $\nabla_W^{(i)}$.
3. Average gradients globally across all ranks via an All-Reduce sum:
   $$\nabla_W = \frac{1}{M} \text{All-Reduce}\left(\sum\_{i=1}^M \nabla_W^{(i)}\right)$$
4. Update local replicas of weights using the synchronized averaged gradients: $W^{(i)} \leftarrow \text{Optimizer}(W^{(i)}, \nabla_W)$.

### PyTorch/Pythonic Blueprint (Educational DDP, Column & Row Parallelism)
```python
import torch
import torch.distributed as dist

# 1. Distributed Initialization Blueprint (Glue/NCCL backend)
def init_distributed(rank, world_size, backend="gloo"):
    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )

# 2. Educational Data Parallel (DDP) Step from Scratch
class MinimalDDPStep:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def step(self, local_x, local_y):
        self.optimizer.zero_grad()
        loss = self.model(local_x) - local_y
        loss.backward()
        
        # Manually All-Reduce and average gradients across ranks
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()
        
        self.optimizer.step()

# 3. Column-Parallel Linear Layer from Scratch
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.local_out_features = out_features // world_size
        
        # Local weight slice: (in_features, out_features/world_size)
        self.weight = torch.nn.Parameter(
            torch.randn(in_features, self.local_out_features)
        )

    def forward(self, x):
        # Local matrix multiply: Y_i = X * W_i
        local_y = torch.matmul(x, self.weight)
        
        # All-Gather to reconstruct the full output activation space
        gather_list = [torch.empty_like(local_y) for _ in range(self.world_size)]
        dist.all_gather(gather_list, local_y)
        
        # Concatenate along the feature dimension
        return torch.cat(gather_list, dim=-1)
```

## 4. Hardware Realities & Compute/Memory Accounting

### Interconnect Performance Scaling
- **NVLink (Intra-node)**: Modern generation (NVLink 5) achieves **1.8 TB/s** of bidirectional bandwidth, connecting the 8 GPUs on a single tray with near-memory-speed capability.
- **Infiniband / RDMA (Inter-node)**: Delivers around **100 GB/s to 200 GB/s** of bandwidth. By utilizing **Remote Direct Memory Access (RDMA)**, GPUs bypass CPU socket buffers, directly reading/writing to remote HBM over Infiniband to drastically minimize protocol latency.
- **Ethernet (Standard)**: Reaches only ~10-40 GB/s and requires socket copies through CPU DRAM, presenting a major bottleneck for distributed scaling unless bypassed via ROCE (RDMA over Converged Ethernet).

### Memory Communication Costs
For an All-Reduce synchronization on $N$ parameters across $M$ ranks:
- **Total volume of data sent/received per rank**:
  $$\text{Data Transferred} = 2 \times \left(\frac{M - 1}{M}\right) \times N \times \text{bytes-per-element}$$
- This scales flatly with $M$ as $M \to \infty$, allowing scaling to massive clusters without blowing up communication volumes per node.

## 5. Empirical Scaling Constraints & Topology Routing

<div id="plotly-cs336-7-ring-allreduce" class="plotly-chart" aria-label="Interactive Plotly chart: Ring AllReduce Communication Volume"></div>

<p><em>Figure: Ring AllReduce scales communication volume independently of cluster size, asymptotically bounding traffic at $2M$.</em></p>

<div id="plotly-cs336-7-tensor-parallel" class="plotly-chart" aria-label="Interactive Plotly chart: Tensor Parallelism Speedup NVLink vs PCIe"></div>

<p><em>Figure: Megatron Tensor Parallelism speedup requires intra-node NVLink bandwidth (900 GB/s); PCIe interconnects hit severe communication bottlenecks.</em></p>

- **Tensor Parallelism (TP)** is highly sensitive to latency because it performs communications multiple times *within* a single transformer layer. Therefore, TP must be restricted to ranks connected via high-speed **NVLink** (usually TP $\le 8$, matching a single node).
- **Pipeline Parallelism (PP)** handles point-to-point boundary activations and can tolerate the slower, high-latency inter-node connections (Infiniband/Ethernet) across nodes.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Silent Bugs
- **Missing CUDA Synchronize in Profiling**: When benchmarking, failing to insert `torch.cuda.synchronize()` before starting and stopping timers leads to measuring the asynchronous launch time rather than actual execution time, making communications or kernels appear near-instantaneous.
- **Distributed Deadlocks**: Inserting `dist.barrier()` calls unevenly across conditional branches will cause ranks that bypass the branch to hang indefinitely, blocking the entire cluster.

### Conceptual Reflection Questions
1. *Why does Tensor Parallelism require a ring/tree All-Reduce in the backward pass of a Row Parallel layer?*
   **Answer**: In a Row Parallel layer, the weight matrix is split along rows: $Y = X_1 W_1 + X_2 W_2$. To compute the gradients with respect to the input $\nabla_X$, each rank $i$ needs the full gradient $\nabla_Y$. Since the forward pass of Row Parallel uses an All-Reduce to sum up partial outputs, its mathematical dual dictates that the backward pass must replicate $\nabla_Y$ to all ranks to compute the local gradient slices asynchronously without further synchronization.

2. *Why does Pipeline Parallelism require micro-batching to be mathematically viable for hardware scaling?*
   **Answer**: Without micro-batching, if a model is sharded sequentially across 4 GPUs, Rank 1 computes layer 1 and then sits idle waiting for Ranks 2, 3, and 4 to finish their forward and backward passes. This results in a pipeline bubble size of $1 - 1/M$, where $3/4$ of the hardware is permanently idle. Micro-batching breaks the batch $B$ into $K$ micro-batches, allowing Rank 1 to immediately start on micro-batch 2 while Rank 2 processes micro-batch 1, filling the pipeline and reducing the bubble to a fraction of the total execution time.
