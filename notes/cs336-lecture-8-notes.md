# CS336 Lecture 8: Parallelism (Advanced)

## 0. Quick-Recall Summary
- **ZeRO-3 / FSDP Memory Sharding**: Shards parameters, gradients, and optimizer states across ranks, turning linear $O(M)$ parameter storage per rank into a highly efficient $O(1/M)$ sharded model footprint.
- **Activation Memory Bottleneck**: For long context training, activation memory (scaling with sequence length as $34 \cdot S \cdot B \cdot H$) quickly dwarfs parameter storage, necessitating active mitigation.
- **Sequence Parallelism**: Solves the activation duplication bottleneck in Megatron-LM by sharding the pointwise layer norms and dropouts along the sequence dimension ($S$), achieving true $O(1/T)$ memory scaling.
- **Expert Parallelism**: Distributes Mixture-of-Expert (MoE) FFNs across ranks. Rather than dense collectives, it relies on high-throughput sparse `all_to_all` point-to-point communication.
- **4D Parallelism Topology**: Combines Data Parallelism (DP), Pipeline Parallelism (PP), Tensor Parallelism (TP), and Expert Parallelism (EP) to optimally map sharding schemes to hardware boundaries.

## 1. Core Paradigm & Systems Overview
- **Objective**: Enable the training of trillions of parameters on massive distributed clusters by sharding the static memory (parameters, gradients, optimizer states) and dynamic memory (activations), while overlapping communications with computations to eliminate cluster latency.
- **Primary Bottleneck**: Inter-node network saturation and pipeline bubbles. Splitting models globally forces constant synchronization; without deep overlapping and bubble-free schedules, GPUs sit idle waiting for network packets.
- **Builds on**: Foundational distributed primitives (Lecture 7) — uses All-Gather and Reduce-Scatter as the underlying transport engines to construct the dynamic parameter gathering of FSDP.

## 2. Theoretical & Mathematical Primitives

### Memory Allocation Math (AdamW Optimizer)
For a model with $N$ parameters trained under standard mixed precision (BF16/FP32 AdamW):
- **Parameters (BF16)**: $2N$ bytes
- **Gradients (BF16)**: $2N$ bytes
- **Optimizer Master Weights (FP32)**: $4N$ bytes
- **Optimizer First Moment / Momentum (FP32)**: $4N$ bytes
- **Optimizer Second Moment / Variance (FP32)**: $4N$ bytes
- **Total Static Footprint**: $16N$ bytes

Under **ZeRO / FSDP Sharding** with $M$ ranks:
- **ZeRO Stage 1** (Shard Optimizer States): 
  $$\text{Memory per rank} = 4N + \frac{12N}{M} \text{ bytes}$$
- **ZeRO Stage 2** (Shard States + Gradients): 
  $$\text{Memory per rank} = 2N + \frac{14N}{M} \text{ bytes}$$
- **ZeRO Stage 3 / FSDP** (Shard States + Gradients + Parameters): 
  $$\text{Memory per rank} = \frac{16N}{M} \text{ bytes}$$

### Activation Memory Accounting (Transformer Layer)
The total activation memory overhead per transformer layer (excluding attention softmax matrices) is modeled as:

$$A = 34 \cdot S \cdot B \cdot H + 5 \cdot \left(\frac{a \cdot S^2}{H}\right) \text{ bytes}$$

Where:
- $S$: Sequence length (tokens)
- $B$: Batch size
- $H$: Model hidden dimension
- $a$: Number of attention heads

1. **Megatron-LM Tensor Parallelism (TP)**: Only reduces the MLP and attention projection activations (the first term) by the TP size $T$. Pointwise layer norms and dropouts ($10 \cdot S \cdot B \cdot H$ bytes) remain duplicated across all $T$ ranks:
   $$A\_{\text{TP}} = \frac{24 \cdot S \cdot B \cdot H}{T} + 10 \cdot S \cdot B \cdot H \text{ bytes}$$

2. **Sequence Parallelism (SP) + TP**: Shards the pointwise layer norm and dropout activations along the sequence dimension ($S$), unlocking full linear memory reduction:
   $$A\_{\text{TP+SP}} = \frac{34 \cdot S \cdot B \cdot H}{T} \text{ bytes}$$

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### Fully Sharded Data Parallel (FSDP / ZeRO-3) Step
1. **Initialize**: Shard the model weights $W$ uniformly across $M$ ranks. Each rank $i$ stores $W\_{\text{sharded}, i} = \frac{W}{M}$.
2. **Forward Pass Loop (Layer $L = 1 \dots L\_{\text{max}}$)**:
   - Rank $i$ performs an **All-Gather** to materialize the full layer weights $W_L$ in local registers/SRAM.
   - Compute forward activation $Y_L = f(X\_{L-1}, W_L)$.
   - **Immediately discard** the non-local parameters of $W_L$, reclaiming HBM space.
3. **Backward Pass Loop (Layer $L = L\_{\text{max}} \dots 1$)**:
   - All-Gather $W_L$ to reconstruct full weights.
   - Compute layer gradients $\nabla\_{X\_{L-1}}$ and $\nabla\_{W_L}$.
   - **Reduce-Scatter** the parameter gradients $\nabla\_{W_L}$ to distribute the gradient shards across ranks.
   - Discard full weights $W_L$ and local un-sharded gradients.
4. **Update**: Optimizer steps locally on sharded states using sharded gradients.

### PyTorch/Pythonic Blueprint (Educational FSDP Layer Execution)
```python
import torch
import torch.distributed as dist

class EducationalFSDPLayer(torch.nn.Module):
    def __init__(self, layer_module, rank, world_size):
        super().__init__()
        self.layer = layer_module
        self.rank = rank
        self.world_size = world_size
        
        # Flatten and shard the parameters of this layer
        self.flat_param = self._flatten_and_shard_params(layer_module)
        
    def _flatten_and_shard_params(self, module):
        # Educational sharding: flattens weights and stores 1 / world_size locally
        all_data = torch.cat([p.data.view(-1) for p in module.parameters()])
        shard_size = (all_data.numel() + self.world_size - 1) // self.world_size
        padded_data = torch.zeros(shard_size * self.world_size)
        padded_data[:all_data.numel()] = all_data
        return torch.nn.Parameter(padded_data.chunk(self.world_size)[self.rank])

    def forward(self, x):
        # 1. Gather all weight shards to reconstruct the full weights on-the-fly
        gathered_flat_weights = torch.empty(
            self.flat_param.numel() * self.world_size, 
            device=x.device
        )
        dist.all_gather_into_tensor(gathered_flat_weights, self.flat_param)
        
        # 2. Assign the gathered parameters back to the local module
        self._reconstruct_module_params(gathered_flat_weights)
        
        # 3. Compute forward pass
        out = self.layer(x)
        
        # 4. Immediately clear gathered weights to save memory
        self._clear_unsharded_params()
        
        return out
```

## 4. Hardware Realities & Compute/Memory Accounting

### Multi-Node Network Topologies
- **TPU 3D Toroidal Mesh**: Traditional Google TPU networking connects accelerators as local neighbors wrapping around in a 3-dimensional coordinate space. This is highly optimized for static, predictable multi-dimensional sharding (like TP or DP) but performs poorly on dynamic, random all-to-all patterns. *Note*: TPUv8 (Virgo network) has converged toward an all-to-all tree topology similar to GPUs.
- **GPU Fat-Tree (NVSwitch)**: Connects nodes hierarchically using high-speed spine switches. Bidirectional all-to-all routing enables highly flexible communication patterns, such as mixture-of-experts (MoE) token routing.

### Overlapping Streams
Hiding communication latency is essential. FSDP utilizes dedicated CUDA streams to pipeline operations:
- **Stream 1 (Computation)**: Performs the forward MatMul for layer $L$ on the GPU.
- **Stream 2 (Communication)**: Asynchronously runs the `All-Gather` collective to fetch the sharded weights for layer $L+1$ from remote nodes.
If computation time $T\_{\text{compute}}(L) \ge T\_{\text{comm}}(L+1)$, communication overhead is completely hidden ("free" distributed training).

## 5. Hyperparameter Heuristics for Parallelism

<div id="plotly-cs336-8-zero-memory" class="plotly-chart" aria-label="Interactive Plotly chart: ZeRO Stages Memory Footprint"></div>

<p><em>Figure: ZeRO-3 / FSDP shards model weights, gradients, and optimizer states, slashing 70B parameter memory footprint from 1120 GB to 17.5 GB per GPU.</em></p>

<div id="plotly-cs336-8-pipeline-bubble" class="plotly-chart" aria-label="Interactive Plotly chart: Pipeline Parallelism 1F1B Bubble Overhead"></div>

<p><em>Figure: Pipeline Parallelism (1F1B) Bubble Overhead $F\_{\text{bubble}} = \frac{p-1}{m}$ drops rapidly as micro-batch count $m$ increases.</em></p>


### The 4D Parallelism Layout
When training massive models, engineers use **4D Parallelism** consisting of:
1. **Tensor Parallelism (TP)**: Shards layer weights intra-node (TP $\le 8$, within the fast NVLink domain).
2. **Expert Parallelism (EP)**: Shards sparse MoE experts across nodes (usually EP $\le 32$ or 64).
3. **Pipeline Parallelism (PP)**: Shards model depth-wise across slower network nodes.
4. **Data Parallelism (DP/FSDP)**: Scales out to cover the rest of the available cluster ranks.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **The Activation Memory Wall**: Forgetting to split layer norms and dropouts via **Sequence Parallelism** during TP training means activation memory will quickly overflow at long context lengths, even if parameters are fully sharded.
- **Silent Expert Dropouts**: Naive MoE routers queue tokens into fixed-size buffers per expert. If too many tokens route to a single expert, the queue overflows, causing the system to silently drop tokens and return zero activations.

### Conceptual Reflection Questions
1. *Why is ZeRO-1 (optimizer sharding) completely free of communication overhead relative to naive DDP?*
   **Answer**: In naive DDP, rank gradients are globally averaged using a single All-Reduce ($2N$ bytes transferred). In ZeRO-1, the All-Reduce is decomposed into a Reduce-Scatter (gradients are reduced and sharded; $N$ bytes) followed by an All-Gather (updated weights are collected; $N$ bytes). Since $N + N = 2N$, the total volume of data moved is identical, making ZeRO-1's optimizer memory savings mathematically free of extra overhead.

2. *Why do we prefer Expert Parallelism (EP) over Tensor Parallelism (TP) for mixture-of-experts routing at scale?*
   **Answer**: Tensor parallelism splits weights of dense linear layers, meaning every single layer requires a high-latency `all_reduce` or `all_gather` across the network, which degrades throughput unless ranks are tightly connected via NVLink. Expert Parallelism natively shards independent experts, where tokens only route to specific ranks via point-to-point `all_to_all` dispatches. This sharding does not reduce individual MatMul sizes, keeping arithmetic intensity high on the GPU cores.
