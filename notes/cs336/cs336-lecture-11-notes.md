# CS336 Lecture 11: Scaling Laws & Advanced Optimizers

## 0. Quick-Recall Summary
- **MUP (Maximal Update Parameterization)**: Re-parameterizes model initialization and learning rates to keep the optimal hyperparameters invariant across model widths.
- **MUP Invariants**: Guarantees that at initialization, activation scales are $O(1)$ and after one gradient step, the feature learning update is also $O(1)$ regardless of width $n$.
- **WSD (Warm-up Stable Decay) Learning Rate**: A constant stable learning rate phase that can be run indefinitely and decayed on-demand for cheap scaling sweeps.
- **Muon Optimizer**: A spectral, momentum-based optimizer designed specifically for matrix parameters, using Newton-Schultz iteration to orthogonalize updates on GPUs.
- **Newton-Schultz Orthogonalization**: Approximates Singular Value Decomposition (SVD) using only fast GPU matrix multiplications, bypassing slow native SVD kernels.

## 1. Core Paradigm & Systems Overview
- **Objective**: Stabilize hyperparameter drift as models scale and introduce advanced optimization schemes that treat distinct parameter types (vectors vs. matrices) according to their unique mathematical structures.
- **Primary Bottleneck**: High-dimensional hyperparameter search. Training frontier models is too expensive for trial-and-error sweeps; hyperparameters must transfer seamlessly from small-scale proxies.
- **Builds on**: Classical Scaling Laws (Lecture 9) — transitions from simple compute-to-parameter fits to structured, scale-invariant optimization recipes.

## 2. Theoretical & Mathematical Primitives

### MUP Activation Scale Invariants
To achieve stable feature learning as width $n 	o \infty$, we require two conditions:
1. **Activation Scale**: $\mathbb{E}[\|x_l\|^2] = \Theta(1)$.
2. **Feature Update Scale**: $\Delta x_l = O(1)$ after one gradient step.

In standard parameterization (SP), when initializing weights $W \sim \mathcal{N}(0, \sigma^2)$ with $\sigma^2 = 1/\text{fan\_in}$, the output activation scales fine, but gradient steps scale as $O(1/n)$, vanishing as width increases.

### MUP Parameter Scaling Rules
Under MUP, for a layer with input dimension $n\_{\text{in}}$ and output dimension $n\_{\text{out}}$:
- **Weight Initialization**:
  $$W \sim \mathcal{N}\left(0, \sigma^2 \cdot \frac{1}{n\_{\text{in}}} \right)$$
- **Layer-wise Learning Rate Scaling**:
  $$\eta\_{\text{layer}} = \eta_0 \cdot \frac{n\_{\text{out}}}{n\_{\text{in}}}$$
- **For Adam Optimizer**: The learning rate must scale as $O(1/n\_{\text{in}})$ because Adam normalizes step sizes by gradient variances, altering the update dynamics:
  $$\eta\_{\text{Adam}} = \frac{\eta_0}{n\_{\text{in}}}$$

### Muon Spectral Orthogonalization
Muon decomposes optimizer updates by treating 2D matrix weights $W$ through spectral orthogonalization. The update step is defined as:

$$G \leftarrow \text{gradient of } W$$

$$M_t \leftarrow \mu M\_{t-1} + (1 - \mu) G \quad \text{(Momentum step)}$$

$$B_t \leftarrow \text{Newton-Schultz-5}(M_t) \quad \text{(Orthogonalized update)}$$

$$W \leftarrow W - \eta \cdot B_t$$

Where **Newton-Schultz-5** iteratively solves for the orthogonalized matrix $O$ of momentum update $M_t$ (clamping singular values to 1) using only matrix multiplications:

$$X_0 \leftarrow \frac{M_t}{\|M_t\|_2}$$

$$X\_{k+1} \leftarrow X_k \left( \frac{15 I - 10 X_k^T X_k + 3 (X_k^T X_k)^2}{8}  \right)$$

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### Newton-Schultz-5 Matrix Orthogonalization Logic
1. Normalize input matrix $M$ to have spectral norm $\le 1$.
2. For $k$ steps (typically 5):
   - Compute $A = X_k^T \cdot X_k$.
   - Compute $B = A^2$.
   - Update: $X\_{k+1} = \frac{1}{8} X_k (15 I - 10 A + 3 B)$.
3. Return $X_5$ as the orthogonalized parameter update direction.

### PyTorch/Pythonic Blueprint (Educational Newton-Schultz-5 step)
```python
import torch

# Educational implementation of Newton-Schultz 5 for Muon
def newton_schultz_5(M, steps=5):
    # M shape: (in_features, out_features)
    # Ensure spectral norm <= 1
    norm = torch.linalg.matrix_norm(M, ord=2)
    X = M / (norm + 1e-7)
    
    for _ in range(steps):
        XX = torch.matmul(X.t(), X)
        XX2 = torch.matmul(XX, XX)
        # Newton-Schultz 5 polynomial coefficients
        poly = 15.0 * torch.eye(XX.shape[0], device=M.device) - 10.0 * XX + 3.0 * XX2
        X = 0.125 * torch.matmul(X, poly)
        
    return X
```

## 4. Hardware Realities & Compute/Memory Accounting
- **Fast GPU Orthogonalization**: Although Singular Value Decomposition (SVD) is mathematically perfect for matrix orthogonalization, running SVD on GPUs is extremely slow due to control divergence and execution dependencies. Since Newton-Schultz relies strictly on matrix multiplications, it maps directly to **Tensor Cores**, running near speed-of-light on modern GPUs.

## 5. Hyperparameter Heuristics

<div id="plotly-cs336-11-mup-transfer" class="plotly-chart" aria-label="Interactive Plotly chart: muP Zero-Shot Learning Rate Transfer"></div>

<p><em>Figure: Maximal Update Parameterization ($\mu\text{P}$) enables zero-shot hyperparameter transfer: optimal learning rate remains constant across model width.</em></p>

<div id="plotly-cs336-11-wsd-schedule" class="plotly-chart" aria-label="Interactive Plotly chart: Warmup-Stable-Decay Schedule"></div>

<p><em>Figure: Warmup-Stable-Decay (WSD) maintains high learning rates during a prolonged stable phase, allowing arbitrary checkpoint decay without retraining.</em></p>

- **Weight Decay in MUP**: Decoupled weight decay must be scaled as $O(1/n\_{\text{in}})$ under MUP to avoid parameter saturation and gradient explosion.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **MUP Failure on Vector Parameters**: MUP is derived assuming infinite width scaling ($n 	o \infty$) of matrix multiplications. Applying MUP scaling rules to 1D vector parameters (such as RMSNorm gains, biases, or positional embeddings) will collapse training, as their learning rates must remain constant.

### Conceptual Reflection Questions
1. *Why does Adam require a different learning rate scaling rule ($1/n$) than SGD ($1/\sqrt{n}$) under MUP?*
   **Answer**: In SGD, parameter updates are directly proportional to the gradients: $\Delta W = \eta 
abla_W$. Since gradients scale as $1/\sqrt{n}$ due to activation dimensions, SGD requires a learning rate scaling of $1/\sqrt{n}$ to balance the update scale. However, Adam divides the gradients element-wise by the running estimate of their second moments: $m_t / (\sqrt{v_t} + \epsilon)$. This normalizes the step size, removing the natural gradient scaling. To maintain an $O(1/\sqrt{n})$ weight update on feature learning, Adam's learning rate must be aggressively scaled down as $1/n$.

2. *Explain the utility of WSD (Warm-up Stable Decay) for scaling laws research.*
   **Answer**: Cosine learning rate decay schedules are tied strictly to a fixed token budget $T$. If an engineer wants to evaluate model losses at $T_1 = 5\text{B}$ and $T_2 = 10\text{B}$ tokens, they must run two separate training runs from scratch. WSD decouples the training duration from the decay phase: a single model can be trained indefinitely at the stable learning rate, and checkpoints can be branched off and annealed on-demand, saving massive compute resources.
