# CS336 Lecture 9: Scaling Laws (Foundations)

## 0. Quick-Recall Summary
- **Power-Law Regularity**: Upstream validation loss scales as a highly predictable polomial function of compute ($C$), data size ($D$), and parameter count ($N$).
- **Kaplan vs. Chinchilla**: Kaplan (2020) claimed model size should scale faster than tokens ($N \propto C^{0.73}$, $D \propto C^{0.27}$), whereas Chinchilla (2022) corrected this to a symmetric 1:1 scaling ($N \propto C^{0.5}$, $D \propto C^{0.5}$).
- **The Chinchilla 20:1 Rule**: For compute-optimal training, models require approximately **20 tokens per non-embedding parameter**.
- **Downstream Divergence**: While pre-training perplexity follows precise power laws, downstream task accuracy often scales as a sigmoidal (discontinuous) function, creating "emergent" phenomena.
- **Inference-Optimal Deviation**: Modern frontier models train far beyond the Chinchilla compute-optimal boundary (e.g., 100+ tokens per parameter) to maximize inference-time throughput.

## 1. Core Paradigm & Systems Overview
- **Objective**: Predict large-scale training performance using small-scale, cheap runs, enabling precise allocation of million-dollar compute budgets.
- **Primary Bottleneck**: Optimization and Hyperparameter sensitivity. If learning rates, batch sizes, or decay schedules are not scaled correctly alongside model size, scaling laws collapse, leading to heavily suboptimal models.
- **Builds on**: Architecture design (Lecture 3) — uses stable layers (such as RMSNorm, SwiGLU, and pre-normalization) to ensure that empirical scaling holds across orders of magnitude of compute without exploding gradients.

## 2. Theoretical & Mathematical Primitives

### Empirical Loss Power Laws
Under resource-constrained scaling (where only one bottleneck is active at a time), validation loss $L$ is modeled as a power-law function:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

Where $N_c, D_c, C_c$ are constant intercepts, and $\alpha_N, \alpha_D, \alpha_C$ are scaling exponents.

### Joint Parametric Fitting (Chinchilla)
The loss surface as a joint function of parameter count $N$ and token count $D$ is modeled as:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Where:
- $E$: Irreducible loss of the data distribution.
- $A, B$: Constant amplitudes.
- $\alpha, \beta$: Exponents governing the scaling efficiency of parameters and tokens respectively.

### Optimization under Flops Constraint
Given a training FLOPs budget $C \approx 6ND$, we solve the constrained optimization problem:

$$\min\_{N, D} L(N, D) \quad \text{subject to} \quad 6ND = C$$

Using Lagrange multipliers, the optimal parameter and token scales are derived as:

$$N(C) = a C^a, \quad D(C) = b C^b$$

Where:

$$a = \frac{\beta}{\alpha + \beta}, \quad b = \frac{\alpha}{\alpha + \beta}$$

### Kaplan vs. Chinchilla Parameters
- **Kaplan (Suboptimal)**: $a \approx 0.73$, $b \approx 0.27$. This implies that for a 10x compute increase, model parameters should scale by 5.4x and data tokens by only 1.8x.
- **Chinchilla (Optimal)**: $\alpha \approx 0.34$, $\beta \approx 0.28$, resulting in $a \approx 0.45$, $b \approx 0.55$. This dictates near-symmetric scaling ($N \propto C^{0.5}$, $D \propto C^{0.5}$), leading to the **20:1 token-to-parameter ratio** at optimal convergence.

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### The IsoFLOP Profile Sweep (Method 2)
1. Select $K$ target FLOP budgets $C_1, C_2, \dots, C_K$.
2. For each budget $C_k$:
   - Choose $J$ different model sizes $N\_{k,1}, N\_{k,2}, \dots, N\_{k,J}$.
   - Compute corresponding token budgets: $D\_{k,j} = \frac{C_k}{6 N\_{k,j}}$.
   - Train each model and record terminal validation loss $L(N\_{k,j}, D\_{k,j})$.
3. Fit a quadratic curve to the loss data points $(N\_{k,j}, L\_{k,j})$ to find the optimal parameter size $N^*_k$ that minimizes the loss for budget $C_k$.
4. Fit a power-law line through the coordinates $(C_k, N^*_k)$ in log-log space to derive the scaling exponents $a$ and $b$.

### PyTorch/Pythonic Blueprint (Educational IsoFLOP Fitting)
```python
import numpy as np
from scipy.optimize import minimize

# Educational script to fit Chinchilla scaling parameters from IsoFLOP sweeps
def fit_chinchilla_parameters(N_runs, D_runs, loss_runs):
    # N_runs, D_runs, loss_runs are 1D arrays of historical model runs
    # Target loss: L(N, D) = E + A / (N**alpha) + B / (D**beta)
    
    def loss_func(params):
        E, A, B, alpha, beta = params
        # Apply strict positivity constraints to exponents
        if alpha <= 0 or beta <= 0 or A <= 0 or B <= 0:
            return 1e10
        predicted_loss = E + A / (N_runs ** alpha) + B / (D_runs ** beta)
        return np.mean((loss_runs - predicted_loss) ** 2)

    # Initial guess
    initial_guess = [1.5, 400.0, 400.0, 0.3, 0.3]
    res = minimize(loss_func, initial_guess, method="Nelder-Mead")
    return res.x  # Returns [E, A, B, alpha, beta]
```

## 4. Hardware Realities & Compute/Memory Accounting
- **FLOPs Accounting Errors**: Kaplan's suboptimal scaling was partly caused by counting the **embedding parameters** ($V \cdot D$) within the model size $N$. Since embedding parameters are only accessed once per token and do not participate in dense matrix multiplication, they do not scale linearly with FLOP usage. Removing them yields the correct, clean non-embedding FLOP relationship $C \approx 6ND$.

## 5. Hyperparameter Heuristics & Pitfalls

<div id="plotly-cs336-9-isoflop" class="plotly-chart" aria-label="Interactive Plotly chart: Chinchilla IsoFLOP Curves"></div>

<p><em>Figure: Chinchilla IsoFLOP Curves — Optimal training trajectory requires symmetric 20:1 token-to-parameter allocation ($N^* \propto C^{0.5}, D^* \propto C^{0.5}$).</em></p>

<div id="plotly-cs336-9-kaplan-vs-chinchilla" class="plotly-chart" aria-label="Interactive Plotly chart: Kaplan vs Chinchilla Parameter Allocation"></div>

<p><em>Figure: Parameter Allocation — Kaplan over-allocates parameters ($N \propto C^{0.73}$), while Chinchilla balances parameter and token scaling symmetrically.</em></p>

- **Learning Rate Schedule Mismatch**: Kaplan held the learning rate schedule constant or decayed it over fixed steps, preventing smaller models on the IsoFLOP curve from fully annealing and underestimating their data capacity.
- **Warm-up Stable Decay (WSD)**: Rather than running separate cosine schedules from scratch for every sweep, modern scaling runs use WSD learning rates (a constant warmth phase, followed by a flat stable phase, and a short 10% annealing/decay phase). The checkpoint can be "rewound" and decayed at any point, allowing multiple data points to be derived from a single run.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **Tailor Approximation Illusion**: Over narrow compute ranges (less than one order of magnitude), any scaling trend will look perfectly linear on both log-log and linear-linear charts due to Taylor expansions. Extrapolating from narrow ranges to frontier models causes catastrophic hyperparameter mismatch at scale.

### Conceptual Reflection Questions
1. *Why does modern LLM training deliberately violate the Chinchilla compute-optimal ratio of 20:1?*
   **Answer**: Chinchilla optimizes for **training compute** efficiency. However, at serving time, model operators care about **inference compute**. A smaller model trained far past its Chinchilla-optimal token count (e.g., 100:1) costs slightly more to train, but requires significantly fewer active parameters and memory bandwidth during sequential decoding, generating massive savings in inference cost.

2. *Explain why the aspect ratio of depth to width ($L/d\_{\text{model}}$) of a transformer converges to a stable minima of ~100 across different compute budgets.*
   **Answer**: According to empirical evaluations, depth is more expensive for parallelization because deeper networks introduce strict serial sequential dependencies, increasing pipeline parallelism overhead. Conversely, extremely wide networks are easier to parallelize via tensor parallel weight splitting. This trade-off between parallelization constraints and representation depth converges stably onto an optimal aspect ratio of approximately 100.
