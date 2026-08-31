# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 3: Regularization and Optimization

#### 0. Quick-Recall Summary
*   **The Regularization Mandate:** Regularization ($R(W)$) intentionally degrades training set performance to prevent overfitting and improve generalization on unseen test data. $L2$ regularization penalizes weight magnitude variance, preferring diffuse, spread-out weight vectors, whereas $L1$ regularization drives sparsity by putting a constant pressure on weight values.
*   **The Gradient Dichotomy:** Numerical gradients are approximate, computationally expensive ($O(\#params)$), but trivial to implement, making them ideal for debugging. Analytical gradients are mathematically exact, fast to compute, but highly error-prone to write, necessitating "gradient checking" using the numerical limit formulation.
*   **The Failures of Vanilla SGD:** Stochastic Gradient Descent struggles with: (1) poor conditioning (highly asymmetric ravines causing severe oscillation), (2) local minima and saddle points (zero-gradient stall zones particularly prevalent in high-dimensional spaces), and (3) gradient noise introduced by mini-batch sampling.
*   **Adaptive Step Optimization:** Modern optimizers leverage physical principles: *Momentum* uses velocity to push through saddle points and dampen high-frequency oscillation; *RMSprop* divides learning rates element-wise by the running average of squared gradients to accelerate flat directions; and *Adam* combines both, utilizing bias correction to prevent massive initial step spikes.
*   **The Scalability Limit of Second-Order Methods:** Second-order optimization (Newton's method using the Hessian matrix) offers quadratic convergence but is practically unusable for deep neural networks due to the $O(N^2)$ memory cost of storing the $N \times N$ mixed partial derivatives.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master the dual machinery of Deep Learning: constraining model capacity via regularization to ensure generalization, and traversing high-dimensional non-convex loss landscapes via first-order optimization. It frames the mathematical transition from brute-force random searching to adaptive gradient-based descent.
*   **Lecture Category:** (c) Training/Optimization Practice (a deep blend of first-order optimization algorithms and model regularization heuristics).
*   **Builds on:** Recaps Lecture 2's image classification paradigm, the semantic gap of 3072 pixel values, k-NN recap, and the data loss formulations of Multiclass SVM and Softmax.

---

#### 2. Mathematical Foundations

##### The Generalized Loss Formulation
The total loss $L(W)$ is composed of a data loss term (measuring model fit) and a regularization term (measuring model complexity), balanced by the regularization strength hyperparameter $\lambda$:
$$L(W) = \frac{1}{N} \sum\_{i=1}^{N} L_i\left(f(x_i, W), y_i\right) + \lambda R(W)$$

##### Weight Regularization Techniques
1.  **$L2$ Regularization (Weight Decay / Tikhonov Regularization):**
    Penalizes the squared Euclidean norm of the weight matrix, driving weights smoothly toward zero:
    $$R(W) = \sum\_{k} \sum\_{l} W\_{k,l}^2$$
2.  **$L1$ Regularization:**
    Penalizes the absolute values of the weight matrix, structurally enforcing sparsity:
    $$R(W) = \sum\_{k} \sum\_{l} |W\_{k,l}|$$
3.  **Elastic Net Regularization (Hybrid):**
    Combines both penalties to balance sparsity and weight group diffuse-sharing:
    $$R(W) = \sum\_{k} \sum\_{l} \left( \beta W\_{k,l}^2 + (1 - \beta) |W\_{k,l}| \right)$$

##### Gradient Calculations & Calculus Foundations
*   **1D Numerical Derivative:**
    $$\frac{df(x)}{dx} = \lim\_{h \to 0} \frac{f(x + h) - f(x)}{h}$$
*   **Multidimensional Analytical Gradient (Jacobian/Vector of Partial Derivatives):**
    $$\nabla_W L = \left[ \frac{\partial L}{\partial W\_{1,1}}, \frac{\partial L}{\partial W\_{1,2}}, \dots, \frac{\partial L}{\partial W\_{R,C}} \right]^T$$

##### Optimization Formulations
1.  **Stochastic Gradient Descent (SGD):**
    $$W\_{t+1} = W_t - \alpha \nabla\_{W_t} L(W_t)$$
2.  **SGD with Momentum:**
    Introduces a friction-dampened velocity term $v$ with momentum decay coefficient $\rho$ (typically $0.9$ or $0.99$):
    $$v\_{t+1} = \rho v_t + \nabla_W L(W_t)$$
    $$W\_{t+1} = W_t - \alpha v\_{t+1}$$
3.  **RMSprop:**
    Maintains a decay-filtered running average of squared gradients ($s$) to normalize step size element-wise:
    $$s\_{t+1} = \gamma s_t + (1 - \gamma) (\nabla_W L(W_t))^2$$
    $$W\_{t+1} = W_t - \frac{\alpha}{\sqrt{s\_{t+1} + \epsilon}} \odot \nabla_W L(W_t)$$
4.  **Adam (Adaptive Moment Estimation):**
    Combines first moment tracking (momentum) and second moment tracking (RMSprop), utilizing time-step $t$ bias correction to scale down initial steps:
    $$m\_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla_W L(W_t) \quad \text{(First Moment)}$$
    $$v\_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla_W L(W_t))^2 \quad \text{(Second Moment)}$$
    $$\hat{m}\_{t+1} = \frac{m\_{t+1}}{1 - \beta_1^t} \quad \text{(Unbiased First Moment)}$$
    $$\hat{v}\_{t+1} = \frac{v\_{t+1}}{1 - \beta_2^t} \quad \text{(Unbiased Second Moment)}$$
    $$W\_{t+1} = W_t - \frac{\alpha}{\sqrt{\hat{v}\_{t+1}} + \epsilon} \odot \hat{m}\_{t+1}$$
    *Note: Standard defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.*

##### Second-Order Optimization (Newton-Raphson Step)
Utilizes the inverse Hessian matrix $H^{-1}$ to calculate the exact quadratic minimum step:
$$W\_{t+1} = W_t - H^{-1} \nabla_W L(W_t)$$
where $H\_{i,j} = \frac{\partial^2 L}{\partial W_i \partial W_j}$ is the $N \times N$ Hessian matrix.

---

#### 3. Architecture / Algorithm Walkthrough

##### Algorithmic Logic of Gradient Descent
```
[Initialize Weights W] 
         │
         ▼
 ┌───────────────┐
 │  Mini-Batch   │ ◄─────────────────────────┐
 │   Sampling    │                           │
 └───────┬───────┘                           │
         │ (e.g. 256 samples)                │
         ▼                                   │
 ┌───────────────┐                           │
 │ Forward Pass  │                           │
 │ Compute Loss  │                           │
 └───────┬───────┘                           │
         │ L = Data Loss + λ * R(W)          │
         ▼                                   │ Loop Epochs
 ┌───────────────┐                           │
 │ Backward Pass │                           │
 │ Compute dW    │                           │
 └───────┬───────┘                           │
         │ Analytic Backpropagation          │
         ▼                                   │
 ┌───────────────┐                           │
 │ Optimizer Step│                           │
 │  Update W     │ ──────────────────────────┘
 └───────────────┘
   W = W - α * dW (Adaptive step)
```

##### PyTorch Blueprint (Adaptive Optimization Comparison)
This blueprint implements a basic training loop with manual parameter tracking to demonstrate the differences between Adam and the decoupled weight decay variant **AdamW**:

```python
import torch
import torch.nn as nn
import math

class CustomAdamOptimizer:
    """
    Illustrative implementation of the Adam and AdamW algorithms 
    built from scratch to demonstrate first/second moment tracking,
    bias correction, and decoupled weight decay.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4, use_adamw=True):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.use_adamw = use_adamw
        self.t = 0
        
        # Initialize moments to zero
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            
            # 1. Handle Weight Decay
            if not self.use_adamw:
                # Standard Adam L2 regularization: Add weight penalty directly to gradient
                grad = grad + self.wd * p.data
            
            # 2. Update biased first moment (momentum-like)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            
            # 3. Update biased second raw moment (RMSprop-like)
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            
            # 4. Compute bias-corrected moments
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            
            # 5. Perform weight update
            step_update = m_hat / (torch.sqrt(v_hat) + self.eps)
            
            if self.use_adamw:
                # AdamW: Decoupled Weight Decay is applied directly to weight vector
                p.data = p.data - self.lr * self.wd * p.data - self.lr * step_update
            else:
                p.data = p.data - self.lr * step_update
```

---

#### 4. Visual Intuition & Interpretability

##### The Regularization Viewpoint: Weight Preference Mechanics
*   **$L2$ Diffuse Weight Preference:**
    Suppose input $x = [1, 1, 1, 1]^T$ and we have two alternative weight vectors: $W_1 = [1, 0, 0, 0]^T$ and $W_2 = [0.25, 0.25, 0.25, 0.25]^T$.
    *   The linear score is identical: $W_1 \cdot x = W_2 \cdot x = 1$.
    *   $L1$ regularizer penalties are identical: $R(W_1) = R(W_2) = 1$.
    *   $L2$ regularizer penalties differ vastly: $R(W_1) = 1^2 = 1.0$ whereas $R(W_2) = 4 \times 0.25^2 = 0.25$.
    *   **Visual Intuition:** $L2$ heavily penalizes peaky weights and prefers diffuse weight distributions because spreading out weight values makes the model robust to any single adversarial pixel change. By distributing weights across all input features, the model captures a holistic scene rather than memorizing localized quirks.
*   **$L1$ Sparsity Contour Geometry:**
    If you visualize L1 vs L2 equidistant contours from the origin, $L1$ forms a diamond with sharp vertices pointing along the coordinate axes, while $L2$ forms a circle (hypersphere). Optimization trajectories striking the $L1$ diamond diamond are mathematically driven toward the vertices, setting non-essential features exactly to 0 (achieving native feature selection/sparsity).

##### The Optimization Viewpoint: Landscape Failures
*   **Poor Conditioning (High Condition Number):**
    Visualized as an elongated, highly narrow, deep valley. The slope is extremely steep along the vertical walls but incredibly flat along the horizontal valley floor. Vanilla SGD bounces wildly (oscillates) between the canyon walls, generating massive orthogonal gradients that cancel each other out while making negligible lateral progress toward the valley's true basin.
*   **Saddle Point Trap:**
    Visualized as a horse saddle. At the exact center point, the slope is zero in all directions—however, the landscape curves upward along one axis and downward along another. Vanilla SGD lands in the flat center, sees a gradient of exactly 0, and freezes indefinitely, completely unaware that a steep downward slope lies just inches away on the orthogonal axis.

```
Poor Conditioning (SGD wild oscillation)     Saddle Point Stall (Zero-gradient trap)
             /\  Vertical Wall                          
            /  \                                              _  _
   ───►    /    \    ◄─── Oscillation                        ( \/ )  ◄── upward curve
          /   /\ \                                            \  /
         /   /  \ \                                           (  )   ◄── flat center (g = 0)
        /___/    \_\                                          /        Valley Floor (Flat lateral path)                       /    \  ◄── downward slope
```

---


<div id="plotly-cs231n-3-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 3 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Interactive Loss Landscape Explorer)
To build a deep visual understanding of optimizer trajectory differences, we propose a conceptual **Interactive 3D Loss Landscape & Optimizer Trajectory Simulator**:

*   **Visualization Type:** 3D Surface Plot utilizing dynamic coordinate projections.
*   **Data Fields & Encoding:**
    *   **X & Y axes:** 2D subspace projection of weight space parameters ($W$).
    *   **Z-axis (Height):** Total Loss value $L(W)$ (lower is better).
    *   **Dynamic Overlay Lines:** Trace trajectories of active optimizers (SGD, SGD+Momentum, RMSprop, Adam) color-coded by type.
*   **Interactive Panel Controls:**
    *   **Regularization Toggle & Lambda Slider ($\lambda$):** Dragging $\lambda$ from $0$ to $\infty$ morphs the underlying loss landscape surface from highly complex, jagged, non-convex local minima into a smooth, convex, parabolic bowl (the $L2$ quadratic envelope), showing how regularization physically "stabilizes" the landscape.
    *   **Condition Number Slider:** Asymmetry multiplier of the valley. Tuning this high compresses the landscape into a narrow slit, visually demonstrating SGD's catastrophic bouncing vs. Adam's smooth, centered descent.
    *   **Optimizer Hyperparameter Toggles:** Play/Pause step-wise iterations to watch Momentum's velocity carry it safely past saddle points and local minima while SGD freezes inside them.

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **The Random Search Baseline:** Randomly sampling weight configurations yields a meager **$15.5\%$ accuracy** on the 10-class CIFAR-10 classification task, showing the extreme limitations of non-gradient optimization.
*   **Deep Learning Limit:** State-of-the-art deep learning architectures utilizing analytical backpropagation and adaptive momentum optimization practically solve CIFAR-10, achieving **$99.7\%$ accuracy**.
*   **Linear Scaling Law (Empirical Batch Heuristic):** If you scale your training batch size by a factor of $N$ to increase parallel hardware throughput, you should scale your learning rate $\alpha$ linearly by that same factor of $N$. This ensures the magnitude of gradient updates scales proportionally to the larger step size representative of macro-batches.
*   **The Adam King:** The Adam optimizer, originally published in 2015, revolutionized deep learning optimization, leading to its receipt of the prestigious **ICLR 2025 Test of Time Award**.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Training Silent Failures
*   **Adam's Zero-Initialization Spike:**
    In the first few steps of Adam, the moments $m$ and $v$ are initialized to 0. If you do not apply the time-dependent bias correction ($\%(1 - \beta^t)\%$), the denominator $\sqrt{v}$ remains extremely small, causing the first step update to balloon into a massive, uncontrolled spike that can permanently ruin weight initialization.
*   **The Learning Rate Cliff:**
    Setting the learning rate too high causes the model to bounce completely out of the local valley, manifesting as an exponentially growing loss curve or `NaN` outputs. Conversely, a learning rate that is too low produces a flat, agonizingly slow loss decay curve that will stall training.

##### Graduate-Level Reflection Questions
1.  **Adam vs. AdamW Dynamics:**
    Standard $L2$ regularization adds $\lambda W$ directly to the gradient term before moment calculations: $\nabla\_{W} L\_{total} = \nabla\_{W} L\_{data} + \lambda W$. In Adam, this means the weight decay penalty gets scaled by the running second moment $v_t$, which divides the penalty by the historical magnitude of the gradients. Why does this scaling mathematically alter the intentionality of weight decay, and how does AdamW’s decoupled weight decay formulation ($W\_{t+1} = W_t(1 - \alpha \lambda) - \text{AdamStep}$) restore true scale-invariant weight decay?
2.  **Saddle Point Frequency vs. Model Dimensionality:**
    In low-dimensional 1D/2D spaces, local minima (bowls) are highly common. However, why does the ratio of saddle points to local minima grow exponentially as we scale neural network capacity from millions to billions of parameters, and how does the Hessian matrix's eigenvalue distribution mathematically explain this shift?
3.  **The Physics of Momentum vs. RMSprop in Ravines:**
    When traversing an extremely narrow ravine with a high Hessian condition number, both Momentum and RMSprop prevent catastrophic oscillations but do so through entirely different mathematical mechanics. Explain how Momentum uses velocity vector cancellation to stabilize the transverse axis, while RMSprop uses gradient magnitude division to physically contract the coordinates of the landscape. Which method is more sensitive to gradient noise and why?
