# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 14: Generative Models 2

#### 0. Quick-Recall Summary
*   **Zero-Sum Minimax Game:** GANs optimize a competitive zero-sum objective between a generator $G$ and discriminator $D$. This creates highly unstable, non-stationary dynamics that lack an interpretable loss curve and often collapse into isolated modes.
*   **Non-Saturating Gradient Hack:** The naive minimax objective suffers from flat gradients early in training when the generator is weak. Resolving this requires a heuristic shift to minimizing $-\log D(G(z))$ to provide strong early gradient signals.
*   **Rectified Flow & ODE Trajectories:** Rectified flow simplifies diffusion by defining straight-line ODE trajectories. It linearly interpolates between clean data $X_0$ and Gaussian noise $Z$, training the model to predict the constant velocity vector $V = Z - X_0$.
*   **Classifier-Free Guidance (CFG):** CFG control sample alignment by linearly combining conditional and unconditional velocity predictions ($v\_{CFG} = (1+w)v_y - w v\_{\emptyset}$). It requires evaluating the model twice per step and is trained by dropping the condition 50% of the time.
*   **Hybrid Latent Diffusion Pipelines:** State-of-the-art architectures (e.g., Stable Diffusion, Flux, Veo) operate in the low-dimensional latent space of a frozen, pretrained VAE. To prevent the typical blurriness of standard VAEs, the latent autoencoder is regularized using a hybrid GAN discriminator to guarantee crisp reconstructions.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master the theoretical and practical transitions of generative modeling from implicit, competitive density estimators (GANs) to iterative, score-matching denoising processes (Rectified Flow and Diffusion Models). The lecture focuses on how these different paradigms address the fundamental challenge of mapping a simple, tractable noise prior $P_z$ to a complex, high-dimensional empirical data manifold $P\_{data}$.
*   **Lecture Category:** (a) Mathematical Foundations and (b) Architecture Design (specifically exploring minimax game dynamics, stochastic differential equations, linear flow interpolations, and hybrid latent space pipelines).
*   **Builds on:** Lecture 13 (Generative Models 1), extending from explicit density autoregressive models and Variational Autoencoders (VAEs) to implicit sampling frameworks (GANs) and iterative continuous probability flow ODEs.

---

#### 2. Mathematical Foundations

##### A. Generative Adversarial Networks (GANs) Minimax Objective
The objective is formulated as a zero-sum game with value function $V(G, D)$:
$$\min\_{G} \max\_{D} V(G, D) = \mathbb{E}\_{x \sim p\_{data}}\left[\log D(x)\right] + \mathbb{E}\_{z \sim p\_{z}}\left[\log (1 - D(G(z)))\right]$$
*   **Discriminator Objective ($D$):** Maximizes the probability of assigning the correct label ($1$ for real, $0$ for generated).
*   **Generator Objective ($G$):** Minimizes the probability that $D$ identifies its generated samples as fake.

##### B. Analytical Proof of the Optimal Discriminator
For a fixed generator $G$ inducing a distribution $p_G$, the objective can be written as an integral over the data space $x$:
$$V(G, D) = \int\_{x} \left( p\_{data}(x) \log D(x) + p_G(x) \log (1 - D(x)) \right) dx$$
To find the optimal discriminator $D^*_G(x)$, take the functional derivative of the integrand with respect to $D(x)$ and set it to zero:
$$\frac{\partial}{\partial D(x)} \left[ p\_{data}(x) \log D(x) + p_G(x) \log (1 - D(x)) \right] = 0$$
$$\frac{p\_{data}(x)}{D(x)} - \frac{p_G(x)}{1 - D(x)} = 0$$
$$p\_{data}(x)(1 - D(x)) = p_G(x) D(x)$$
$$p\_{data}(x) - p\_{data}(x) D(x) = p_G(x) D(x)$$
$$D^*_G(x) = \frac{p\_{data}(x)}{p\_{data}(x) + p_G(x)}$$
Substituting $D^*_G(x)$ back into the minimax equation demonstrates that when $D$ is optimal, the generator minimizes the Jensen-Shannon Divergence ($JSD$) between $p_G$ and $p\_{data}$. The global minimum occurs if and only if $p_G = p\_{data}$, yielding $D^*_G(x) = 0.5$.

##### C. Rectified Flow Linear Interpolation & Velocity Fields
Rectified flow defines a continuous probability path by constructing straight-line trajectories between data $X_0 \sim P\_{data}$ and noise $Z \sim \mathcal{N}(0, I)$.
*   **Linear Interpolation (Flow Path):**
    $$X_t = (1 - t) X_0 + t Z, \quad t \in [0, 1]$$
*   **Constant Reconstructive Velocity Vector:**
    $$V\_{GT} = \frac{dX_t}{dt} = Z - X_0$$
*   **Velocity Vector Field Loss:**
    $$L(\theta) = \mathbb{E}\_{X_0 \sim P\_{data}, Z \sim \mathcal{N}(0, I), t \sim \mathcal{U}(0, 1)} \left[ \| v_\theta(X_t, t) - (Z - X_0) \|^2 \right]$$
    where $v_\theta$ is a neural network parameterized by $\theta$ that inputs the noisy sample $X_t$ and the scalar time $t$ to predict the velocity field.

##### D. Theoretical Target of the Velocity Predictor
Because multiple straight-line paths can intersect at an intermediate point $X_t$, the optimal network prediction $v^*(x, t)$ must marginalize over all possible trajectories:
$$v^*(x, t) = \mathbb{E}[Z - X_0 \mid X_t = x]$$
This represents the conditional expectation of the velocity vector given the intermediate state $X_t = x$.

##### E. Classifier-Free Guidance (CFG) Vector Formulation
During training, the conditioning signal $y$ (e.g., text embeddings) is randomly set to a null token $\emptyset$ with probability $p\_{drop} \approx 0.5$. At inference, the guidance vector $v\_{CFG}$ is computed as a linear extrapolation:
$$v\_{CFG} = (1 + w) v_\theta(X_t, t, y) - w v_\theta(X_t, t, \emptyset)$$
where $w \ge 0$ is the guidance scale.
*   Setting $w = 0$ yields standard conditional generation.
*   Setting $w > 0$ amplifies the conditioning direction relative to the unconditional prior.

##### F. Generalization of Diffusion Path Formulations
Different continuous-time diffusion formulations represent variations of the linear path parameters:
$$X_t = a_t X_0 + b_t Z$$
*   **Rectified Flow:** $a_t = 1 - t$ and $b_t = t$.
*   **Variance Preserving (VP):** Constrains the path to preserve unit variance under independent assumptions:
    $$a_t^2 + b_t^2 = 1 \implies X_t = \sqrt{1 - \sigma_t^2} X_0 + \sigma_t Z$$
*   **Variance Exploding (VE):** Preserves the raw data scale while adding unbounded noise:
    $$a_t = 1 \quad \text{and} \quad b_t = \sigma_t \implies X_t = X_0 + \sigma_t Z$$

---

#### 3. Architecture / Algorithm Walkthrough

##### A. Alternate Minimax Optimization Loop (Modified Generator Loss)
The training dynamics of GANs require alternating optimization steps between the discriminator and generator weights.

```
Real Data X ──> [ Discriminator D ] ──> log D(X) (Maximize)
                                               │
Noise Prior Z ──> [ Generator G ] ──> G(Z) ────┴──> log(1 - D(G(Z))) (Minimize G / Maximize D)
```

To prevent gradient vanishing early in training when $D$ easily rejects poor generations, the generator's objective is modified:

```python
# Conceptual pseudocode for alternating GAN step
for epoch in range(num_epochs):
    for real_images in data_loader:
        # 1. Update Discriminator: Maximize log D(x) + log(1 - D(G(z)))
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        d_loss = - (torch.log(discriminator(real_images)) + torch.log(1.0 - discriminator(fake_images))).mean()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # 2. Update Generator: Minimize -log D(G(z)) (the non-saturating gradient hack)
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        g_loss = - torch.log(discriminator(fake_images)).mean() # Modified generator loss
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
```

##### B. Rectified Flow Training and Inference Mechanics
Rectified flow bypasses competitive optimization by using a direct regression objective.

```
[Training Loop]
X_0 ~ P_data ──┐
               ├─> X_t = (1-t)*X_0 + t*Z ──> [ Model v_theta ] ──> v_pred
Z ~ N(0, I)  ──┤                                                      │
t ~ U(0, 1)  ──┘                                       (MSE Loss) <───┴──> V_GT = Z - X_0

[Inference Loop]
X_1 ~ N(0, I) ──> [ Model v_theta(X_1, t=1) ] ──> v_pred ──> X\_{t-dt} = X_t - dt * v_pred ──> X_0
```

##### C. PyTorch Blueprint: Rectified Flow with Classifier-Free Guidance
This blueprint implements both training and inference steps for a Rectified Flow model, including a classifier-free guidance interface:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityNet(nn.Module):
    """
    A placeholder network representing a Diffusion Transformer (DiT) or U-Net 
    predicting velocity vectors from noisy inputs, timestamps, and class embeddings.
    """
    def __init__(self, channels=3, latent_dim=128, num_classes=10):
        super().__init__()
        self.spatial_conv = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.time_embedding = nn.Linear(1, 64)
        self.class_embedding = nn.Embedding(num_classes + 1, 64) # +1 for null token
        self.out_proj = nn.Conv2d(64, channels, kernel_size=3, padding=1)

    def forward(self, x_t, t, y):
        # x_t: [B, C, H, W]
        # t: [B, 1] scalar timestamps
        # y: [B] class labels (with y_null index mapping to num_classes)
        feat_x = self.spatial_conv(x_t)
        feat_t = self.time_embedding(t).unsqueeze(-1).unsqueeze(-1)
        feat_y = self.class_embedding(y).unsqueeze(-1).unsqueeze(-1)
        
        # Inject condition information via scale/shift/addition
        h = feat_x + feat_t + feat_y
        return self.out_proj(F.relu(h))

class RectifiedFlowPipeline:
    def __init__(self, model, optimizer, num_classes=10, null_label=10):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.null_label = null_label

    def train_step(self, x_0, y):
        """
        Performs one training step of rectified flow.
        """
        self.model.train()
        batch_size = x_0.size(0)
        device = x_0.device
        
        # 1. Sample continuous noise Z ~ N(0, I) matching the shape of data
        z = torch.randn_like(x_0)
        
        # 2. Sample uniform time steps t ~ U(0, 1)
        t = torch.rand(batch_size, 1, device=device)
        t_expanded = t.unsqueeze(-1).unsqueeze(-1) # [B, 1, 1, 1] for broadcasting
        
        # 3. Linearly interpolate to get intermediate corrupted states x_t
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * z
        
        # 4. Apply 50% classifier-free guidance drop probability
        y_dropped = y.clone()
        drop_mask = torch.rand(batch_size, device=device) < 0.5
        y_dropped[drop_mask] = self.null_label
        
        # 5. Predict velocity field and compute MSE loss against target (Z - X_0)
        v_pred = self.model(x_t, t, y_dropped)
        v_gt = z - x_0
        
        loss = F.mse_loss(v_pred, v_gt)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample_cfg(self, shape, y, num_steps=50, w=1.5):
        """
        Samples from the noise prior to the data manifold using Euler integration 
        and Classifier-Free Guidance.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        batch_size = y.size(0)
        
        # Start at full noise at t = 1.0
        x_t = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        
        # Loop backwards from t = 1.0 to t = 0.0 (marching noise to data)
        for i in range(num_steps):
            t_val = 1.0 - (i * dt)
            t_tensor = torch.full((batch_size, 1), t_val, device=device)
            
            # Predict conditional velocity v(x_t, t, y)
            v_cond = self.model(x_t, t_tensor, y)
            
            # Predict unconditional velocity v(x_t, t, null)
            y_null = torch.full_like(y, self.null_label)
            v_uncond = self.model(x_t, t_tensor, y_null)
            
            # Linear extrapolation step
            v_cfg = (1.0 + w) * v_cond - w * v_uncond
            
            # Take a small step along the vector flow direction (Euler step)
            x_t = x_t - dt * v_cfg
            
        return x_t # Returns reconstructed data x_0
```

---

#### 4. Visual Intuition & Interpretability

##### A. Non-Saturating Loss Curve Visual Geometry
The original minimax formulation uses $\log(1 - D(G(z)))$ for the generator. When the generator is completely unoptimized at step zero, $D(G(z)) \approx 0$, which places the gradients at the extremely flat left boundary of the logarithm function. 
By swapping the generator objective to minimize $-\log D(G(z))$ (equivalent to maximizing the probability of the fake being classified as real), the left boundary shifts to a steep, high-gradient curve:

```
Gradient Magnitude
  ▲
  │   Original Loss: log(1 - D(G(z))) ────> [Flat Curve, No Gradients]
  │   Modified Loss: -log D(G(z))     ────> [Steep Curve, High Gradients]
  │
  └───────────────────────────────► D(G(Z))
```

##### B. Vector Intersections in Flow Fields
During the forward training of a rectified flow model, straight lines are drawn between empirical data points $X_0$ and random Gaussian prior vectors $Z$. In high-dimensional spaces, these straight paths inevitably cross. 
As a result, a single intermediate state $X_t$ could be reached from multiple distinct starting points $X_0^{(1)}$ and $X_0^{(2)}$:

```
X_0^{(1)} (Cat) ───\                     /─── Z_a (Noise)
                    \                   /
                     \─► X_t (Crossing) ─► [Optimal Predictor v*(x,t)] -> Mean Expected Direction
                    /                   \
X_0^{(2)} (Dog) ───/                     \─── Z_b (Noise)
```

At these intersection points, the network cannot predict both paths simultaneously. The $L_2$ regression objective forces the model to predict the average of all intersecting paths at that location. This represents the expected velocity field pointing toward the average mean of the spatial modes.

##### C. Classifier-Free Guidance Vector Extrapolation
If trained naively, conditional models often ignore the conditional prompt $y$ due to representation shortcuts. Classifier-Free Guidance amplifies the visual attributes by calculating both the conditional vector $v\_{cond}$ and the unconditional vector $v\_{uncond}$, then pushing the inference vector $v\_{CFG}$ past the conditional prediction:

```
                Unconditional Vector (v_uncond)
                     ┌───────────►
                     │
  Starting State X_t ┼───────────┼───────────► Extrapolated Guidance Vector (v_cfg)
                     │           v_cond
                     └───────────►
                Conditional Vector (v_cond)
```

---


<div id="plotly-cs231n-14-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 14 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)

We propose an interactive **Probability Flow ODE Vector Field Trajectory Simulator** to visualize how continuous paths transport noise to data:

*   **Visualization Type:** 2D Particle Flow Grid with Vector Fields.
*   **Data Fields & Encoding:**
    *   **X/Y Coordinates:** The 2D projection of the latent space.
    *   **Background Heatmap:** Probability Density estimate $P(X_t)$ at any selected timestamp $t$ (changing from standard Normal at $t=1$ to distinct multimodal clusters of real data at $t=0$).
    *   **Flow Particles:** A swarm of 100 particles representing active samples marching from noise to data.
*   **Interactive Controls:**
    *   **Timestamp Slider ($t \in$):** Moves particles forward/backward along their trajectory.
    *   **Guidance Scale Slider ($w \in$):** Dynamically scales the magnitude of the vector projection. Increasing $w$ visually collapses the trajectories into tight, highly-focused modal centers (demonstrating how high guidance trades off visual diversity for sample quality).
    *   **Noise Schedule Selector:** Toggle between *Rectified Flow* (straight-line trajectories), *Variance Preserving* (curved trajectories following a spherical boundary), and *Variance Exploding* (widely dispersing paths before snapping to modes).

---

#### 6. Empirical Design Heuristics & Benchmark Results

##### A. CIFAR-10 & ImageNet Milestones
*   **Early GAN Baselines:** The original DCGAN model (5-layer CNN, 2015) produced early low-resolution samples ($64\times64$) that served as a proof of concept.
*   **The Blur Barrier:** Variational Autoencoders (VAEs) scale poorly to sharp high-frequency features, historically resulting in blurry reconstructions. GANs bypass this by using an adversarial discriminator that acts as a trainable local high-frequency loss function.
*   **State-of-the-Art Scaling:** Current state-of-the-art architectures combine VAE, GAN, and Diffusion objectives:
    1.  A **VAE** compresses the image 8x spatially (e.g., $512\times512\times3 \rightarrow 64\times64\times16$) to reduce sequence length.
    2.  The VAE is trained with an adversarial **GAN loss** to ensure reconstruction sharpness.
    3.  A **Diffusion Transformer (DiT)** is trained on this compressed latent space.

##### B. Guidance, Scheduling, & Sampling Heuristics
*   **CFG Trade-offs:** High guidance values ($w \ge 4.0$) improve prompt compliance and sample sharpness but reduce sample diversity and can cause color saturation artifacts.
*   **Euler Step Limits:** Rectified flow can sample high-quality images in 30 to 50 steps using simple first-order Euler solvers.
*   **Noise Scheduling:** For high-resolution images, sampling noise levels from a **logit-normal** distribution concentrates training steps in the difficult middle regions ($t \approx 0.5$) rather than the trivial boundaries ($t \to 0$ or $t \to 1$).

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### A. Gotchas & Silent Failure Modes
*   **Mode Collapse:** The generator discovers a small subset of safe spatial modes that easily fool the discriminator (e.g., generating only one type of car) and concentrates all probability mass there. This is diagnosed by monitoring sample diversity rather than the highly uninformative individual loss curves.
*   **CFG Computational Cost:** Because CFG requires evaluating the model twice per step (once for the conditional vector $v(x_t, t, y)$ and once for the unconditional vector $v(x_t, t, \emptyset)$), it doubles the inference latency.
*   **Intractable Backpropagation in VAEs:** Sampling $Z \sim q(Z \mid X)$ directly breaks backpropagation because sampling is a non-differentiable operation. This is resolved by using the **Reparameterization Trick**, which routes gradients through the deterministic parameters:
    $$Z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

##### B. Graduate-Level Reflection Questions
1.  **Divergence Limits of the Minimax Game:** Prove mathematically why a perfectly optimal discriminator $D^*_G(x)$ causes the gradient of the generator's objective to vanish when optimizing the original minimax game $\log(1 - D(G(z)))$. Show how the non-saturating generator loss $-\log D(G(z))$ resolves this.
2.  **Path Crossing in Rectified Flow:** Since the optimal velocity predictor in Rectified Flow learns to output the conditional expectation $\mathbb{E}[Z - X_0 \mid X_t]$ at intersecting path coordinates, explain why this leads to a "curving" trajectory at inference when using finite Euler steps. How do distillation algorithms address this to enable single-step generation?
3.  **The Multimodal Grounding Boundary:** Clip-like representations struggle with spatial compositionality (e.g., confusing "a mug in grass" with "grass in a mug"). If a latent diffusion model relies on a frozen CLIP text encoder for conditioning, explain how these compositionality errors propagate through the cross-attention layers of the Diffusion Transformer (DiT).
