# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 13: Generative Models 1

#### 0. Quick-Recall Summary
*   **The Unsupervised Paradigm Shift:** Unsupervised learning uncovers data distributions $P(X)$ without manual labels, bridging self-supervised pretext/contrastive representations (SimCLR, MoCo, DINO) with true density estimation.
*   **Generative Taxonomy:** Generative modeling splits into *explicit density* (tractable e.g., Autoregressive; approximate e.g., VAEs) and *implicit density* (direct sampling e.g., GANs; indirect iterative e.g., Diffusion).
*   **Autoregressive Chain Rule:** Autoregressive models factorize joint image probabilities into a 1D chain of conditional probabilities $P(X) = \prod\_{t=1}^T P(X_t | X\_{<t})$, casting image generation as high-dimensional, sequential next-pixel/token classification.
*   **ELBO Axiom:** Variational Autoencoders resolve the intractable true posterior $P(Z|X)$ by optimizing the Evidence Lower Bound (ELBO), balancing data reconstruction log-likelihood against latent space prior regularization.
*   **The Reparameterization Highway:** Isolating stochasticity by sampling random noise externally ($\epsilon \sim \mathcal{N}(0, I)$) enables deterministic gradient propagation through the VAE sampling bottleneck back to the encoder weights.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master the theoretical frameworks of unsupervised representation learning and explicit density generative architectures. This lecture details how neural networks model complex, high-dimensional probability distributions, transitioning from strict autoregressive next-value prediction to variational approximations that structure continuous latent spaces.
*   **Lecture Category:** Blend of (a) Mathematical Foundations (probabilistic density derivations, Bayes' rule relationships, ELBO formulation) and (d) Specific Vision Task (Unconditional and Conditional Generative Modeling).
*   **Builds on:** Lecture 12 (Self-Supervised Learning) by establishing how pretext tasks and contrastive learning representations (which align representations without explicit probabilities) connect to formal probabilistic density models.

---

#### 2. Mathematical Foundations

##### Probability Distribution Normalization & Competition
A valid probability density function $P(X)$ must satisfy the normalization constraint over its support:
$$P(X) \ge 0, \quad \int\_{X} P(X) \, dX = 1$$
This constraint introduces a zero-sum competition for probability mass: increasing the likelihood of one data point $X_a$ necessitates a reduction in density for other regions of the image space $X$.

##### Taxonomy of Probabilistic Frameworks
*   **Discriminative Models:** Model the conditional distribution $P(Y|X)$ of labels given high-dimensional data. Labels compete for mass for each fixed $X$; there is no competition across images.
*   **Unconditional Generative Models:** Model the raw data distribution $P(X)$. All possible images in the universe compete for a fixed unit of probability mass.
*   **Conditional Generative Models:** Model the distribution of high-dimensional data conditioned on an auxiliary signal $Y$ (e.g., text, labels): $P(X|Y)$.
*   **Bayesian Convergence:** These models are mathematically unified through Bayes' Rule:
$$P(X|Y) = \frac{P(Y|X)P(X)}{P(Y)}$$

##### Autoregressive Sequence Factorization
By applying the probability chain rule, the joint distribution of a high-dimensional image $X$ composed of $T$ subparts (pixels/tokens) is decomposed into a product of 1D conditional sequences:
$$P(X) = P(X_1, X_2, \dots, X_T) = \prod\_{t=1}^{T} P(X_t \mid X_1, X_2, \dots, X\_{t-1})$$
To avoid exponential growth in vocabulary state-space ($V^T$), autoregressive networks predict only the immediate next token $X_t$ conditioned on the historical prefix $X\_{<t}$.
*   **Log-Space Optimization:** To guarantee numerical stability under floating-point constraints, training maximizes the log-likelihood over a dataset of $N$ independent samples:
$$\max\_{\theta} \sum\_{i=1}^N \log P(X^{(i)}; \theta) = \max\_{\theta} \sum\_{i=1}^N \sum\_{t=1}^T \log P(X_t^{(i)} \mid X\_{<t}^{(i)}; \theta)$$

##### Variational Autoencoder (VAE) & ELBO Derivation
VAEs assume a generative process where continuous latent variables $Z$ are drawn from a simple prior $P(Z) = \mathcal{N}(0, I)$, and the observed data $X$ is sampled from $P(X|Z)$. Since the true posterior $P(Z|X) = \frac{P(X|Z)P(Z)}{P(X)}$ requires an intractable integral $\int_Z P(X|Z)P(Z)dZ$, a variational encoder $Q_\phi(Z|X)$ is introduced to approximate it.

The log-likelihood of data $\log P(X)$ is derived step-by-step:
1.  **Introduce Latent Marginalization & Encoder Approximation:**
    $$\log P(X) = \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log P(X) \right] = \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log \frac{P(X, Z)}{P(Z|X)} \right]$$
2.  **Multiply by Identity $\frac{Q_\phi(Z|X)}{Q_\phi(Z|X)}$ to introduce Variational Bounds:**
    $$\log P(X) = \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log \left( \frac{P(X, Z)}{Q_\phi(Z|X)} \cdot \frac{Q_\phi(Z|X)}{P(Z|X)} \right) \right]$$
3.  **Decompose Logarithm into Sum of Terms:**
    $$\log P(X) = \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log \frac{P(X, Z)}{Q_\phi(Z|X)} \right] + \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log \frac{Q_\phi(Z|X)}{P(Z|X)} \right]$$
4.  **Isolate the Intractable True Posterior as a KL Divergence:**
    $$\log P(X) = \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log \frac{P(X|Z)P(Z)}{Q_\phi(Z|X)} \right] + D\_{KL}\left( Q_\phi(Z|X) \parallel P(Z|X) \right)$$
5.  **Expand Joint Generative Probability:**
    $$\log P(X) = \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log P(X|Z) \right] - \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log \frac{Q_\phi(Z|X)}{P(Z)} \right] + D\_{KL}\left( Q_\phi(Z|X) \parallel P(Z|X) \right)$$
    $$\log P(X) = \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log P(X|Z) \right] - D\_{KL}\left( Q_\phi(Z|X) \parallel P(Z) \right) + D\_{KL}\left( Q_\phi(Z|X) \parallel P(Z|X) \right)$$
6.  **Formulate the Variational Lower Bound (ELBO):**
    Since KL Divergence is strictly non-negative ($D\_{KL}(\cdot \parallel \cdot) \ge 0$), dropping the intractable posterior term yields the lower bound:
    $$\log P(X) \ge \mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log P(X|Z) \right] - D\_{KL}\left( Q_\phi(Z|X) \parallel P(Z) \right)$$
    $$\text{ELBO}(\theta, \phi) = \underbrace{\mathbb{E}\_{Z \sim Q_\phi(Z|X)} \left[ \log P_\theta(X|Z) \right]}\_{\text{Reconstruction Term}} - \underbrace{D\_{KL}\left( Q_\phi(Z|X) \parallel P(Z) \right)}\_{\text{Prior Regularization Term}}$$

##### The Reparameterization Trick Formulation
Standard sampling of a latent vector $Z \sim Q_\phi(Z|X) = \mathcal{N}(\mu_\phi(X), \Sigma_\phi(X))$ is a stochastic node that blocks backpropagation. VAEs reparameterize the sampling process by isolating the stochasticity:
$$Z = \mu_\phi(X) + \sigma_\phi(X) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
where $\odot$ denotes the element-wise Hadamard product. The gradients can now flow deterministically through the differentiable mappings of $\mu$ and $\sigma$:
$$\frac{\partial Z}{\partial \mu_\phi} = 1, \quad \frac{\partial Z}{\partial \sigma_\phi} = \epsilon$$

---

#### 3. Architecture / Algorithm Walkthrough

##### Data-Flow Routing
*   **Autoregressive Models (PixelRNN/PixelCNN):**
    ```
    [Input Pixels X_<t] ──> [RNN / Causally Masked Conv] ──> [Softmax Logits over] ──> [Negative Log Likelihood Loss]
    ```
*   **Variational Autoencoder (VAE) Training Loop:**
    ```
    [Input Image X] ──> [Encoder Net Q_phi] ──> [Predict mean (mu) & variance (log_var)]
                                                        │
                                                        ├──> [Compute Prior KL Loss]
                                                        │
                      [Auxiliary Noise epsilon] ──> [Reparameterization Trick Z = mu + exp(0.5*log_var)*epsilon]
                                                        │
                                                        └──> [Decoder Net P_theta] ──> [Reconstructed Image X_hat]
                                                                                            │
                                                                                            └──> [Compute Reconstruction L2 Loss]
    ```

##### PyTorch Blueprint (Variational Autoencoder with Reparameterization)
This blueprint implements a standard VAE training pipeline, using continuous Gaussian latent distributions and sharded loss calculations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    # Probabilistic Variational Autoencoder (VAE) featuring Gaussian latent modeling
    # and the reparameterization trick for stable gradient flow.
    def __init__(self, input_dim=3072, hidden_dim=512, latent_dim=128):
        super(VariationalAutoencoder, self).__init__()
        # Encoder bottleneck pathway
        self.encoder_fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim) # Predicts log(sigma^2) for stability
        
        # Decoder reconstruction pathway
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.fc_recon = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_fc(x))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Reparameterization trick: isolates stochasticity to external node epsilon.
        std = torch.exp(0.5 * log_var) # std = sqrt(exp(log_var))
        eps = torch.randn_like(std)    # sample epsilon ~ N(0, I)
        return mu + std * eps          # differentiable latent sample

    def decode(self, z):
        h = F.relu(self.decoder_fc(z))
        # Reconstruct pixel mean; output bounded/unbounded based on likelihood assumption
        return torch.sigmoid(self.fc_recon(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

def loss_function_elbo(x_recon, x, mu, log_var):
    # Computes the negative Evidence Lower Bound (-ELBO).
    # Assuming Gaussian decoder likelihood with identity covariance, 
    # reconstruction loss simplifies to Mean Squared Error (L2 loss).
    # Reconstruction term: log P(X|Z)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # Prior regularization term: D_KL(Q(Z|X) || N(0, I))
    # Closed form KL divergence for diagonal Gaussians
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss to minimize: negative ELBO
    return recon_loss + kl_loss
```

---

#### 4. Visual Intuition & Interpretability

##### The Latent Tension Tug-of-War
The VAE objective function functions as an equilibrium balance between two opposing optimization forces in visual representation:
*   **What Reconstruction Wants:** To drive variance to zero ($\sigma \rightarrow 0$) and map every image $X^{(i)}$ to an isolated, deterministic delta peak latent coordinate $Z^{(i)}$. This allows perfect pixel memorization but fractures the latent space into discrete, non-interpolatable points.
*   **What Prior KL Regularization Wants:** To collapse all latent projections directly onto the origin ($\mu \rightarrow 0, \sigma \rightarrow 1$). This ensures a uniform Gaussian prior but destroys all discriminative reconstruction features, reducing the latent space to complete randomness.
*   **The Equilibrium:** The optimization results in a continuous, smooth latent manifold where coordinates represent semantic attributes (e.g., stroke thickness, tilt) allowing continuous visual morphing.

##### Interpolating Latent Spaces Visualized
Moving along paths in the continuous Gaussian latent space generates smooth, semantically consistent image transformations (e.g., digits morphing smoothly from a '0' to a '9' without visual breaks).

```
[Latent coordinate Z_0: "thick 0"] ──> [Mating Path] ──> [Interpolated Coordinate: "tilted 8"] ──> [Latent coordinate Z_1: "skinny 9"]
```

---


<div id="plotly-cs231n-13-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 13 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)

##### Interactive VAE Latent Landscape Visualizer
*   **Visualization Type:** 2D Latent Grid Scatter Plot paired with a Dynamic Decoder Image Reconstruction Window.
*   **Data Fields & Encoding:**
    *   **Main 2D Plot:** Projects a 2-dimensional slice of the latent space ($Z_1, Z_2$). Colored markers represent different digit classes (MNIST '0'-'9').
    *   **Image Reconstruction Box:** Renders the 2D pixel reconstruction output of the decoder from the coordinate of a draggable crosshair pointer.
*   **Interactive Controls:**
    *   **Prior KL Weight Slider ($\beta$):** Controls the scale coefficient of the KL divergence regularization term.
        *   *Setting $\beta = 0$:* Shows the scatter plot collapse into highly isolated, scattered clusters (memorized templates, zero interpolation capacity).
        *   *Setting $\beta \gg 1$:* Shows the different classes merge completely into a single overlapping Gaussian cloud, where the decoder reconstructs only blurry mean artifacts.
    *   **2D Crosshair Drag:** Dragging the cursor smoothly across the latent space morphs the image output in the reconstruction box, revealing hidden semantic axes (e.g., tilt angle, width).

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **Autoregressive Bottlenecks:** Naive rasterization of 1024x1024 RGB images generates sequences of **3,000,000 values**. Autoregressive sequence execution (PixelRNN/PixelCNN) scales quadratically with sequence length, making real-time, high-resolution generation computationally intractable without tokenization bottlenecks.
*   **DINOv2 Foundation Scaling:** Self-supervised pre-training was scaled from ImageNet-1M to **142 million unlabelled images** to provide robust, zero-shot visual features for transfer learning.
*   **Historical Sample Quality Milestone:** Prior to the scaling of deep latent VAEs and diffusion, GANs were highly favored because VAEs suffered from a structural ceiling, characteristically generating blurry reconstructions due to pixel-level Gaussian likelihood assumptions ($L_2$ pixel optimization).

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas
*   **Intractable Posterior Trap:** Attempting to optimize VAEs using direct maximum likelihood requires evaluating $\int_Z P(X|Z)P(Z)dZ$, which is computationally impossible over the parameter space of deep neural networks, making variational bounds (ELBO) mathematically mandatory.
*   **Diagonal Covariance Assumption:** VAE decoders model diagonal covariances (per-pixel variances) to avoid $H^2 W^2$ full covariance matrices. However, sampling directly from diagonal covariances amounts to adding independent pixel-level white noise, which destroys structural correlations.
*   **Autoregressive Sequence Collapse:** Training autoregressive image models with raw 1D pixel strings ignores the 2D spatial context. Modern architectures instead use discrete tokenizers (VQ-VAE / VQ-GAN) to cast the image into compact 1D latent token arrays before autoregression.

##### Graduate-Level Reflection Questions
1.  **Symmetry in VAE Objectives:** In VAE optimization, why does modeling the decoder output as a Gaussian distribution with an identity covariance matrix ($\Sigma = I$) reduce the reconstruction objective mathematically to Mean Squared Error ($L_2$ pixel loss)? What occurs to the optimization dynamics if the decoder is also tasked with predicting individual pixel-level variances?
2.  **The Autoregressive vs. Variational Trade-off:** Autoregressive models compute exact probability densities $P(X)$ but exhibit sequential inference bottlenecks. VAEs output approximate lower-bound densities but draw samples in a single feed-forward pass. Prove why autoregressive likelihood calculation is strictly sequential, and analyze why VAEs are able to bypass this sequential barrier.
3.  **Bayesian Posterior Collapse:** During VAE training with powerful decoders (such as deep causally-masked Transformers), the prior KL divergence regularization term often collapses to zero ($D\_{KL} \rightarrow 0$), causing the encoder to ignore the input $X$ entirely. Explain why a high-capacity decoder encourages this collapse, and outline how the reparameterization trick fails to prevent it.
