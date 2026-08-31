# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 11: Diffusion Models

### 1. Summary
This lecture introduces **Diffusion Models**, which have emerged as the state-of-the-art paradigm for generative modeling in computer vision, robotics, and sequential action modeling, eclipsing historical frameworks like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). The instructor breaks down the core probabilistic formulation of diffusion, which consists of a **known Forward (Noising) Process** that iteratively perturbs data into standard Gaussian noise, and a **learned Reverse (Denoising) Process** modeled as a Markov chain parameterized by a deep neural network. The lecture connects diffusion to the EM algorithm and variational inference by deriving the **Evidence Lower Bound (ELBO)**. By applying the chain rule of KL divergence, the joint trajectory optimization is decomposed into tractable, step-by-step Gaussian mean-matching losses. Finally, the lecture presents a continuous-time perspective of diffusion using Stochastic Differential Equations (SDEs) to formally justify the Gaussian parameterization of the reverse process.

---

### 2. Key Concepts & Definitions
- **Generative Modeling**: The task of learning a model $p_\theta(x)$ over a data distribution $p\_{\text{data}}$ given empirical samples, such that we can generate novel, realistic samples from the learned distribution.
- **Forward (Noising) Process ($q$)**: A fixed Markov chain that iteratively adds small amounts of Gaussian noise to a clean data point $x_0 \sim p\_{\text{data}}$ over $T$ steps, eventually transforming it into pure white noise $x_T \sim \mathcal{N}(0, I)$.
- **Reverse (Denoising) Process ($p\_\theta$)**: A learned Markov chain that starts from standard Gaussian noise $x\_T$ and reconstructs clean data step-by-step by predicting and removing the added noise.
- **Score Function**: The gradient of the log probability density with respect to the data, $\nabla\_x \log p\_t(x)$. In continuous-time diffusion, the reverse drift relies heavily on this score function to steer the noisy trajectory back toward the high-density regions of the data manifold.
- **Evidence Lower Bound (ELBO)**: A mathematical surrogate used to maximize data likelihood $\log p_\theta(x_0)$ by introducing a variational posterior distribution over latent variables (here, the noisy intermediate steps $x\_{1:T}$).
- **Markovian Decomposition**: The property of Markov chains where the future state is conditionally independent of the past given the current state. This allows diffusion models to decompose a highly non-trivial denoising task into a sequence of local, easily learnable, single-step transitions.

---

### 3. Mathematical Formulations & Derivations

#### A. The Forward Noising Process
Starting with a clean image $x\_0 \in \mathbb{R}^d$, we define the forward transition at step $t \in \{1, \dots, T\}$ by scaling the previous step and adding independent and identically distributed (IID) Gaussian noise:

$$
x_t = \sqrt{1 - \beta_t} x\_{t-1} + \sqrt{\beta_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$
where $\beta\_t \in (0, 1)$ is a predefined noise variance schedule (typically very small, e.g., $10^{-4}$ to $10^{-2}$).

##### Why choose these specific coefficients?
Consider the covariance of $x_t$ under this transition rule. Because the noise $\epsilon_t$ is independent of $x\_{t-1}$, their covariances are additive:

$
\text{Cov}(x_t) = (1 - \beta_t) \text{Cov}(x\_{t-1}) + \beta_t \text{Cov}(\epsilon_t) = (1 - \beta_t) \text{Cov}(x\_{t-1}) + \beta_t I$

This represents a **linear interpolation (convex combination)** of the previous covariance and the identity matrix $I$. This formulation prevents the variance of the activations from exploding during the forward pass. As $t \to \infty$, the covariance asymptotically converges to identity, meaning the distribution of the final latent variable $x\_T$ converges to a standard normal Gaussian:


<div id="plotly-11-variance-schedule" class="plotly-chart" aria-label="Interactive Plotly chart: Variance stability of the noising schedule"></div>

<p><em>Figure: Variance of x_t with and without the sqrt(1-beta) scaling — the convex combination keeps variance bounded and converging to 1.</em></p>
$$x\_T \sim \mathcal{N}(0, I)$$$

#### B. Closed-Form Direct Marginalization
Instead of iteratively sampling $t$ steps to generate a noisy image $x\_t$, we can express the distribution of $x\_t$ directly conditioned on $x\_0$. Let $\alpha\_t = 1 - \beta\_t$ and define the cumulative product of scaling factors as:

$$
\bar{\alpha}_t = \prod\_{s=1}^t \alpha_s
$$

Unrolling the recursive definition of $x\_t$ yields:

$$
x_t = \sqrt{\alpha_t} x\_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t
$$
$$
x_t = \sqrt{\alpha_t} \left( \sqrt{\alpha\_{t-1}} x\_{t-2} + \sqrt{1 - \alpha\_{t-1}} \epsilon\_{t-1} \right) + \sqrt{1 - \alpha_t} \epsilon_t
$$

By recursively substituting and leveraging the fact that the sum of independent Gaussians is also Gaussian, we simplify the unrolled equation to:

$
x\_t = \sqrt{\bar{\alpha}\_t} x\_0 + \sqrt{1 - \bar{\alpha}\_t} \bar{\epsilon}, \quad \bar{\epsilon} \sim \mathcal{N}(0, I)$

This allows us to express the conditional distribution $q(x\_t \mid x\_0)$ in closed form as:

$
q(x\_t \mid x\_0) = \mathcal{N}(x\_t; \sqrt{\bar{\alpha}\_t} x\_0, (1 - \bar{\alpha}\_t)I)$

Since $\bar{\alpha}\_t \to 0$ as $t \to \infty$ (due to multiplying numbers less than 1), $\sqrt{\bar{\alpha}\_t} \to 0$ and $\sqrt{1 - \bar{\alpha}\_t} \to 1$, ensuring that $q(x\_T \mid x\_0)$ converges cleanly to $\mathcal{N}(0, I)$.

#### C. The Parameterized Reverse Process
Because the true reverse process $q(x\_{t-1} \mid x_t)$ is unknown and analytically intractable without knowing the entire data distribution, we approximate it using a parameterized model $p_\theta$:

$
p_\theta(x\_{0:T}) = p(x_T) \prod\_{t=1}^T p_\theta(x\_{t-1} \mid x_t)$

We parameterize each reverse transition step as a Gaussian distribution:

$
p_\theta(x\_{t-1} \mid x_t) = \mathcal{N}(x\_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$
where $\mu\_\theta(x\_t, t)$ is represented by a deep neural network (e.g., an MLP or U-Net) that predicts the mean given the noisy input $x\_t$ and the current timestep $t$, and $\sigma\_t^2$ is typically chosen as a fixed variance schedule (such as $\beta\_t$ or $\tilde{\beta}\_t$).

---

### 4. Step-by-Step Optimization & Training Loss Derivation

To train the parameters $\theta$, we seek to maximize the log-likelihood of the observed data, $\log p_\theta(x_0)$. Using the intermediate noisy states $x\_{1:T}$ as latent variables $z$, we formulate the Evidence Lower Bound (ELBO):

$$
\log p_\theta(x_0) \ge \mathbb{E}\_{q(x\_{1:T} \mid x_0)} \left[ \log \frac{p_\theta(x\_{0:T})}{q(x\_{1:T} \mid x_0)} \right]
$$

#### Decomposing the ELBO
By substituting the factorization of $p_\theta(x\_{0:T})$ and $q(x\_{1:T} \mid x_0)$ and applying the chain rule of KL divergence recursively, the joint trajectory KL term simplifies into a sum of step-wise KL terms:

$$
\log p_\theta(x_0) \ge \mathbb{E}\_q \left[ \log p_\theta(x_0 \mid x_1) \right] - \sum\_{t=2}^T \mathbb{E}\_{q(x_t \mid x_0)} \left[ D\_{\text{KL}}(q(x\_{t-1} \mid x_t, x_0) \parallel p_\theta(x\_{t-1} \mid x_t)) \right] - D\_{\text{KL}}(q(x_T \mid x_0) \parallel p(x_T))
$$

- **The Intrinsic Insight**: Reconstructing $x\_{t-1}$ from $x_t$ directly is exceptionally difficult without knowing what clean image we started with. However, if we condition the reverse step on the clean image $x_0$, the forward posterior $q(x\_{t-1} \mid x_t, x_0)$ becomes highly tractable and easy to compute analytically using Bayes' rule.
- **Matching the Surrogate**: The ELBO training objective forces our model $p_\theta(x\_{t-1} \mid x_t)$ to match the tractable, $x_0$-conditioned posterior distribution $q(x\_{t-1} \mid x_t, x_0)$.

#### Analytical Derivation of the Forward Posterior
Using Bayes' rule, we expand the tractable posterior:

$
q(x\_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x\_{t-1}, x_0) q(x\_{t-1} \mid x_0)}{q(x_t \mid x_0)}$

Since $x\_{1:T}$ is a Markov chain, $q(x_t \mid x\_{t-1}, x_0) = q(x_t \mid x\_{t-1})$. This term is simply the single-step forward transition $\mathcal{N}(x_t; \sqrt{\alpha_t}x\_{t-1}, \beta_t I)$. The other terms are the direct marginals $q(x\_{t-1} \mid x_0)$ and $q(x_t \mid x_0)$. 

Plugging in their respective Gaussian density functions and simplifying the quadratic terms inside the exponential reveals that $q(x\_{t-1} \mid x_t, x_0)$ is also a Gaussian distribution:

$
q(x\_{t-1} \mid x_t, x_0) = \mathcal{N}(x\_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$
where the mean $\tilde{\mu}\_t(x\_t, x\_0)$ is a linear combination of $x\_t$ and $x\_0$ with coefficients derived from the cumulative scaling products:

$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}\_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}\_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0
$$
and the posterior variance $\tilde{\beta}\_t$ is:

$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$
#### Formulating the Mean-Matching Loss
We choose the model's reverse variance $\sigma_t^2$ to match the posterior variance $\tilde{\beta}_t$. Under this choice, the KL divergence between the two Gaussians $q(x\_{t-1} \mid x_t, x_0)$ and $p_\theta(x\_{t-1} \mid x_t)$ reduces elegantly to the squared $L_2$ distance of their means:

$$
D\_{\text{KL}}(q(x\_{t-1} \mid x_t, x_0) \parallel p_\theta(x\_{t-1} \mid x_t)) = \frac{1}{2\tilde{\beta}\_t} \| \mu_\theta(x_t, t) - \tilde{\mu}_t(x_t, x_0) \|^2
$$

Thus, training a diffusion model boils down to solving a sequence of simple, weighted **least-squares regression problems** to minimize the mean-matching loss:

$$
\mathcal{L}\_{t-1}(\theta) = \frac{1}{2\tilde{\beta}\_t} \| \mu_\theta(x_t, t) - \tilde{\mu}_t(x_t, x_0) \|^2
$$

<div id="plotly-11-diffusion-trajectory" class="plotly-chart" aria-label="Interactive Plotly chart: Forward noising paths and the reverse posterior mean alignment"></div>

<p><em>Figure: Forward noising paths and the reverse posterior mean alignment.</em></p>

---

### 5. Continuous-Time Perspective: Stochastic Differential Equations (SDEsTo theoretically justify why both the forward and reverse processes can be modeled as Gaussian transitions, the lecture introduces a continuous-time framework.

#### The Forward SDE
If we let the step size $\Delta t \to 0$ and the number of steps $T \to \infty$, the forward noising process can be formally modeled as a Stochastic Differential Equation (SDE):

$
dx_t = f(x, t) dt + g(t) dw_t
$$
where $f(x, t)$ is a deterministic drift term representing the shrinking of the image, $g(t)$ represents the continuous noise scaling $\sqrt{\beta\_t}$, and $dw\_t$ is a standard Brownian motion (Wiener process).

#### The Reverse SDE (Anderson, 1985)
A fundamental theorem in stochastic calculus (Anderson, 1985) proves that if the forward process follows the SDE above, then the time-reversed process $y_\tau = x\_{T-\tau}$ is also a diffusion process that satisfies the following SDE:

$$
dy_\tau = \left[ -f(y, T-\tau) + g^2(T-\tau) \nabla_y \log p\_{T-\tau}(y) \right] d\tau + g(T-\tau) d\bar{w}_\tau
$$

where $d\bar{w}\_\tau$ represents Brownian motion flowing backward in time, and $\nabla\_y \log p\_t(y)$ is the **score function** of the distribution at time $t$.

##### Theoretical Takeaways:
1. **Mathematical Justification**: Because any stochastic process obeying these smooth regularity conditions yields a Gaussian SDE in the continuous limit, parameterizing the discrete reverse transitions as Gaussians is mathematically rigorous.
2. **The Score Role**: The drift term in the reverse SDE consists of the original forward drift $f$ modified by the score function $\nabla \log p\_t(y)$. This score function acts as a guide vector, showing how to adjust the noisy values to move toward high-density regions of the true data distribution.

---

### 6. Applications
- **High-Resolution Image Generation**: Generating photorealistic natural images (e.g., ImageNet, face generation, or celebrity datasets) starting from pure white noise, surpassing the sample quality of VAEs and GANs.
- **Vision-Language-Action (VLA) Models for Robotics**: In modern robotics, VLAs utilize diffusion models to model continuous and complex multi-modal action trajectories, allowing robots to perform precise physical manipulations in real-world environments.
- **Parallel Text Generation**: Exploiting continuous-space text diffusion to generate linguistic sequences in parallel rather than autoregressively, significantly accelerating inference and scaling performance.

---

### 7. Reflection Questions
1. **Linear Interpolation vs. Standard Noise**: Why do we scale the previous image $x\_{t-1}$ by $\sqrt{1-\beta_t}$ instead of just adding noise directly as $x_t = x\_{t-1} + \sqrt{\beta_t}\epsilon_t$? What mathematical catastrophe would occur to the variance of our latent representations as $t \to \infty$ if we did not scale down the signal?
2. **The Role of $x_0$ in the Reverse Pass**: Explain why reconstructing the cleaner image $x\_{t-1}$ from the noisier image $x_t$ is highly challenging in a direct model, but becomes analytically simple when conditioning on the original clean image $x_0$. How does Bayes' rule exploit this conditional dependency?
3. **Continuous vs. Discrete Denoising**: Why does stretching the chain to a very large number of steps (e.g., $T = 1000$) make the Gaussian transition assumption more accurate? Contrast this continuous-time theoretical limit with modern engineering constraints in product deployment.

---

### 8. Further Reading & Resources
- **Shao-Hua Sun’s Stanford Lecture Notes on Diffusion**: Provides complete multi-page algebraic proofs of the unrolled marginal $q(x\_t \mid x\_0)$ and the Bayesian posterior expansions.
- **Anderson (1985) Reverse-Time SDEs**: The seminal mathematical paper establishing the continuous-time SDE reverse formulation and the role of the score function.
- **Prof. Stefano Ermon’s Research Papers**: Cutting-edge work on continuous score-based generative modeling, fast ODE solvers for diffusion, and continuous text-diffusion startups.
