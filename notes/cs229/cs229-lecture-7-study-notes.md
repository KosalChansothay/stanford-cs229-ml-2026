# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 7: Neural Networks 1 (Architecture)

### 1. Summary
This lecture introduces the foundational architecture of deep learning and neural networks. Shifting from linear and affine models to parameterized nonlinear models, the lecture establishes a mathematically rigorous framework for supervised learning under nonlinear regimes. It defines the concepts of continuous regression and multi-class classification (using Softmax and Cross-Entropy loss). The discussion then dives deep into why classical second-order optimization methods (like Newton's Method) fail under non-convexity, setting up Stochastic Gradient Descent (SGD) and Minibatch SGD as the workhorses of modern machine learning. Finally, the lecture constructs neural networks from the ground up—explaining the single neuron, activation functions (such as ReLU, Sigmoid, tanh, and GELU), Multi-layer Perceptrons (MLPs) via vectorized matrix multiplications, Residual Connections (ResNets), Layer Normalization (including RMS Norm), and Convolutional Neural Networks (CNNs).

---

### 2. Key Concepts & Definitions
- **Nonlinear Model**: A model that is non-linear in its parameters ($\theta$), making it fundamentally different from models that are merely non-linear in the input features (e.g., polynomial features), which can be easily re-parameterized into a linear system.
- **Logits**: The raw, unnormalized $k$-dimensional outputs of a multi-class neural network before they are passed through a softmax function to produce probabilities.
- **Activation Function**: A scalar function $\sigma: \mathbb{R} \to \mathbb{R}$ applied entrywise to vector-valued linear combinations, introducing non-linear expressive power to the network.
- **Multi-layer Perceptron (MLP)**: An feedforward artificial neural network architecture consisting of multiple stacked layers of matrix multiplications followed by entrywise non-linear activation functions.
- **Residual Block**: A network component that adds the input vector directly back to the output of one or more intermediate parameterized layers. This allows the network to explicitly model the *residual* or error correction term.
- **Scaling Invariance**: A property of normalization layers (like LayerNorm or RMSNorm) where scaling the input vector by any constant $\alpha > 0$ yields the exact same output, preventing exponential growth of activations across deep networks.
- **Convolution**: A specialized matrix multiplication where the weight matrix has a structured Toeplitz-like format, restricting nonzeros to local receptive fields and sharing parameters to enforce shift invariance.

---

### 3. Mathematical Formulations & Derivations

#### A. Supervised Learning under Nonlinear Regimes
In a supervised learning setup with dataset $\{(x^{(i)}, y^{(i)})\}\_{i=1}^n$, where $x^{(i)} \in \mathbb{R}^d$ and parameters $\theta$ represent the weights of a non-linear hypothesis $h_\theta(x)$:
- **Continuous Regression**: The targets $y^{(i)} \in \mathbb{R}$. The standard squared error loss for a single example is:
  $$\mathcal{L}(\theta) = \left(y - h_\theta(x)\right)^2$$
- **Multi-class Classification**: The targets $y^{(i)} \in \{1, 2, \dots, k\}$. The hypothesis outputs a vector of $k$ logits, $h_\theta(x) \in \mathbb{R}^k$.

#### B. The Softmax Function
To map the logits to a valid probability distribution on the simplex (where values are non-negative and sum to 1), we apply the softmax activation entrywise:
$$P(y = j \mid x; \theta) = \frac{e^{h\_{\theta, j}(x)}}{\sum\_{l=1}^k e^{h\_{\theta, l}(x)}}$$
where $h\_{\theta, j}(x)$ denotes the $j$-th element of the logit vector.

#### C. Derivation of Cross-Entropy Loss
Under a maximum likelihood estimation framework, we minimize the Negative Log-Likelihood (NLL) of the correct class $y$:
$$\mathcal{L}\_{\text{CE}}(\theta) = -\log P(y \mid x; \theta)$$
Substituting the softmax formulation:
$$\mathcal{L}\_{\text{CE}}(\theta) = -\log \left( \frac{e^{h\_{\theta, y}(x)}}{\sum\_{l=1}^k e^{h\_{\theta, l}(x)}} \right)$$
Applying logarithmic identities yields the familiar **Cross-Entropy Loss** formula:
$$\mathcal{L}\_{\text{CE}}(\theta) = -h\_{\theta, y}(x) + \log \sum\_{l=1}^k e^{h\_{\theta, l}(x)}$$
*Concept Note*: This is mathematically equivalent to the Kullback-Leibler (KL) divergence or cross-entropy between the true distribution (represented as a one-hot label vector) and the model's predicted probability distribution.

---

### 4. Step-by-Step Optimization & Hardware Realities

Under non-linear models, the loss landscape $\mathcal{L}(\theta)$ is highly non-convex. 

#### Why Newton's Method Fails
In linear/convex regimes, second-order optimization methods like Newton's Method converge rapidly. However, on non-convex landscapes, Newton's Method is not guaranteed to find a local minimum. At a stationary point where the gradient is $\nabla_\theta \mathcal{L}(\theta) = 0$, a non-convex function can easily land on a saddle point or local maximum unless the Hessian matrix $H$ is guaranteed to be positive semi-definite (PSD) in all directions.

#### The Stochastic Gradient Descent Alternative
Stochastic Gradient Descent (SGD) and Minibatch SGD are the universal optimizers for deep learning:
1. **Initialize** parameters $\theta$.
2. **For each training iteration**:
   - Sample a minibatch $\mathcal{B}$ of size $B$ uniformly from the dataset.
   - Compute the average gradient of the loss over the batch:
     $$\nabla_\theta \mathcal{L}\_{\mathcal{B}}(\theta) = \frac{1}{B} \sum\_{i \in \mathcal{B}} \nabla_\theta \mathcal{L}_i(\theta)$$
   - Update the parameters along the steepest descent direction:
     $$\theta := \theta - \eta \nabla_\theta \mathcal{L}\_{\mathcal{B}}(\theta)$$
     where $\eta$ is the learning rate.

#### Modern Scaling and GPU Constraints
- **Computational Cost**: In 2026, datasets scale up to 1 trillion tokens. Computing the full gradient over $n$ examples is strictly impossible. Minibatching provides an unbiased estimator of the full gradient ($\mathbb{E}[\nabla_\theta \mathcal{L}\_{\mathcal{B}}(\theta)] = \nabla_\theta \mathcal{L}(\theta)$) at a fraction of the cost.
- **Correlation Redundancy**: Individual training examples are highly correlated, making full batch calculations mathematically wasteful.
- **GPU Hardware Underutilization**: Setting the batch size $B=1$ (pure SGD) is too noisy and results in low GPU FLOPs utilization. Modern accelerators rely on massive parallelization; a larger minibatch size $B$ keeps multiple streaming multiprocessors occupied, maximizing training speed.

<div id="plotly-sgd-unbiased" class="plotly-chart" aria-label="Interactive Plotly chart: stochastic gradient vectors clustering around the true full-batch gradient, demonstrating unbiasedness"></div>

<p><em>Figure: The Gradient Approximation — each blue arrow is a single-example stochastic gradient; their centroid (hollow marker) aligns with the black full-batch gradient arrow, visualizing why minibatch SGD is an unbiased estimator: $\mathbb{E}[\nabla_\theta \mathcal{L}\_{\mathcal{B}}(\theta)] = \nabla_\theta \mathcal{L}(\theta)$.</em></p>

---

### 5. Neural Network Building Blocks & Deep Architectures

#### A. The Single Neuron
The elementary building block of a network takes an input vector $x \in \mathbb{R}^d$, multiplies it by a weight vector $w \in \mathbb{R}^d$, adds a scalar bias $b \in \mathbb{R}$, and applies a non-linear activation $\sigma$:
$$a = \sigma(w^T x + b)$$

#### B. Primary Activation Functions
- **Rectified Linear Unit (ReLU)**: Inspired by biological neuron firing thresholds.
  $$\text{ReLU}(t) = \max(t, 0)$$
- **Sigmoid**: Historically popular but computationally prone to vanishing gradients.
  $$\sigma(t) = \frac{1}{1 + e^{-t}}$$
- **Hyperbolic Tangent (tanh)**: Maps values to $(-1, 1)$.
  $$\tanh(t) = \frac{e^t - e^{-t}}{e^t + e^{-t}}$$
- **Leaky ReLU**: Retains a tiny slope on the negative domain to prevent dead neurons.
  $$\text{LeakyReLU}(t) = \max(\alpha t, t) \quad (0 < \alpha \ll 1)$$
- **Gaussian Error Linear Unit (GELU)**: A smooth non-linear function used in contemporary transformers.
  $$\text{GELU}(t) = t \cdot \Phi(t)$$
  where $\Phi(t)$ is the standard normal cumulative distribution function. Unlike ReLU, GELU dips slightly below 0 on the negative domain.

#### C. Multi-layer Perceptron (MLP) Layer-by-Layer Vectorization
To compute an entire hidden layer of $m$ neurons efficiently, we group the weight vectors into a matrix $W^{(1)} \in \mathbb{R}^{m \times d}$ and the biases into a vector $b^{(1)} \in \mathbb{R}^m$:
$$a^{(1)} = \sigma\left(W^{(1)} x + b^{(1)}\right)$$
For a deep feedforward network, we recursively apply this operation across layers $l \in \{1, 2, \dots, L\}$:
$$a^{(l)} = \sigma\left(W^{(l)} a^{(l-1)} + b^{(l)}\right)$$
where $a^{(0)} = x$ and the output of the final layer represents the logits.

#### D. Residual Connections (ResNets)
To mitigate vanishing gradients and enable optimization of extremely deep models, residual blocks bypass linear layers:
$$\text{Res}(z) = \sigma\left(W^{(2)} \sigma\left(W^{(1)} z + b^{(1)}\right) + b^{(2)}\right) + z$$
*Intuition*: Instead of forcing the network layers to model the mapping $y \approx f(z)$, the layers only model the *residual mapping* $F(z) = f(z) - z$. If $z$ is already a decent approximation, modeling the correction is significantly easier and stabilizes the optimization Hessian eigenvalues.

<div id="plotly-residual-block" class="plotly-chart" aria-label="Interactive Plotly diagram: a residual block with skip connection, hover each node for details"></div>

<p><em>Figure: Residual Connection — the input $z$ splits into a transform path (Weight → Activation → Weight) and a shortcut path; the two meet at a $+$ node, giving $\text{Res}(z) = F(z) + z$. Hover each node for its role.</em></p>

---

### 6. Layer Normalization

#### A. Traditional LayerNorm
Layer Normalization normalizes the activations of a single layer across its hidden feature dimensions $m$ for an individual training example.
For a layer vector $z \in \mathbb{R}^m$:
1. **Compute Empirical Mean**:
   $$\hat{\mu} = \frac{1}{m} \sum\_{i=1}^m z_i$$
2. **Compute Empirical Variance**:
   $$\hat{\sigma}^2 = \frac{1}{m} \sum\_{i=1}^m (z_i - \hat{\mu})^2$$
3. **Normalize and Apply Learnable Parameters ($\gamma, \beta \in \mathbb{R}^m$)**:
   $$\text{LN}(z) = \gamma \odot \left( \frac{z - \hat{\mu}}{\hat{\sigma} + \epsilon} \right) + \beta$$
   where $\odot$ represents entrywise multiplication, and $\epsilon$ is a small constant for numerical stability.

#### B. RMS Norm (Root Mean Square Normalization)
Modern models (like LLaMA and other state-of-the-art transformers) replace LayerNorm with the computationally simpler RMS Norm:
1. **Compute Root Mean Square**:
   $$\hat{\sigma} = \sqrt{\frac{1}{m} \sum\_{i=1}^m z_i^2}$$
2. **Normalize and Scale**:
   $$\text{RMSNorm}(z) = \gamma \odot \left( \frac{z}{\hat{\sigma} + \epsilon} \right)$$
*Note*: RMS Norm eliminates mean subtraction, saving crucial GPU cycles while achieving comparable empirical performance.

#### C. Mathematical Properties: Scaling Invariance
A core strength of LayerNorm and RMSNorm is **scaling invariance**. For any scaling factor $\alpha > 0$:
$$\text{LN}(\alpha z) = \text{LN}(z)$$
- **Invariance Benefit**: This prevents intermediate network activations from compounding exponentially (exploding to $10^{20}$) or vanishing to 0 during deep propagation, simplifying initialization.
- **Optimization Caveat**: Although the forward pass is scaling invariant, the gradients are *not* scaling invariant, creating complex, self-stabilizing optimization dynamics.

---

### 7. Applications
- **Large Language Models (LLMs)**: Under the hood of models like GPT-4 or Claude, every single token predicted is chosen by calculating a softmax over a massive multi-class vocabulary (typically 50,000 to 250,000 target classes) with Cross-Entropy Loss.
- **Computer Vision (Transformers and CNNs)**: High-resolution RGB images are mapped to lower-dimensional latent representations using sequential MLP/CNN blocks and normalized using LayerNorm/RMSNorm to stabilize training.

---

### 8. Reflection Questions
1. **Linearity vs. Nonlinearity**: Explain why a network constructed as $h_\theta(x) = W^{(2)}\left(W^{(1)}x + b^{(1)}\right) + b^{(2)}$ without any activation function $\sigma$ collapses back into a simple linear model. Prove this algebraically by finding an equivalent single-layer weight matrix $\tilde{W}$ and bias vector $\tilde{b}$.
2. **Newton's Method under Non-convexity**: Why does the existence of negative eigenvalues in the Hessian matrix $\nabla^2 \mathcal{L}(\theta)$ of a non-convex loss function cause standard Newton's Method to fail? How does Minibatch SGD circumvent this failure mode?
3. **The LayerNorm Invariance Trade-off**: Explain the mathematical mechanism behind the scaling invariance property of LayerNorm ($\text{LN}(\alpha z) = \text{LN}(z)$). What would happen to the optimization process of a 100-layer network if all LayerNorm blocks were removed?

---

### 9. Further Reading & Resources
- **He et al. (2015) "Deep Residual Learning for Image Recognition"**: The seminal paper introducing the ResNet architecture and residual bypass connections.
- **"Root Mean Square Layer Normalization" (Bizanis et al.)**: The reference paper proving the efficiency and optimization benefits of RMS Norm over traditional LayerNorm.
- **Deep Learning Book (Goodfellow, Bengio, and Courville)**: Chapters 6 (Deep Feedforward Networks) and 8 (Optimization for Training Deep Models) are highly recommended for understanding multi-layer backpropagation.
