# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 4: Exponential Family, GLMs & Classification

### 1. Summary
This lecture introduces the **Exponential Family** of probability distributions, establishing a unifying mathematical framework for many of the classical machine learning models covered in the course. The instructor, Chris, shows how common error distributions (such as Bernoulli for binary outcomes and Gaussian for continuous outcomes) are special cases of this family. Building on this foundation, he introduces **Generalized Linear Models (GLMs)**, which provide a highly structured, three-step recipe to construct prediction models for virtually any data type. The lecture wraps up by focusing on the most industrially important GLM classification algorithm: **Softmax Regression** (multi-class classification). Chris connects this to modern deep learning architectures, such as the attention heads of Transformer models (e.g., GPT-2), and introduces practical regularization techniques like **Label Smoothing**.

---

### 2. Key Concepts & Definitions
- **Exponential Family**: A class of probability distributions whose probability density function (or probability mass function) can be expressed in the canonical form:
  $$p(y; \eta) = b(y) \exp\left(\eta^T T(y) - a(\eta)\right)$$
- **Natural Parameter ($\eta$)**: The canonical parameter (or vector of parameters) that defines the distribution.
- **Sufficient Statistic ($T(y)$)**: A function of the data $y$ that captures all the information the data contains about the parameters. For most setups in this course, $T(y) = y$ (the identity function).
- **Base Measure ($b(y)$)**: A scaling factor/scalar function that depends only on the data $y$ and is independent of the parameters.
- **Log Partition Function ($a(\eta)$)**: The normalization constant that ensures the probability distribution integrates or sums to 1. It is defined as:
  $$a(\eta) = \log \int b(y) \exp\left(\eta^T T(y)\right) dy$$
- **Link Function**: A function $g^{-1}$ that maps the expected value of the target distribution to the natural parameter $\eta$.
- **Canonical Response Function**: A function $g$ that maps the natural parameter $\eta$ to the expected value of the target distribution, $\mathbb{E}[T(y)]$.
- **Softmax Function**: A generalization of the logistic/sigmoid function that maps a vector of arbitrary real-valued scores (logits) to a probability distribution over $k$ discrete classes.
- **Label Smoothing**: A regularization technique that replaces hard one-hot target vectors with a soft, smoothed target distribution to mitigate overfitting and improve generalization.

---

### 3. Mathematical Formulations & Derivations

#### A. The Unifying Properties of $a(\eta)$
The log partition function $a(\eta)$ acts as a cumulant generating function. This ensures that its derivatives automatically yield the statistical moments of the sufficient statistic $T(y)$:

1. **First Derivative (Expectation)**:
   $$\frac{\partial}{\partial \eta} a(\eta) = \mathbb{E}[T(y)]$$
2. **Second Derivative (Variance)**:
   $$\frac{\partial^2}{\partial \eta^2} a(\eta) = \text{Var}(T(y))$$

Because the second derivative represents variance (which is always non-negative), $a(\eta)$ is guaranteed to be a **convex function**, making gradient-based optimization highly stable and globally convergent for these distributions.

#### B. Proof: Bernoulli as an Exponential Family
Consider a coin flip modeled as a Bernoulli distribution with parameter $\phi \in (0, 1)$ representing the probability of heads ($y=1$):
$$p(y; \phi) = \phi^y (1-\phi)^{1-y}$$
To map this into the exponential family canonical form, we exponentiate the logarithm of the distribution:
$$p(y; \phi) = \exp\left(\log\left(\phi^y (1-\phi)^{1-y}\right)\right)$$
$$p(y; \phi) = \exp\left(y \log \phi + (1-y) \log(1-\phi)\right)$$
$$p(y; \phi) = \exp\left(y \log \phi - y\log(1-\phi) + \log(1-\phi)\right)$$
$$p(y; \phi) = \exp\left(y \log\left(\frac{\phi}{1-\phi}\right) + \log(1-\phi)\right)$$

By matching this terms to the canonical form $b(y) \exp\left(\eta^T T(y) - a(\eta)\right)$, we define:
- $b(y) = 1$
- $T(y) = y$
- $\eta = \log\left(\frac{\phi}{1-\phi}\right) \quad \text{(This is the log-odds or logit function)}$
- $a(\eta) = -\log(1-\phi)$

To write $a(\eta)$ purely as a function of $\eta$, we solve for $\phi$ in terms of $\eta$:
$$\mathrm{e}^\eta = \frac{\phi}{1-\phi} \implies \phi(1-\phi) \mathrm{e}^\eta = \phi \implies \phi = \frac{\mathrm{e}^\eta}{1 + \mathrm{e}^\eta} = \frac{1}{1 + \mathrm{e}^{-\eta}}$$
This recovers the familiar **sigmoid/logistic function**! Plugging $\phi$ back into $a(\eta)$ yields:
$$a(\eta) = -\log\left(1 - \frac{1}{1+\mathrm{e}^{-\eta}}\right) = -\log\left(\frac{\mathrm{e}^{-\eta}}{1+\mathrm{e}^{-\eta}}\right) = \log\left(1 + \mathrm{e}^\eta\right)$$

#### C. Proof: Gaussian as an Exponential Family
Consider a Gaussian distribution with mean $\mu$ and variance fixed at $\sigma^2 = 1$:
$$p(y; \mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(y-\mu)^2}{2}\right)$$
Expanding the quadratic term in the exponent:
$$p(y; \mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2} + \mu y - \frac{\mu^2}{2}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right) \exp\left(\mu y - \frac{\mu^2}{2}\right)$$

By matching terms to $b(y) \exp\left(\eta^T T(y) - a(\eta)\right)$:
- $b(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right)$
- $T(y) = y$
- $\eta = \mu$
- $a(\eta) = \frac{\mu^2}{2} = \frac{\eta^2}{2}$

---

### 4. Generalized Linear Models (GLMs)
To build a GLM to predict a target variable $y$ given features $x \in \mathbb{R}^{d+1}$ and learned parameters $\theta \in \mathbb{R}^{d+1}$, we make three design assumptions:

1. **Exponential Family Noise**: The conditional distribution of $y$ given $x$ belongs to the Exponential Family parameterized by $\eta$:
   $$y \mid x; \theta \sim \text{ExponentialFamily}(\eta)$$
2. **Predicting the Expectation**: Our goal is to predict the expected value of $T(y)$ given $x$. That is, our hypothesis function $h_\theta(x)$ is defined as:
   $$h_\theta(x) = \mathbb{E}[T(y) \mid x]$$
3. **Linearity**: The natural parameter $\eta$ and features $x$ are linearly related via our model weights $\theta$:
   $$\eta = \theta^T x$$

#### The Common Error-Correcting Gradient Update
Under maximum likelihood estimation (MLE), taking the log-likelihood of a GLM and taking the gradient with respect to $\theta$ yields an identical, elegant "error-correcting" update rule for Stochastic Gradient Descent (SGD) across *all* distributions in the family:
$$\theta_j := \theta_j - \alpha \left( h_{\theta^{(t)}}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
This is not an algebraic coincidence; the structure is a direct consequence of the exponential family's canonical properties.

<div id="plotly-glm-crank" class="plotly-chart" aria-label="Interactive Plotly diagram: the GLM parameter crank mapping features through weights, the natural parameter, and the canonical response function to the prediction"></div>

---

### 5. Multi-Class Classification: Softmax Regression
When predicting discrete outcomes among $k > 2$ classes (e.g., classifying a pixel image as a cat, dog, car, or bus), we use Softmax Regression.

#### One-Hot Vector Representation
Rather than labeling classes as scalars $\{1, 2, 3, 4\}$, we represent them as orthogonal one-hot support vectors $y \in \mathbb{R}^k$:
- $\text{Cat} = [1, 0, 0, 0]^T$
- $\text{Dog} = [0, 1, 0, 0]^T$
- $\text{Car} = [0, 0, 1, 0]^T$
- $\text{Bus} = [0, 0, 0, 1]^T$

Our parameters consist of $k$ distinct parameter vectors, one for each class: $\theta_1, \theta_2, \dots, \theta_k \in \mathbb{R}^{d+1}$. The score (or logit) for class $j$ given input $x$ is $\theta_j^T x$.

#### The Softmax Formulation
The conditional probability that input $x$ belongs to class $j$ is computed by exponentiating and normalizing the scores:
$$p(y = j \mid x; \theta) = \frac{\exp(\theta_j^T x)}{\sum_{l=1}^k \exp(\theta_l^T x)}$$

#### Cross-Entropy Loss
Under maximum likelihood estimation, maximizing the multinomial log-likelihood is equivalent to minimizing the cross-entropy loss over $n$ training examples:
$$\mathcal{L}(\theta) = -\sum_{i=1}^n \sum_{j=1}^k y_j^{(i)} \log p(y^{(i)} = j \mid x^{(i)}; \theta)$$

<div id="plotly-softmax-geometry" class="plotly-chart" aria-label="Interactive Plotly chart: softmax decision regions for four classes in a 2D feature plane with probability heatmap"></div>

### 6. Label Smoothing
A standard "hard" one-hot label $y^{(i)} = [1, 0, 0, 0]^T$ forces cross-entropy to drive the model's logits $\theta_1^T x \to \infty$ relative to all other classes to achieve a probability of exactly $1.0$. This leads to severe overfitting and overconfident predictions.

**Label Smoothing** modifies the target vector by distributing a small probability mass $\epsilon$ uniformly across all classes:
$$y_{\text{smooth}, j} = y_j (1 - \epsilon) + \frac{\epsilon}{k}$$
For example, in a 4-class classification problem with $\epsilon = 0.1$, the targets change from:
$$y = [1, 0, 0, 0]^T \implies y_{\text{smooth}} = [0.925, 0.025, 0.025, 0.025]^T$$

#### Benefits:
- **Overfitting Mitigation**: Prevents the weights $\theta$ from exploding to infinity.
- **Robustness**: Makes the model robust to mislabeled dataset errors (label noise).
- **Gradient Stability**: Ensures continuous, healthy gradient flow during backpropagation.

---

### 7. Applications
- **Next-Token Prediction in LLMs**: Modern generative foundation models (like ChatGPT and Claude) utilize a massive softmax layer as their final prediction head to select the most probable next word/token from a large vocabulary (typically $50k$ to $250k$ discrete tokens).
- **The Self-Attention Mechanism**: Softmax normalization is the mathematical cornerstone of the self-attention formula in the Transformer architecture:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  where softmax scales and normalizes token relevance dynamically.

---

### 8. Reflection Questions
1. **Mathematical Convexity**: Why is the log partition function $a(\eta)$ guaranteed to be convex? What implications does this convexity have for fitting GLMs via Stochastic Gradient Descent?
2. **Softmax Redundancy**: Show that the softmax formulation is overparameterized by proving that subtracting a constant vector $\psi$ from all class weights $\theta_j$ does not change the resulting conditional probabilities $p(y = j \mid x; \theta)$. How is this redundancy typically resolved in statistics versus modern deep learning practice?
3. **The Limits of Linear Separability**: In low-dimensional spaces, linear GLMs like logistic or softmax regression can only separate data that is cleanly linearly separable. Why does high-dimensional feature space (such as the $768$-dimensional space of GPT-2) naturally alleviate this limitation and make linear classifiers highly effective?

---

### 9. Further Reading & Resources
- **Convex Optimization (Boyd & Vandenberghe)**: Focus on Chapter 3's treatment of log-convex functions and partition function geometry.
- **The Exponential Family in Statistics (Brown, 1986)**: The canonical textbook reference for deeper measure-theoretic properties of exponential families.
- **"Large Language Monkeys" (CS229 Lecture Lore)**: A reference to the instructor's research exploring the statistical power of stochastic token sampling and how temperature scaling influences generation paths.
