# Stanford CS229: Machine Learning (Spring 2026)

## Lecture 2: Supervised Learning Setup

**Instructors**: Chris Ré (Professor of Computer Science) and Tengyu Ma (Assistant Professor of Computer Science)

### 1. Summary

This lecture formally sets up the supervised learning paradigm, utilizing linear regression with least squares as the foundational learning algorithm. The instructor, Chris Ré, breaks down the core notations, defining the hypothesis function, the training set, and the features. He introduces the concept of Empirical Risk Minimization (ERM) through the Least Squares cost function. The lecture then investigates two primary mathematical pathways for optimization: iterative methods via Batch, Stochastic (SGD), and Minibatch Gradient Descent, and an analytical closed-form solution via the Normal Equations. Chris also highlights critical engineering and hardware realities of modern machine learning—such as GPU memory constraints, step-size (learning rate) scheduling, and the distinct focus of machine learning on generalization over the parameter recovery focus of classical statistics.

---

### 2. Key Concepts & Definitions

- **Supervised Machine Learning**: A paradigm where the learning algorithm is trained on explicit pairs of inputs and target labels.
- **Hypothesis ($h$)**: A function $h: \mathcal{X} \to \mathcal{Y}$ mapping the input space $\mathcal{X}$ to the output space $\mathcal{Y}$.
- **Training Set**: A collection of $n$ training examples denoted as $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$.
- **Notation Conventions**:
  - $x^{(i)}$: The input vector (features) for the $i$-th training example.
  - $y^{(i)}$: The target output/label for the $i$-th training example.
  - $n$: The total number of training examples.
  - $d$: The number of input features or dimensions.
  - $x_0 = 1$: The intercept/bias term convention, meaning $x \in \mathbb{R}^{d+1}$ and the parameters $\theta \in \mathbb{R}^{d+1}$.
- **Regression**: A supervised learning task where the output space $\mathcal{Y}$ is continuous (e.g., predicting housing prices).
- **Classification**: A supervised learning task where the output space $\mathcal{Y}$ is discrete (e.g., predicting cat vs. dog or predicting the next token in a sequence).
- **Empirical Risk Minimization (ERM)**: The process of choosing parameters $\theta$ that minimize the average loss (risk) over the training set.
- **Positive Semidefinite (PSD)**: A symmetric matrix $M$ is PSD if $z^T M z \ge 0$ for all vectors $z$. The matrix $X^T X$ is positive semidefinite, which ensures the cost function is convex (bowl-shaped) and gradient descent will converge to a global minimum.

---

### 3. Mathematical Formulations & Derivations

#### The Linear Hypothesis

The linear hypothesis is formulated as:
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_d x_d = \sum_{j=0}^d \theta_j x_j = \theta^T x$$
where $\theta$ represents the parameter vector (weights) and $x_0 = 1$ is the intercept term.

#### Least Squares Cost Function

To measure how well the hypothesis fits the training data, we define the Least Squares cost function $J(\theta)$:
$$J(\theta) = \frac{1}{2} \sum_{i=1}^n \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$

- **Why Square?** Squaring ensures positive error penalties, places a heavier penalty on larger outliers, and results in a smooth, differentiable quadratic "bowl" shape.
- **Why $\frac{1}{2}$?** The constant is added by convention so that the factor of 2 from the derivative of the squared term cancels out, simplifying the gradient update rule.

#### Analytical Derivation of the Gradient

For a single training example $(x, y)$, the partial derivative of $J(\theta)$ with respect to a single parameter $\theta_j$ is derived using the chain rule:
$$\frac{\partial}{\partial \theta_j} \left( \frac{1}{2} (h_\theta(x) - y)^2 \right) = (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_\theta(x) - y)$$
Since $h_\theta(x) = \sum_{k=0}^d \theta_k x_k$, we have $\frac{\partial}{\partial \theta_j} h_\theta(x) = x_j$. Therefore:
$$\frac{\partial}{\partial \theta_j} J(\theta) = (h_\theta(x) - y)x_j$$

---

### 4. Step-by-Step Optimization Algorithms

#### A. Batch Gradient Descent (BGD)

Batch gradient descent computes the gradient over the entire training set before performing a single parameter update.

1. **Initialize** parameters $\theta$ randomly or to 0.
2. **Loop until convergence**:
   For every parameter $j \in \{0, \dots, d\}$, update:
   $$\theta_j := \theta_j - \alpha \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
   where $\alpha$ is the learning rate (step size).
3. **Properties**: Mathematically guaranteed to converge to the global minimum for convex bowl-shaped functions like Least Squares. However, it is computationally prohibitive for large datasets (e.g., scanning the entire internet) because each step requires summing over all $n$ examples.

#### Step-Size (Learning Rate) Convergence Condition

The update $\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$ only converges when the step size is small enough relative to the curvature of $J$. For least squares, the curvature is governed by the eigenvalues of $X^T X$:

$$0 < \alpha < \frac{2}{\lambda_{\max}(X^T X)}$$

where $\lambda_{\max}$ is the largest eigenvalue of $X^T X$. If $\alpha$ exceeds this threshold, each update overshoots the minimum and the iterates diverge — on the cost curve, $J(\theta)$ oscillates with growing amplitude instead of descending into the bowl. This is why practical training uses step-size schedules (e.g., decay or cosine schedules) rather than a fixed $\alpha$.

#### B. Stochastic Gradient Descent (SGD)

To address the computational bottleneck of batch updates, Stochastic Gradient Descent updates parameters incrementally after examining a single training example.

1. **Shuffle** the training set randomly to ensure independent, representative samples.
2. **Loop through the training set**:
   For each training example $i \in \{1, \dots, n\}$:
   For every parameter $j \in \{0, \dots, d\}$, update:
   $$\theta_j := \theta_j - \alpha \left( h_{\theta^{(t)}}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
   where $\theta^{(t)}$ is the parameter vector from the previous step.
3. **Properties**: Much faster to start making progress. Instead of taking a direct path to the minimum, it "drunkenly stumbles" or wiggles. As it nears the optimum, it bounces around in a high-dimensional ball proportional to the step size $\alpha$.

#### C. Minibatch Gradient Descent

Minibatch gradient descent is the modern workhorse of deep learning (e.g., PyTorch defaults). It strikes a balance between BGD and SGD by computing the gradient over a small subset of size $B$ (batch size).

1. **Form a random batch** $\mathcal{B}$ of size $B$ from the training set.
2. **Update rule**:
   $$\theta := \theta - \alpha_{\mathcal{B}} \frac{1}{B} \sum_{i \in \mathcal{B}} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}$$
   where $\alpha_{\mathcal{B}}$ is the normalized learning rate adjusted for the batch size.

```
[Insert diagram: Comparison of Optimization Trajectories: Batch Gradient Descent (direct path to center of elliptical contours) vs. Stochastic Gradient Descent (noisy, jagged "drunken walk" path bouncing around the optimum)]
```

---

### 5. Analytical Solution: The Normal Equations

For linear regression with least squares, we can bypass iterative optimization and solve for the optimal parameters $\theta$ analytically in closed form.

#### Matrix Setup

Let the **Design Matrix** $X \in \mathbb{R}^{n \times (d+1)}$ contain all training inputs stacked by rows, and $y \in \mathbb{R}^n$ be the target output vector:
$$X = \begin{bmatrix} (x^{(1)})^T \\ (x^{(2)})^T \\ \vdots \\ (x^{(n)})^T \end{bmatrix} = \begin{bmatrix} 1 & x_1^{(1)} & \dots & x_d^{(1)} \\ 1 & x_1^{(2)} & \dots & x_d^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(n)} & \dots & x_d^{(n)} \end{bmatrix}, \quad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)} \end{bmatrix}$$

#### Matrix Derivation

We can rewrite the Least Squares cost function in vector-matrix form using the $L_2$ norm inner product:
$$J(\theta) = \frac{1}{2} (X\theta - y)^T (X\theta - y)$$
Expanding the terms:
$$J(\theta) = \frac{1}{2} \left( (X\theta)^T X\theta - (X\theta)^T y - y^T X\theta + y^T y \right)$$
Since $(X\theta)^T y$ is a scalar, it is equal to its transpose $y^T X\theta$:
$$J(\theta) = \frac{1}{2} \left( \theta^T X^T X \theta - 2\theta^T X^T y + y^T y \right)$$

Taking the gradient with respect to the vector $\theta$ and setting it to 0 to find the minimum:
$$\nabla_\theta J(\theta) = \frac{1}{2} \left( 2 X^T X \theta - 2 X^T y \right) = 0$$
$$X^T X \theta = X^T y$$
This system is known as the **Normal Equations**.

Solving for $\theta$ yields the closed-form least-squares estimator:
$$\theta = (X^T X)^{-1} X^T y$$

#### Worked Micro-Example

Take two training points, $(x^{(1)}, y^{(1)}) = (0, 1)$ and $(x^{(2)}, y^{(2)}) = (1, 3)$, with the intercept convention $x_0 = 1$:

$$X = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad y = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$$

**Normal Equations path:**
$$X^T X = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}, \quad X^T y = \begin{bmatrix} 4 \\ 3 \end{bmatrix} \implies \theta = (X^T X)^{-1} X^T y = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

So $h_\theta(x) = 1 + 2x$, which passes exactly through both points (zero training error).

**Gradient descent path:** starting at $\theta = (0, 0)$ with $\alpha = 0.1$, the first update gives residuals $(1, 3)$, so:
$$\theta_1 = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - 0.1 \begin{bmatrix} 4 \\ 3 \end{bmatrix} = \begin{bmatrix} -0.4 \\ -0.3 \end{bmatrix}$$
Iterating converges toward $(1, 2)$ — the same solution the Normal Equations give in one step. This illustrates the trade-off: iterative methods scale to huge $n$ and $d$, while the closed form is exact but requires an $O(d^3)$ matrix inversion.

#### Probabilistic Interpretation

Least squares is not an arbitrary choice: it is the **maximum likelihood estimator (MLE)** under the assumption that targets are linear plus Gaussian noise:
$$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}, \quad \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$
Maximizing the Gaussian log-likelihood $\sum_i \log p(y^{(i)} \mid x^{(i)}; \theta)$ reduces exactly to minimizing $J(\theta)$ — the $\frac{1}{2}$ factor and the squared error both fall out of the Gaussian exponent. This connects least squares to the broader ERM framework and foreshadows other losses (e.g., logistic loss for classification) derived from other noise/likelihood assumptions.

#### Invertibility & Redundancy

This derivation assumes that $X^T X$ is invertible.

- If $X^T X$ is **singular (non-invertible)**, it occurs when there are redundant, linearly dependent features or when there are fewer training examples than features ($n < d$).
- In this case, there exists a non-trivial **null space**. Any vector added from this null space results in an equally optimal parameter set, meaning there is an entire subspace of valid solutions.

---

### 6. Engineering & Hardware Lore in Modern ML

The lecture features a rich discussion on how optimization choices in practice deviate from pure statistical theory:

- **CapEx and Model Flops Utilization (MFU)**: Modern deep learning is heavily gated by massive capital expenditures (CapEx) for hardware infrastructure. Selecting batch sizes and step-size schedules (e.g., cosine decay schedules) is often dictated by hardware constraints to keep GPU memory (High Bandwidth Memory, or HBM) fully utilized and maximize computation speed.
- **The Redundancy Trade-off**: Smaller batches have higher variance, but they excel in highly redundant datasets. If a dataset has 90 copies of the same image, processing them in a large batch is wasteful; a smaller batch allows the model to update 90 times, converging faster.
- **Why Machine Learning is Not Just Statistics**: Traditional statistics focus on parameter recovery guarantees (finding the true $\theta^*$) and rigorous convergence tests. Modern ML focuses on generalization error on unseen test sets. We often train models until we run out of compute credits, stop there, and find that they still generalize beautifully.

---

### 7. Applications

- **Real Estate Valuation**: Using features like square footage, lot area, and number of bedrooms to predict continuous home sale prices (e.g., using real-world data like the Ames Housing Dataset).
- **Sequence Modeling (Classifier Heads)**: Although linear models are simple, they serve as the crucial final output step (the "classification head") of modern deep learning architectures, such as predicting the next token in Large Language Models using softmax activations.

---

### 8. Reflection Questions

1. **Step-Size Divergence**: What mathematically occurs to the parameter updates in gradient descent if the learning rate $\alpha$ is chosen to be excessively large, and how does this manifest visually on the cost curve $J(\theta)$?
2. **Singular Design Matrices**: If a dataset contains fewer training examples than dimensions ($n < d$), why does $X^T X$ become non-invertible? What does this imply about the uniqueness of the parameter vector $\theta$ that minimizes the least squares cost?
3. **Statistics vs. Machine Learning Goals**: Explain the conceptual conflict between minimizing the training loss $J(\theta)$ to the highest degree of numerical precision and maximizing the model's generalization capabilities on a test set, particularly in the context of batch size selection.

---

### 9. Further Reading & Resources

- **Steven Boyd's Convex Optimization**: Highly recommended by the instructors for mastering vector/matrix calculus and the properties of semidefinite matrices.
- **Friday TA Sections**: The essential venue for mastering matrix gradients, rank, null spaces, and vector derivatives.
- **Data Mixing & Cosine Scheduling Literature**: Academic papers on how foundation models schedule learning rates (e.g., Leslie's cosine scaling) and mix training corpora.
