# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 2: Supervised Learning Setup

### 1. Summary
This lecture formally sets up the supervised learning paradigm, utilizing linear regression with least squares as the foundational learning algorithm [65, 72]. The instructor, Chris, breaks down the core notations, defining the hypothesis function, the training set, and the features [77, 80, 107]. He introduces the concept of Empirical Risk Minimization (ERM) through the Least Squares cost function [97, 115]. The lecture then investigates two primary mathematical pathways for optimization: iterative methods via Batch, Stochastic (SGD), and Minibatch Gradient Descent, and an analytical closed-form solution via the Normal Equations [74, 75, 157]. Chris also highlights critical engineering and hardware realities of modern machine learning—such as GPU memory constraints, step-size (learning rate) scheduling, and the distinct focus of machine learning on generalization over the parameter recovery focus of classical statistics [150, 172, 174, 201].

---

### 2. Key Concepts & Definitions
- **Supervised Machine Learning**: A paradigm where the learning algorithm is trained on explicit pairs of inputs and target labels [65, 80].
- **Hypothesis ($h$)**: A function $h: \mathcal{X} \to \mathcal{Y}$ mapping the input space $\mathcal{X}$ to the output space $\mathcal{Y}$ [77].
- **Training Set**: A collection of $n$ training examples denoted as $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$ [80, 110].
- **Notation Conventions**:
  - $x^{(i)}$: The input vector (features) for the $i$-th training example [81, 107].
  - $y^{(i)}$: The target output/label for the $i$-th training example [80, 107].
  - $n$: The total number of training examples [110].
  - $d$: The number of input features or dimensions [110].
  - $x_0 = 1$: The intercept/bias term convention, meaning $x \in \mathbb{R}^{d+1}$ and the parameters $\theta \in \mathbb{R}^{d+1}$ [105, 110].
- **Regression**: A supervised learning task where the output space $\mathcal{Y}$ is continuous (e.g., predicting housing prices) [66, 79, 86].
- **Classification**: A supervised learning task where the output space $\mathcal{Y}$ is discrete (e.g., predicting cat vs. dog or predicting the next token in a sequence) [65, 78, 87].
- **Empirical Risk Minimization (ERM)**: The process of choosing parameters $\theta$ that minimize the average loss (risk) over the training set [97, 100].
- **Positive Semidefinite (PSD)**: A symmetric matrix $M$ is PSD if $z^T M z \ge 0$ for all vectors $z$ [227]. The matrix $X^T X$ is positive semidefinite, which ensures the cost function is convex (bowl-shaped) and gradient descent will converge to a global minimum [208, 214, 227].

---

### 3. Mathematical Formulations & Derivations
#### The Linear Hypothesis
The linear hypothesis is formulated as:
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_d x_d = \sum_{j=0}^d \theta_j x_j = \theta^T x$$
where $\theta$ represents the parameter vector (weights) and $x_0 = 1$ is the intercept term [91, 105, 106].

#### Least Squares Cost Function
To measure how well the hypothesis fits the training data, we define the Least Squares cost function $J(\theta)$ [115, 116]:
$$J(\theta) = \frac{1}{2} \sum_{i=1}^n \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$
- **Why Square?** Squaring ensures positive error penalties, places a heavier penalty on larger outliers, and results in a smooth, differentiable quadratic "bowl" shape [116, 118, 136].
- **Why $\frac{1}{2}$?** The constant is added by convention so that the factor of 2 from the derivative of the squared term cancels out, simplifying the gradient update rule [123, 143].

#### Analytical Derivation of the Gradient
For a single training example $(x, y)$, the partial derivative of $J(\theta)$ with respect to a single parameter $\theta_j$ is derived using the chain rule [143]:
$$\frac{\partial}{\partial \theta_j} \left( \frac{1}{2} (h_\theta(x) - y)^2 \right) = (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_\theta(x) - y)$$
Since $h_\theta(x) = \sum_{k=0}^d \theta_k x_k$, we have $\frac{\partial}{\partial \theta_j} h_\theta(x) = x_j$. Therefore:
$$\frac{\partial}{\partial \theta_j} J(\theta) = (h_\theta(x) - y)x_j$$

---

### 4. Step-by-Step Optimization Algorithms
#### A. Batch Gradient Descent (BGD)
Batch gradient descent computes the gradient over the entire training set before performing a single parameter update [145].
1. **Initialize** parameters $\theta$ randomly or to 0 [140].
2. **Loop until convergence**:
   For every parameter $j \in \{0, \dots, d\}$, update:
   $$\theta_j := \theta_j - \alpha \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
   where $\alpha$ is the learning rate (step size) [140, 141].
3. **Properties**: Mathematically guaranteed to converge to the global minimum for convex bowl-shaped functions like Least Squares [208, 214]. However, it is computationally prohibitive for large datasets (e.g., scanning the entire internet) because each step requires summing over all $n$ examples [145].

#### B. Stochastic Gradient Descent (SGD)
To address the computational bottleneck of batch updates, Stochastic Gradient Descent updates parameters incrementally after examining a single training example [155, 157].
1. **Shuffle** the training set randomly to ensure independent, representative samples [160, 163, 194].
2. **Loop through the training set**:
   For each training example $i \in \{1, \dots, n\}$:
   For every parameter $j \in \{0, \dots, d\}$, update:
   $$\theta_j := \theta_j - \alpha \left( h_{\theta^{(t)}}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
   where $\theta^{(t)}$ is the parameter vector from the previous step [162, 163].
3. **Properties**: Much faster to start making progress [157]. Instead of taking a direct path to the minimum, it "drunkenly stumbles" or wiggles [207, 209]. As it nears the optimum, it bounces around in a high-dimensional ball proportional to the step size $\alpha$ [150, 207].

#### C. Minibatch Gradient Descent
Minibatch gradient descent is the modern workhorse of deep learning (e.g., PyTorch defaults) [158]. It strikes a balance between BGD and SGD by computing the gradient over a small subset of size $B$ (batch size) [158, 185].
1. **Form a random batch** $\mathcal{B}$ of size $B$ from the training set [159, 185].
2. **Update rule**:
   $$\theta := \theta - \alpha_{\mathcal{B}} \frac{1}{B} \sum_{i \in \mathcal{B}} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}$$
   where $\alpha_{\mathcal{B}}$ is the normalized learning rate adjusted for the batch size [185, 186, 204].

```
[Insert diagram: Comparison of Optimization Trajectories: Batch Gradient Descent (direct path to center of elliptical contours) vs. Stochastic Gradient Descent (noisy, jagged "drunken walk" path bouncing around the optimum)]
```

---

### 5. Analytical Solution: The Normal Equations
For linear regression with least squares, we can bypass iterative optimization and solve for the optimal parameters $\theta$ analytically in closed form [75, 217].

#### Matrix Setup
Let the **Design Matrix** $X \in \mathbb{R}^{n \times (d+1)}$ contain all training inputs stacked by rows, and $y \in \mathbb{R}^n$ be the target output vector [218]:
$$X = \begin{bmatrix} (x^{(1)})^T \\ (x^{(2)})^T \\ \vdots \\ (x^{(n)})^T \end{bmatrix} = \begin{bmatrix} 1 & x_1^{(1)} & \dots & x_d^{(1)} \\ 1 & x_1^{(2)} & \dots & x_d^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(n)} & \dots & x_d^{(n)} \end{bmatrix}, \quad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)} \end{bmatrix}$$

#### Matrix Derivation
We can rewrite the Least Squares cost function in vector-matrix form using the $L_2$ norm inner product [219, 220]:
$$J(\theta) = \frac{1}{2} (X\theta - y)^T (X\theta - y)$$
Expanding the terms:
$$J(\theta) = \frac{1}{2} \left( (X\theta)^T X\theta - (X\theta)^T y - y^T X\theta + y^T y \right)$$
Since $(X\theta)^T y$ is a scalar, it is equal to its transpose $y^T X\theta$:
$$J(\theta) = \frac{1}{2} \left( \theta^T X^T X \theta - 2\theta^T X^T y + y^T y \right)$$

Taking the gradient with respect to the vector $\theta$ and setting it to 0 to find the minimum [221, 222, 223]:
$$\nabla_\theta J(\theta) = \frac{1}{2} \left( 2 X^T X \theta - 2 X^T y \right) = 0$$
$$X^T X \theta = X^T y$$
This system is known as the **Normal Equations** [75, 217].

Solving for $\theta$ yields the closed-form least-squares estimator [224]:
$$\theta = (X^T X)^{-1} X^T y$$

#### Invertibility & Redundancy
This derivation assumes that $X^T X$ is invertible [225].
- If $X^T X$ is **singular (non-invertible)**, it occurs when there are redundant, linearly dependent features or when there are fewer training examples than features ($n < d$) [225].
- In this case, there exists a non-trivial **null space** [226]. Any vector added from this null space results in an equally optimal parameter set, meaning there is an entire subspace of valid solutions [226].

---

### 6. Engineering & Hardware Lore in Modern ML
The lecture features a rich discussion on how optimization choices in practice deviate from pure statistical theory:
- **CapEx and Model Flops Utilization (MFU)**: Modern deep learning is heavily gated by massive capital expenditures (CapEx) for hardware infrastructure [200]. Selecting batch sizes and step-size schedules (e.g., cosine decay schedules) is often dictated by hardware constraints to keep GPU memory (High Bandwidth Memory, or HBM) fully utilized and maximize computation speed [172, 182, 191, 201].
- **The Redundancy Trade-off**: Smaller batches have higher variance, but they excel in highly redundant datasets [176]. If a dataset has 90 copies of the same image, processing them in a large batch is wasteful; a smaller batch allows the model to update 90 times, converging faster [177, 178].
- **Why Machine Learning is Not Just Statistics**: Traditional statistics focus on parameter recovery guarantees (finding the true $\theta^*$) and rigorous convergence tests [150, 151, 174]. Modern ML focuses on generalization error on unseen test sets [102, 131]. We often train models until we run out of compute credits, stop there, and find that they still generalize beautifully [152, 174, 215, 216].

---

### 7. Applications
- **Real Estate Valuation**: Using features like square footage, lot area, and number of bedrooms to predict continuous home sale prices (e.g., using real-world data like the Ames Housing Dataset) [88, 89, 102].
- **Sequence Modeling (Classifier Heads)**: Although linear models are simple, they serve as the crucial final output step (the "classification head") of modern deep learning architectures, such as predicting the next token in Large Language Models using softmax activations [87, 90, 127].

---

### 8. Reflection Questions
1. **Step-Size Divergence**: What mathematically occurs to the parameter updates in gradient descent if the learning rate $\alpha$ is chosen to be excessively large, and how does this manifest visually on the cost curve $J(\theta)$? [149, 207, 208]
2. **Singular Design Matrices**: If a dataset contains fewer training examples than dimensions ($n < d$), why does $X^T X$ become non-invertible? What does this imply about the uniqueness of the parameter vector $\theta$ that minimizes the least squares cost? [225, 226]
3. **Statistics vs. Machine Learning Goals**: Explain the conceptual conflict between minimizing the training loss $J(\theta)$ to the highest degree of numerical precision and maximizing the model's generalization capabilities on a test set, particularly in the context of batch size selection. [150, 151, 171, 174]

---

### 9. Further Reading & Resources
- **Steven Boyd's Convex Optimization**: Highly recommended by the instructors for mastering vector/matrix calculus and the properties of semidefinite matrices [214].
- **Friday TA Sections**: The essential venue for mastering matrix gradients, rank, null spaces, and vector derivatives [75, 76, 227].
- **Data Mixing & Cosine Scheduling Literature**: Academic papers on how foundation models schedule learning rates (e.g., Leslie's cosine scaling) and mix training corpora [191, 195, 196].
