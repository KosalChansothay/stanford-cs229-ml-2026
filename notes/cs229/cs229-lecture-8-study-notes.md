# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 8: Neural Networks 2 (Backpropagation)

### 1. Summary
This lecture details **backpropagation** (also referred to as **reverse-mode automatic differentiation** or **auto-differentiation**), the core algorithm used to compute the gradients of a loss function with respect to neural network parameters. Establishing a general framework through the lens of **differentiable circuits** (computational graphs), the lecture presents a profound complexity theorem: if a real-valued scalar function $f$ can be computed by a circuit of size $N$ (the forward pass), its gradient $\nabla f$ can be computed in identical asymptotic time complexity $O(N)$ by a circuit of size $O(N)$ (the backward pass). The instructor, Tengyu, provides a step-by-step mathematical treatment of vector and matrix derivatives using the **multivariate chain rule**. This includes rigorous analytical derivations for the backward pass of a standard matrix multiplication layer ($u = Wz + b$)—unveiling its connection to biological **Hebbian learning**—and an element-wise activation layer ($u = \sigma(z)$). Finally, the lecture explores the mathematical elegance of **Hessian-vector products**, the Markovian nature of backpropagation, and the profound system and memory engineering realities involved in caching and freeing activations on modern GPU architectures.

---

### 2. Key Concepts & Definitions
- **Differentiable Circuit**: A computational graph representing a function as a directed acyclic graph (DAG) composed of a sequence of basic arithmetic operations (addition, subtraction, division) and elementary differentiable functions (such as exponentials, sines, cosines, and sigmoids).
- **Backpropagation / Auto-Differentiation**: A computationally optimal algorithm that applies the chain rule recursively from the output layer back to the input layer, allowing the automatic and exact computation of gradients for complex composed functions.
- **Forward Pass**: The execution phase where input data is propagated forward through the differentiable circuit layer-by-layer to compute intermediate activations and evaluate the final scalar loss function.
- **Backward Pass**: The execution phase where derivative signals are propagated in reverse through the circuit to compute the gradient of the loss function with respect to all parameters.
- **Jacobian Matrix**: A matrix $J \in \mathbb{R}^{n \times m}$ containing all first-order partial derivatives of an $n$-dimensional vector-valued function $g(z)$ with respect to its $m$-dimensional input vector $z$.
- **Hadamard Product ($\odot$)**: An entrywise (element-wise) multiplication of two vectors or matrices of identical dimensions, denoted mathematically as $[A \odot B]\_{ij} = A\_{ij} B\_{ij}$.
- **Hessian-Vector Product**: The product of the $N \times N$ second-order partial derivative matrix (Hessian) and an arbitrary vector $v \in \mathbb{R}^N$. While calculating the full Hessian is computationally prohibitive, the Hessian-vector product can be evaluated efficiently in $O(N)$ time.
- **Hebbian Learning Rule**: A biological learning principle where synaptic connection weights are updated based on the product of pre-synaptic activation and post-synaptic error. In backpropagation, the gradient with respect to a weight parameter simplifies to this exact product.

---

### 3. Mathematical Formulations & Derivations

#### A. The Differentiable Circuit Theorem
Let $\mathcal{C}$ be a differentiable circuit of size $N$ that computes a real-valued scalar function $f: \mathbb{R}^l \to \mathbb{R}$. The size $N$ represents the total count of elementary operations and activations in the computational graph. 

Assuming each basic operation takes $O(1)$ time to evaluate, the function value $f(x)$ is computed in time $O(N)$.
**Theorem**: The gradient vector $\nabla_x f(x) \in \mathbb{R}^l$ can be computed in time $O(N)$ by a circuit of size $O(N)$.
- *Implication in Deep Learning*: The loss function represents the scalar output $f$. The input variables represent the network parameters $\theta$. Since the number of operations required to evaluate a neural network is proportional to its parameter count, both the loss computation (forward pass) and the gradient computation (backward pass) execute in $O(|\theta|)$ time.

#### B. The Vector-Valued Chain Rule
Consider an intermediate step in a computational graph where a vector $z \in \mathbb{R}^m$ is mapped to a vector $u \in \mathbb{R}^n$ via a differentiable function $g$, and $u$ is subsequently mapped to a scalar loss $J \in \mathbb{R}$ via a function $f$:
$$u = g(z) \in \mathbb{R}^n, \quad J = f(u) \in \mathbb{R}$$

Suppose we are given the gradient of the loss with respect to the output variable $u$, denoted as $\frac{\partial J}{\partial u} \in \mathbb{R}^n$. We wish to compute the gradient with respect to the input variable $z$, denoted as $\frac{\partial J}{\partial z} \in \mathbb{R}^m$.
Using the multivariate chain rule, the partial derivative with respect to a single coordinate $z_i$ is derived by summing over all intermediate pathways $u_j$:
$$\frac{\partial J}{\partial z_i} = \sum\_{j=1}^n \frac{\partial J}{\partial u_j} \frac{\partial u_j}{\partial z_i} = \sum\_{j=1}^n \frac{\partial J}{\partial u_j} \frac{\partial g_j(z)}{\partial z_i}$$

We can stack these partial derivatives into a vector. This formulation is equivalent to multiplying the transpose of the Jacobian matrix by the downstream gradient vector:
$$\frac{\partial J}{\partial z} = \left(\frac{\partial g}{\partial z}\right)^T \frac{\partial J}{\partial u}$$
where the Jacobian $\frac{\partial g}{\partial z} \in \mathbb{R}^{n \times m}$ is defined as $\left[\frac{\partial g}{\partial z}\right]\_{ji} = \frac{\partial u_j}{\partial z_i}$.

#### C. Backward Pass of a Matrix Multiplication Layer
Let the forward pass of a fully-connected layer be defined as:
$$u = W z + b$$
where $z \in \mathbb{R}^m$ is the input activation vector, $W \in \mathbb{R}^{n \times m}$ is the weight matrix, $b \in \mathbb{R}^n$ is the bias vector, and $u \in \mathbb{R}^n$ is the output logit vector.

##### 1. Gradient with respect to input $z$:
The $j$-th coordinate of the output vector is given by:
$$u_j = b_j + \sum\_{k=1}^m W\_{jk} z_k$$
Differentiating $u_j$ with respect to an input coordinate $z_i$:
$$\frac{\partial u_j}{\partial z_i} = \frac{\partial}{\partial z_i} \left( b_j + \sum\_{k=1}^m W\_{jk} z_k \right) = W\_{ji}$$
This implies that the Jacobian matrix is exactly the weight matrix: $\frac{\partial u}{\partial z} = W$.
Applying the vector chain rule:
$$\frac{\partial J}{\partial z} = W^T \frac{\partial J}{\partial u}$$

##### 2. Gradient with respect to weight parameters $W$:
We wish to compute the partial derivative of the scalar loss $J$ with respect to an individual weight parameter $W\_{ij}$. Applying the chain rule:
$$\frac{\partial J}{\partial W\_{ij}} = \sum\_{k=1}^n \frac{\partial J}{\partial u_k} \frac{\partial u_k}{\partial W\_{ij}}$$
Expanding the term $u_k$:
$$\frac{\partial u_k}{\partial W\_{ij}} = \frac{\partial}{\partial W\_{ij}} \left( b_k + \sum\_{s=1}^m W\_{ks} z_s \right)$$
This derivative is non-zero only when the row index matches ($k=i$) and the column index matches ($s=j$). Therefore:
$$\frac{\partial u_k}{\partial W\_{ij}} = \begin{cases} z_j & \text{if } k=i \\ 0 & \text{if } k \neq i \end{cases}$$
Substituting this back into the chain rule summation collapses the sum to a single term:
$$\frac{\partial J}{\partial W\_{ij}} = \frac{\partial J}{\partial u_i} z_j$$
*Biological Synaptic Connection*: This is identical to the Hebbian learning rule. The synaptic weight update $\frac{\partial J}{\partial W\_{ij}}$ is the product of the post-synaptic activation error $\frac{\partial J}{\partial u_i}$ and the pre-synaptic activation value $z_j$.

We can write this elegantly in vector notation as an outer product:
$$\frac{\partial J}{\partial W} = \left(\frac{\partial J}{\partial u}\right) z^T$$
For a single training example, the weight gradient matrix $\frac{\partial J}{\partial W} \in \mathbb{R}^{n \times m}$ is a **rank-1 matrix**.

##### 3. Gradient with respect to bias parameters $b$:
Since $u_j = b_j + \sum\_{k=1}^m W\_{jk} z_k$, we have $\frac{\partial u_j}{\partial b_i} = 1$ if $j=i$, and $0$ otherwise.
Applying the chain rule:
$$\frac{\partial J}{\partial b} = \frac{\partial J}{\partial u}$$

#### D. Backward Pass of an Element-wise Activation Layer
Let the forward pass of an activation layer be defined as:
$$u = \sigma(z)$$
where $\sigma$ is an activation function applied entrywise to $z \in \mathbb{R}^n$.

Since the $i$-th output $u_i = \sigma(z_i)$ depends strictly on the $i$-th input coordinate $z_i$, the off-diagonal entries of the Jacobian matrix are zero:
$$\frac{\partial u_j}{\partial z_i} = \begin{cases} \sigma'(z_i) & \text{if } j=i \\ 0 & \text{if } j \neq i \end{cases}$$
The Jacobian matrix $\frac{\partial u}{\partial z} \in \mathbb{R}^{n \times n}$ is a diagonal matrix:
$$\frac{\partial u}{\partial z} = \operatorname{diag}\left(\sigma'(z_1), \\sigma'(z_2), \dots, \sigma'(z_n)\right)$$
Applying the vector chain rule:
$$\frac{\partial J}{\partial z} = \operatorname{diag}\left(\sigma'(z)\right) \frac{\partial J}{\partial u} = \sigma'(z) \odot \frac{\partial J}{\partial u}$$
where $\odot$ represents the element-wise Hadamard product, avoiding the need to instantiate a sparse $n \times n$ matrix in memory.

---

### 4. Step-by-Step Optimization Algorithms & Workflows

The complete vectorized backpropagation workflow for a multi-layer feedforward neural network proceeds as follows:

#### I. The Forward Pass
For layers $l = 1, \dots, L$:
1. Compute the pre-activation linear combinations:
   $$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
2. Apply the entrywise activation function:
   $$a^{(l)} = \sigma\left(z^{(l)}\right)$$
   where $a^{(0)} = x$ is the input vector.
3. Evaluate the final scalar loss $J = \mathcal{L}(a^{(L)}, y)$.

#### II. The Backward Pass
We initialize the backward pass at the output layer $l = L$ by computing the gradient of the loss with respect to the output activations:
$$\delta^{(L)} = \frac{\partial J}{\partial a^{(L)}} = \nabla\_{a^{(L)}} \mathcal{L}(a^{(L)}, y)$$

For layers $l = L, L-1, \dots, 1$:
1. Backpropagate the gradient through the non-linear activation function at layer $l$:
   $$\gamma^{(l)} = \frac{\partial J}{\partial z^{(l)}} = \delta^{(l)} \odot \sigma'\left(z^{(l)}\right)$$
2. Compute the gradients with respect to the parameters of layer $l$:
   $$\frac{\partial J}{\partial W^{(l)}} = \gamma^{(l)} \left(a^{(l-1)}\right)^T, \quad \frac{\partial J}{\partial b^{(l)}} = \gamma^{(l)}$$
3. Backpropagate the gradient to the activations of the preceding layer $l-1$ for the next iteration:
   $$\delta^{(l-1)} = \frac{\partial J}{\partial a^{(l-1)}} = \left(W^{(l)}\right)^T \gamma^{(l)}$$

<div id="plotly-backprop-graph" class="plotly-chart" aria-label="Interactive Plotly diagram: computational graph of backpropagation with forward pass left-to-right and backward pass right-to-left"></div>

<p><em>Figure: Computational Graph of Backpropagation — the top row shows the forward pass (left → right): activations through weight matrices to the loss $J$. The bottom row shows the backward pass (right → left): error signals $\delta$ flow via transposed weight matrices, branching at each layer into the rank-1 parameter gradient $\frac{\partial J}{\partial W^{(l)}} = \gamma^{(l)} (a^{(l-1)})^T$. Hover any node for its formula.</em></p>

#### The "Markovian" Memory Trick
An essential computational property of backpropagation is that it is **Markovian**. Once the downstream error signal $\frac{\partial J}{\partial u}$ is calculated, all future nodes in the computational graph can be completely ignored. This local encapsulation enables PyTorch and other deep learning frameworks to organize computations as modular `forward` and `backward` methods per layer.

---

### 5. Advanced Mathematical Extensions & Optimization Theory

#### Hessian-Vector Products
In optimization, second-order curvature information is captured by the symmetric Hessian matrix $H \in \mathbb{R}^{N \times N}$, where $N$ is the parameter dimension. Computing, storing, or inverting the full Hessian is computationally intractable for large networks since it requires $O(N^2)$ memory and $O(N^3)$ operations.

However, we can compute the **Hessian-vector product** ($H v$) for an arbitrary vector $v$ in $O(N)$ time and memory without ever instantiating the full Hessian matrix.
1. Define a scalar auxiliary function $g(\theta)$ as the inner product of the gradient $\nabla J(\theta)$ and the vector $v$:
   $$g(\theta) = \left(\nabla_\theta J(\theta)\right)^T v$$
2. Compute the gradient of $g(\theta)$ with respect to $\theta$:
   $$\nabla_\theta g(\theta) = \nabla_\theta \left( \nabla_\theta J(\theta)^T v \right) = H(\theta) v$$
3. Since $g(\theta)$ is a scalar function that is efficiently computable, we can apply backpropagation a second time (differentiating through the gradient computation) to obtain $H v$ in $O(N)$ time.

#### First-Order vs. Second-Order Optimizers
Modern machine learning heavily favors first-order methods (SGD, Adam, AdaGrad) over second-order methods (Newton's Method) due to computational efficiency per step:
- **SGD Step Complexity**: $O(d)$ per parameter update, or $O(B \cdot d)$ for minibatch size $B$.
- **Newton's Method Step Complexity**: $O(n d^2 + d^3)$ due to Hessian computation and inversion.
- **Curvature Approximations**: Rather than running full second-order algorithms, optimizers like **AdaGrad** and **Adam** estimate diagonal approximations of the Hessian to dynamically scale the learning rate along different parameter dimensions, ensuring steps are larger in flat directions and smaller in steep directions.

---

### 6. Engineering & Systems Lore

- **Activation Caching vs. Memory Footprint**: During the backward pass, computing the parameter gradient requires access to the forward activation values ($\frac{\partial J}{\partial W} = \gamma a^T$). Consequently, all intermediate activations $a^{(l)}$ computed during the forward pass must be retained (cached) in GPU High Bandwidth Memory (HBM). This caching mechanism is the primary reason training deep networks consumes significantly more memory than running inference.
- **Frontier Lab Interviews**: Because backpropagation is the core engine running under the hood of every training run, deriving backpropagation from scratch (especially matrix gradients) remains a standard technical interview question at frontier AI research labs.

---

### 7. Applications
- **Automatic Differentiation Engines**: Backpropagation forms the backbone of PyTorch (`autograd`), JAX, and TensorFlow, enabling modern neural network libraries to compute exact gradients automatically.
- **Meta-Learning**: Differentiating through the optimization process itself (applying backpropagation through gradient descent updates) to meta-learn optimal weight initializations or adaptively tune learning rates.
- **Test-Time Training (TTT)**: Fine-tuning and optimizing neural network representations dynamically during model inference using self-supervised objectives.

---

### 8. Reflection Questions
1. **Rank-1 Property of Single-Example Gradients**: Why is the gradient of the weight matrix, $\frac{\partial J}{\partial W} = \gamma z^T$, a rank-1 matrix for a single training example? What happens mathematically to the rank of the aggregated gradient when we sum or average over a minibatch of size $B$?
2. **Computational Complexity of Hessians**: Explain why computing a Hessian-vector product $H v$ is asymptotically cheaper than computing the full Hessian matrix $H$. If our model has $10^9$ parameters, why is calculating $H$ impossible on modern hardware while calculating $Hv$ is completely feasible?
3. **The Activation Caching Bottleneck**: If a neural network is incredibly deep, explain the memory bottleneck that arises during training compared to inference. Which specific mathematical term in the parameter gradient derivation forces us to cache forward activations?

---

### 9. Further Reading & Resources
- **CS229 Friday TA Sections**: The essential venue for mastering advanced matrix and tensor calculus, vector-valued derivatives, and Kronecker products.
- **The Matrix Cookbook**: A highly recommended mathematical reference for vector-matrix derivative identities.
- **Adam: A Method for Stochastic Optimization (Kingma & Ba)**: For reading on how diagonal curvature estimation is combined with momentum to form the world's most popular deep learning optimizer.
