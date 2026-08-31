# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 4: Neural Networks and Backpropagation

#### 0. Quick-Recall Summary
*   **The Power of Non-Linearity:** Multi-layer networks (MLPs) stack linear transformations with element-wise non-linear activations; removing the activation collapses any deep stack of linear layers back into a single mathematically equivalent linear classifier.
*   **Hierarchical Part Templates:** Unlike linear models restricted to a single class template, a two-layer network with a hidden layer of size $H$ learns $H$ sub-templates representing reusable object parts (e.g., eyes, wheels) shared across classes.
*   **The Backprop Paradigm:** Backpropagation is a highly localized algorithm that decomposes a global loss calculation into a computational graph of simple, differentiable operators, passing upstream gradients backward to multiply with local gradients.
*   **Canonical Gate Behaviors:** Standard operations behave as fundamental gradient gates: the **Add gate** acts as a distributor, the **Multiply gate** acts as a scaled switcher, the **Copy gate** acts as a summer, and the **Max gate** acts as a router.
*   **Vectorized Gradient Scaling:** For matrix operations, explicitly computing high-dimensional Jacobian matrices (which can exceed 256 GB for typical batch/hidden dimensions) is avoided by using elegant dimension-matching transposed matrix multiplications.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To introduce the architecture of Multi-Layer Perceptrons (MLPs) and master the backpropagation algorithm. This lecture establishes how complex, high-dimensional functions can be optimized end-to-end by representing them as computational graphs of simple, locally differentiable operations.
*   **Lecture Category:** (a) Mathematical Foundations and (c) Training/Optimization Practice (gradient derivation and graph modularization).
*   **Builds on:** Lecture 2 and 3's scoring functions and gradient descent optimization landscapes.

---

#### 2. Mathematical Foundations

##### Score Function of a Two-Layer Neural Network (MLP)
The score function $f$ of a 2-layer neural network with a single hidden layer of size $H$ and non-linear activation is defined as:
$$f(x, W_1, W_2) = W_2 \max(0, W_1 x)$$
where $x \in \mathbb{R}^{D}$ is the input feature vector, $W_1 \in \mathbb{R}^{H \times D}$ represents the first-layer weight matrix, $W_2 \in \mathbb{R}^{C \times H}$ represents the second-layer weight matrix, and the $\max(0, \cdot)$ operation represents the element-wise Rectified Linear Unit (ReLU) activation. 
*(Note: Bias terms $b_1 \in \mathbb{R}^H$ and $b_2 \in \mathbb{R}^C$ are omitted here for algebraic simplicity but are fully incorporated in practice.)*

##### Hinge (SVM) Loss Form
The Multi-class Support Vector Machine (hinge) loss $L_i$ for a single training example is defined as:
$$L_i = \sum\_{j \neq y_i} \max(0, s_j - s\_{y_i} + \Delta)$$
where $s = f(x_i, W)$ represents the predicted class scores, $s\_{y_i}$ is the score of the correct class, $s_j$ are the scores of the incorrect classes, and $\Delta$ is the margin parameter (typically set to $1$).

##### Collapse of Stacked Linear Layers
If the activation function is removed from a two-layer network, the model collapses into a single linear classifier:
$$f(x) = W_2 (W_1 x) = (W_2 W_1) x = W_3 x$$
where $W_3 = W_2 W_1 \in \mathbb{R}^{C \times D}$. Since the product of two matrices is simply a new matrix, stacking arbitrary linear layers provides no additional representational capacity over a single linear hyperplane.

##### Modular Sigmoid Local Gradient
The Sigmoid activation function is defined as:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
Its local gradient with respect to $x$ is highly convenient as it can be formulated entirely as a function of its output:
$$\frac{d\sigma(x)}{dx} = \sigma(x) \cdot (1 - \sigma(x))$$

---

##### Step-by-Step Backpropagation Derivation (Scalar Example)
Let us trace the forward and backward pass for a single neuron using a sigmoid activation:
$$f(w, x) = \frac{1}{1 + e^{-(w_0 x_0 + w_1 x_1 + w_2)}}$$

We decompose this complex function into a computational graph of simple intermediate variables:
1.  **Multiplication 1:** $u_0 = w_0 x_0$
2.  **Multiplication 2:** $u_1 = w_1 x_1$
3.  **Addition 1:** $q = u_0 + u_1$
4.  **Addition 2 (Bias):** $a = q + w_2 = w_0 x_0 + w_1 x_1 + w_2$
5.  **Negation:** $b = -a$
6.  **Exponentiation:** $c = e^b$
7.  **Increment:** $d = 1 + c$
8.  **Reciprocal (Sigmoid Output):** $f = \frac{1}{d}$

###### Forward Pass Trace
Given inputs:
$$w_0 = 2.0, \quad x_0 = -1.0$$
$$w_1 = -3.0, \quad x_1 = -2.0$$
$$w_2 = -3.0$$

The forward intermediate values are computed as:
$$u_0 = 2.0 \times (-1.0) = -2.0$$
$$u_1 = -3.0 \times (-2.0) = 6.0$$
$$q = -2.0 + 6.0 = 3.0$$
$$a = 3.0 + (-3.0) = 0.0$$
$$b = -0.0 = 0.0$$
$$c = e^{0} = 1.0$$
$$d = 1.0 + 1.0 = 2.0$$
$$f = \frac{1}{2.0} = 0.5$$

###### Backward Pass Trace (Chain Rule)
Starting at the end of the network, the gradient of the output with respect to itself is defined as:
$$\frac{\partial f}{\partial f} = 1.0$$

Now we apply the chain rule step-by-step moving backward through the graph:
1.  **Reciprocal Gate ($f = 1/d$):**
    The local gradient is $\frac{\partial f}{\partial d} = -\frac{1}{d^2}$.
    $$\frac{\partial f}{\partial d} = 1.0 \times \left(-\frac{1}{2.0^2}\right) = -0.25$$

2.  **Increment Gate ($d = 1 + c$):**
    The local gradient is $\frac{\partial d}{\partial c} = 1$.
    $$\frac{\partial f}{\partial c} = \frac{\partial f}{\partial d} \frac{\partial d}{\partial c} = -0.25 \times 1 = -0.25$$

3.  **Exponentiation Gate ($c = e^b$):**
    The local gradient is $\frac{\partial c}{\partial b} = e^b$.
    $$\frac{\partial f}{\partial b} = \frac{\partial f}{\partial c} \frac{\partial c}{\partial b} = -0.25 \times e^{0.0} = -0.25 \times 1.0 = -0.25$$

4.  **Negation Gate ($b = -a$):**
    The local gradient is $\frac{\partial b}{\partial a} = -1$.
    $$\frac{\partial f}{\partial a} = \frac{\partial f}{\partial b} \frac{\partial b}{\partial a} = -0.25 \times (-1) = 0.25$$

5.  **Bias Addition Gate ($a = q + w_2$):**
    Addition acts as a gradient distributor (local gradients are $1$).
    $$\frac{\partial f}{\partial w_2} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial w_2} = 0.25 \times 1 = 0.25$$
    $$\frac{\partial f}{\partial q} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial q} = 0.25 \times 1 = 0.25$$

6.  **Sum Addition Gate ($q = u_0 + u_1$):**
    Again, addition distributes the upstream gradient.
    $$\frac{\partial f}{\partial u_0} = 0.25, \quad \frac{\partial f}{\partial u_1} = 0.25$$

7.  **Multiplication Gate 1 ($u_0 = w_0 x_0$):**
    The multiplication gate swaps the inputs and scales by the upstream gradient.
    $$\frac{\partial f}{\partial w_0} = \frac{\partial f}{\partial u_0} x_0 = 0.25 \times (-1.0) = -0.25$$
    $$\frac{\partial f}{\partial x_0} = \frac{\partial f}{\partial u_0} w_0 = 0.25 \times 2.0 = 0.50$$

8.  **Multiplication Gate 2 ($u_1 = w_1 x_1$):**
    $$\frac{\partial f}{\partial w_1} = \frac{\partial f}{\partial u_1} x_1 = 0.25 \times (-2.0) = -0.50$$
    $$\frac{\partial f}{\partial x_1} = \frac{\partial f}{\partial u_1} w_1 = 0.25 \times (-3.0) = -0.75$$

---

##### Vectorized & Matrix Backpropagation Derivations
When variables are matrices or tensors rather than scalars, the gradient of a scalar loss $L$ with respect to a tensor must have the exact same shape as the tensor itself:
$$\dim\left(\frac{\partial L}{\partial X}\right) = \dim(X)$$

###### Matrix Multiplication Gate
Let us define the forward matrix multiplication operation:
$$Y = X W$$
where $X \in \mathbb{R}^{N \times D}$, $W \in \mathbb{R}^{D \times M}$, and $Y \in \mathbb{R}^{N \times M}$. Given the upstream gradient $\frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times M}$ coming from the subsequent layer, we derive the downstream gradients using dimension-matching matrix algebra:

1.  **Gradient with respect to weights $W$:**
    To match the shape of $W$ ($D \times M$), we must multiply the transpose of the input $X^T$ ($D \times N$) by the upstream gradient $\frac{\partial L}{\partial Y}$ ($N \times M$):
    $$\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}$$

2.  **Gradient with respect to inputs $X$:**
    To match the shape of $X$ ($N \times D$), we must multiply the upstream gradient $\frac{\partial L}{\partial Y}$ ($N \times M$) by the transpose of the weight matrix $W^T$ ($M \times D$):
    $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T$$

These formulas form the computational foundation of backpropagation through Fully Connected/Linear layers.

---

#### 3. Architecture / Algorithm Walkthrough

##### Algorithmic Logic (Modular Forward/Backward APIs)
Modern deep learning frameworks modularize backpropagation by defining each operation (e.g., addition, matrix multiplication, ReLU) as an independent computational node containing both a `forward` and `backward` API:
*   **The `forward` method:** Receives input tensors, computes the output tensor, and **caches** any intermediate variables necessary to compute gradients during the backward pass (e.g., caching input values in a multiplication gate).
*   **The `backward` method:** Receives the upstream gradient ($\frac{\partial L}{\partial \text{out}}$), calculates the local gradients ($\frac{\partial \text{out}}{\partial \text{in}}$), and returns the downstream gradients as their product ($\frac{\partial L}{\partial \text{in}} = \frac{\partial L}{\partial \text{out}} \times \frac{\partial \text{out}}{\partial \text{in}}$).

```
                FORWARD PASS (Left to Right)
     X ───> ┌───────────────────────────┐ ───> Y (Output)
     W ───> │     Computational Node    │
            │  (Saves inputs in cache)  │
     dX <── ├───────────────────────────┤ <─── dY (Upstream Gradient)
     dW <── └───────────────────────────┘
                BACKWARD PASS (Right to Left)
```

##### PyTorch Blueprint (Illustrative Custom Autograd Modular Stack)
This PyTorch implementation demonstrates custom modular forward and backward passes for both a linear matrix multiplication layer and a ReLU activation layer, highlighting input caching and manual gradient calculations.

```python
import torch

class CustomLinear(torch.autograd.Function):
    # Illustrative fc linear layer with transposes to match shapes
    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X, W, b)
        return torch.mm(X, W) + b

    @staticmethod
    def backward(ctx, grad_upstream):
        X, W, b = ctx.saved_tensors
        grad_X = torch.mm(grad_upstream, W.t())   # Shape: (N, D) = (N, M) * (M, D)
        grad_W = torch.mm(X.t(), grad_upstream)   # Shape: (D, M) = (D, N) * (N, M)
        grad_b = grad_upstream.sum(dim=0, keepdim=True)
        return grad_X, grad_W, grad_b


class CustomReLU(torch.autograd.Function):
    # Illustrative element-wise ReLU activation layer
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        return torch.clamp(X, min=0)

    @staticmethod
    def backward(ctx, grad_upstream):
        X, = ctx.saved_tensors
        mask = (X > 0).float()
        grad_X = grad_upstream * mask
        return grad_X


class ModularMLP:
    def __init__(self, d_in, h, d_out):
        self.W1 = torch.randn(d_in, h) * 0.01
        self.b1 = torch.zeros(1, h)
        self.W2 = torch.randn(h, d_out) * 0.01
        self.b2 = torch.zeros(1, d_out)

    def forward_backward_pipeline(self, X, Y_target):
        # Forward Pass
        h_linear = CustomLinear.apply(X, self.W1, self.b1)
        h_activated = CustomReLU.apply(h_linear)
        scores = CustomLinear.apply(h_activated, self.W2, self.b2)
        
        # Simple MSE loss: L = 0.5 * sum((scores - target)^2)
        loss = 0.5 * torch.sum((scores - Y_target) ** 2)
        print(f"Forward Pass Completed. Loss: {loss.item():.4f}")
        
        grad_scores = scores - Y_target
        
        # Backward Pass (manual chain rule propagation)
        grad_h_activated, grad_W2, grad_b2 = CustomLinear.backward(None, grad_scores) # Dummy backward trigger
        grad_h_linear = CustomReLU.backward(None, grad_h_activated)
        grad_X, grad_W1, grad_b1 = CustomLinear.backward(None, grad_h_linear)
        
        return grad_W1, grad_W2
```

---

#### 4. Visual Intuition & Interpretability

##### The Visual Transformation of Non-Separable Spaces
In linear classification, a single linear hyperplane cannot separate complex layouts (such as concentric circles or an XOR distribution). The non-linear activation in a neural network (e.g., ReLU) performs a non-linear warping of the input space. As illustrated below, this warps the raw coordinates such that the classes become linearly separable in the subsequent layer's feature space:

```
      NON-LINEAR SPACE (Raw Input)           LINEARLY SEPARABLE FEATURE SPACE
            o   o   o                                  o   o   o
          o   x   x   o                              ─────────────────────── (Boundary)
            o   o   o                                  x   x   x
     (No single line can separate)             (Line easily separates classes)
```

##### Part-Based Reusable Templates
In linear classifiers, the model is restricted to learning exactly one holistic template per class (e.g., the "car template" resembles a red blob). In a Multi-Layer Percepton, the hidden layer neurons (e.g., $100$ neurons) can act as specialized "part detectors" (e.g., detecting circular wheels, horizontal edges, shiny metallic textures, or eyes). Because classes share common visual parts (e.g., both dogs and cats have eyes), the network can compose these part-based templates hierarchically to represent highly complex, multimodal class structures.

##### Hidden Layer Decision Boundary Geometry
Visualizing the decision boundary of a neural network shows that as the number of hidden neurons increases, the model's decision boundaries become increasingly complex and jagged:
*   **Small Hidden Layer (e.g., $H=3$):** Smooth, simple, polygonal-like decision boundaries. High bias, low variance.
*   **Large Hidden Layer (e.g., $H=100$):** High capacity, allowing the boundary to wrap tightly around individual data points, capturing fine-grained noise and risking overfitting.

---


<div id="plotly-cs231n-4-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 4 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Modular Backprop Flow Graph)
To visualize how gradients flow and scale dynamically across different gates, we propose a **Computational Graph Backprop Simulator**:

*   **Visualization Type:** Directed Acyclic Graph (DAG) node-link layout.
*   **Data Fields & Encoding:**
    *   **Node Color:** Encodes operation type (Blue = Add, Orange = Multiply, Green = Activation/Sigmoid).
    *   **Link Width:** Proportional to the absolute magnitude of the value flowing through the link.
    *   **Link Color:** Encodes value sign (Green = Positive, Red = Negative).
    *   **Floating Text Over links:** Dual values showing the forward activation value (top, e.g., `x=3.0`) and the backward gradient value (bottom, e.g., `dx=-4.0`).
*   **Interactive Controls:**
    *   **Input Sliders:** Real-time sliders to change the raw inputs ($w_0, x_0, w_1, x_1, w_2$). Changing the slider instantly triggers the forward pass recalculation, showing link widths updating.
    *   **Gate Selector:** Dropdown to toggle the activation function in the middle layer (Sigmoid vs. ReLU vs. Leaky ReLU). This dynamically displays how the backward gradients are affected (e.g., selecting Sigmoid with high inputs immediately turns link colors in early layers to faint gray, visually demonstrating the *vanishing gradient* problem).

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **ReLU as the Default Activator:** Rectified Linear Units ($\max(0, x)$) are the standard empirical starting point for hidden layers due to their cheap computational cost (a simple threshold comparison) and linear gradient flow for positive values.
*   **Vanishing Gradient Bounds of Sigmoid/Tanh:** Traditional sigmoidal activations squash real numbers into narrow ranges ($[0, 1]$ for sigmoid, $[-1, 1]$ for tanh). When inputs are saturated (very positive or very negative), their local gradients approach zero, terminating gradient propagation to earlier layers during backprop.
*   **Capacity Tuning Rule of Thumb:** Never decrease hidden layer size to prevent overfitting. Instead, use a larger network architecture and strictly control generalization bounds by tuning the regularization strength parameter ($\lambda$).
*   **The Adam 10-Year Milestone:** Highlighting the impact of optimization techniques discussed, the foundational **Adam Optimizer** paper (originally published in 2015) was formally awarded the prestigious **ICLR 2025 Test of Time Award**.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Common Implementation Bugs
*   **The "Dead ReLU" Neuron:** If a neuron is initialized such that it outputs negative values across the entire data distribution, or if a large gradient update knocks it into this regime, its gradient becomes exactly zero. Once a ReLU neuron is "dead," its weights will never update again because it always passes a zero gradient during the backward pass.
*   **Vanishing Sigmoid Gradients:** Saturated activations scale upstream gradients by near-zero local gradients, leaving early layer weights completely unchanged. Always ensure input data is normalized to avoid immediately entering the saturated regimes of sigmoid/tanh.
*   **Transposed Weight Shape Alignment:** In vectorized linear layers ($Y = XW$), a common silent bug is transposing the weight matrix incorrectly during backward updates. Always verify that the dimensions of $\frac{\partial L}{\partial W}$ exactly match $W$ ($D \times M$), and $\frac{\partial L}{\partial X}$ exactly match $X$ ($N \times D$).

##### Graduate-Level Reflection Questions
1.  **Analytical Proof of Stacked Linearity:** Prove mathematically that a 100-layer neural network with weight matrices $W_1, W_2, \dots, W\_{100}$ and no activation functions has the exact same representational capacity as a single linear layer classifier.
2.  **The Jacobian Memory Bottleneck:** Suppose you are training a linear layer with batch size $N = 100$, input dimension $D = 4096$, and output dimension $M = 4096$. 
    *   What are the dimensions of the Jacobian matrix $\frac{\partial Y}{\partial X}$?
    *   Why is storing this Jacobian computationally prohibitive (calculate the memory footprint in gigabytes assuming FP32), and how does the backprop equation $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T$ completely bypass this bottleneck?
3.  **ELU vs. Leaky ReLU Symmetrical Advantages:** Why are zero-centered activation functions (such as the Exponential Linear Unit, ELU) mathematically preferred for optimization stability over standard asymmetric ReLUs, and how does this affect the mean activation shifts across deep layers?
