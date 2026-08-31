# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 5: Image Classification with CNNs

#### 0. Quick-Recall Summary
*   **The Spatial Preservation Axiom:** Unlike fully connected layers that flatten inputs and destroy 2D spatial relationships, 2D convolutions preserve spatial coordinates ($H \times W$) by performing local template matching sliding across spatial channels.
*   **The Linear Composition Bug:** Stacking multiple convolution layers without interspersed nonlinear activation functions collapses the representational power to a single linear operation due to the associativity of linear operators.
*   **Receptive Field Dilation:** The effective receptive field (the area of the original input affecting a specific activation) grows linearly with convolutional depth but can be scaled exponentially using strided convolutions or pooling.
*   **Computational Disparity:** Convolutions require orders of magnitude more multiply-accumulate (MAC) operations than fully connected layers, yet they achieve drastic parameter savings by sharing weights across spatial locations.
*   **Translation Equivariance:** The mathematical structure of convolution and pooling guarantees translation equivariance ($\text{Translate} \circ \text{Conv} = \text{Conv} \circ \text{Translate}$), baking spatial shift invariance into the network's architectural fabric.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To transition from standard Multi-Layer Perceptrons (MLPs) to Convolutional Neural Networks (CNNs). This lecture details the design of 2D convolutional and pooling primitives, derives their spatial arithmetic, and demonstrates how they exploit local spatial structures to learn robust hierarchical visual representations.
*   **Lecture Category:** (b) Architecture Design and (a) Mathematical Foundations.
*   **Builds on:** Lecture 4's Backpropagation and Computational Graphs, scaling up modular matrix operations to spatially-aware sliding operators.

---

#### 2. Mathematical Foundations

##### Spatial Dimension Propagation Formulas
For an input tensor of spatial dimensions $W \times H$, a convolutional kernel of size $K_w \times K_h$, padding $P$, and stride $S$, the output spatial dimensions $W'$ and $H'$ are defined as:
$$W' = \left\lfloor \frac{W - K_w + 2P}{S} \right\rfloor + 1$$
$$H' = \left\lfloor \frac{H - K_h + 2P}{S} \right\rfloor + 1$$

*   **Boundary Preserving Padding:** To ensure the output spatial dimensions match the input ($W' = W$) when stride $S=1$, the padding $P$ must be set as a function of an odd-sized kernel $K$:
    $$P = \frac{K - 1}{2}$$

##### 4D Convolution Tensor Formulation (Batched Mode)
Given a batch of $N$ images with $C\_{in}$ channels, spatial dimensions $H \times W$:
*   **Input Tensor $X$:** Shape $(N, C\_{in}, H, W)$.
*   **Filter Bank Matrix $W$:** Shape $(C\_{out}, C\_{in}, K_h, K_w)$ where $C\_{out}$ is the number of filters.
*   **Bias Vector $b$:** Shape $(C\_{out},)$.
*   **Output Tensor $Y$:** Shape $(N, C\_{out}, H', W')$.

For a single sample index $n \in [1, N]$, output channel $c \in [1, C\_{out}]$, and output coordinate $(i, j)$:
$$Y[n, c, i, j] = b[c] + \sum\_{ch=1}^{C\_{in}} \sum\_{ki=1}^{K_h} \sum\_{kj=1}^{K_w} X[n, ch, i \cdot S + ki, j \cdot S + kj] \cdot W[c, ch, ki, kj]$$

##### Gradient Propagation through 2D Convolutions
During backpropagation, we compute the gradient of the loss $L$ with respect to the filter weights $W$ and the input activations $X$. For simplicity of notation, let $S=1, P=0$:
$$\frac{\partial L}{\partial W[c, ch, ki, kj]} = \sum\_{n=1}^{N} \sum\_{i=1}^{H'} \sum\_{j=1}^{W'} \frac{\partial L}{\partial Y[n, c, i, j]} \cdot X[n, ch, i + ki, j + kj]$$
$$\frac{\partial L}{\partial X[n, ch, r, s]} = \sum\_{c=1}^{C\_{out}} \sum\_{ki=1}^{K_h} \sum\_{kj=1}^{K_w} \frac{\partial L}{\partial Y[n, c, r - ki, s - kj]} \cdot W[c, ch, ki, kj]$$
*(This mathematically represents a transposed convolution of the upstream gradient with the filter weights).*

##### Translation Equivariance Operator Proof
An operator $f$ is equivariant to a translation operator $g_t$ if:
$$f(g_t(x)) = g_t(f(x))$$
Let $g_t(x)[n, c, i, j] = x[n, c, i - t_y, j - t_x]$ be the translation operator. Applying convolution $f$:
$$f(g_t(x))[n, c, i, j] = \sum\_{ch, ki, kj} g_t(x)[n, ch, i + ki, j + kj] \cdot W[c, ch, ki, kj]$$
$$= \sum\_{ch, ki, kj} x[n, ch, (i - t_y) + ki, (j - t_x) + kj] \cdot W[c, ch, ki, kj]$$
$$= f(x)[n, c, i - t_y, j - t_x] = g_t(f(x))[n, c, i, j]$$
Thus proving translation equivariance analytically.

---

#### 3. Architecture / Algorithm Walkthrough

##### Data Flow & Layer Interspersal
In a standard CNN architecture, the spatial representation undergoes a progressive transition: spatial width and height decrease ($H \downarrow, W \downarrow$) while the channel depth increases ($C \uparrow$).

```
Raw Image Tensor: (3, 32, 32)
   │
   ▼   [Convolution: 6 filters, 5x5, stride 1, padding 2]
Activation Tensor: (6, 32, 32)
   │
   ▼   [ReLU Activation: Pointwise Non-linearity]
Non-linear Tensor: (6, 32, 32)
   │
   ▼   [Max Pooling: 2x2, stride 2]
Downsampled Tensor: (6, 16, 16)
   │
   ▼   [Fully Connected Layer: Reshape to Vector & Project]
Logit Vector: (10, )
```

##### PyTorch Blueprint (Custom 2D Convolution & Pooling Hierarchy)
This blueprint implements a modular CNN reflecting the classical `Conv -> ReLU -> Pool` sequence and details how dimensions are calculated mathematically.

```python
import torch
import torch.nn as nn

class CustomConvNet(nn.Module):
    """
    Illustrative blueprint modeling spatial preservation, non-linear activation,
    and spatial pooling downsampling.
    """
    def __init__(self, in_channels=3, out_channels=10, num_classes=10):
        super(CustomConvNet, self).__init__()
        
        # Hyperparameters
        self.kernel_size = 5
        self.stride = 1
        self.padding = 2 # P = (K-1)/2 preserves spatial size for stride=1
        
        # 1. 2D Convolution Layer (Preserves spatial resolution)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True # 1 scalar bias learned per filter
        )
        
        # 2. ReLU Non-linearity (Solves the Linear Composition Bug)
        self.relu = nn.ReLU(inplace=True)
        
        # 3. Max Pooling (Downsamples spatial dimensions by half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 stride 2
        
        # 4. Fully Connected Output Head (Class scores projection)
        # Input size: 32x32 -> Pooled size: 16x16
        self.fc = nn.Linear(out_channels * 16 * 16, num_classes)

    def forward(self, x):
        # x shape: [N, 3, 32, 32]
        out = self.conv(x)      # Shape: [N, 10, 32, 32]
        out = self.relu(out)    # Shape: [N, 10, 32, 32]
        out = self.pool(out)    # Shape: [N, 10, 16, 16]
        
        # Flatten spatial representation to vector
        out = torch.flatten(out, start_dim=1) # Shape: [N, 10 * 16 * 16]
        
        # Map to class scores
        logits = self.fc(out)   # Shape: [N, 10]
        return logits
```

---

#### 4. Visual Intuition & Interpretability

##### The Hierarchical Feature Pyramid
Through backpropagation and gradient descent, the model automatically differentiates and specializes filters without human manual feature engineering.
1.  **Early Layers (Low-Level Primitives):** Learn simple edge detectors at various orientations (Gabor-like filters), color-blobs, and opposing color boundaries (e.g., red/green, blue/yellow contrasts).
2.  **Mid-Level Layers (Part Abstractions):** Combine early-layer lines and spots to respond to visual corners, circular contours, textures, and repeating geometric grids.
3.  **High-Level Layers (Object Semantic Templates):** Receptive fields expand to view large spatial structures, specializing to detect highly complex, abstract semantic concepts such as eyes, text characters, animal paws, or car wheels.

##### Receptive Field Mechanics (Visualized)
As layers stack sequentially, the effective receptive field grows linearly with depth.

```
Input Image (32x32)
   ███████  <── 7x7 Patch in Input
     \ /
    A1 (3x3) <── 5x5 Patch in Layer 1
     \ /
    A2 (3x3) <── 3x3 Patch in Layer 2
     \ /
    A3 (1x1) <── Single activation in Layer 3 sees a 7x7 input area
```
A stack of three $3\times3$ convolutions with stride 1 has the same effective receptive field as a single $7\times7$ convolution but is significantly more non-linear and parameter-efficient.

##### Translation Equivariance Commutative Diagram
The spatial alignment of local kernels guarantees that shifting an object in the input results in an identical shift of its feature representations in the output activation maps.

```
Input Image (X)  ───[ Translate by t ]───>  Shifted Image (g_t(X))
       │                                           │
   [ Conv(f) ]                                 [ Conv(f) ]
       │                                           │
       ▼                                           ▼
Feature Map (Y)  ───[ Translate by t ]───>  Shifted Map (g_t(Y))
```
*The output is identical whether you shift-then-convolve or convolve-then-shift.*

---


<div id="plotly-cs231n-5-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 5 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Interactive Kernel & Feature Map Slicer)
*To build outside of this notebook environment, utilize the following specifications:*
*   **Purpose:** To demonstrate how changing convolutional hyperparameters ($K, P, S$) transforms output activations and parameter counts in real-time.
*   **Interactive Panel Components:**
    *   **Input Image Grid:** Upload custom $H \times W$ images.
    *   **Hyperparameter Sliders:** Kernel Size ($K \in$), Padding ($P \in$), Stride ($S \in$), and Filter Count ($C\_{out} \in$).
    *   **Dynamic Output Indicators:** Parameter Count, Total Floating Point Operations (FLOPs), and Receptive Field Size.
*   **Interactive Graphics:**
    *   **3D Tensor Block:** Displays a 3D volumetric representation of the input $X$ and output $Y$.
    *   **Sliding 2D Overlay:** Hovering over any element in the output volume dynamically highlights the exact patch of input pixels that computed it (the receptive field), illustrating the boundary effects of Padding and Stride.

---

#### 6. Empirical Design Heuristics & Benchmark Results

##### Convolutional vs. Fully Connected Parameter/FLOP Scaling
To highlight the efficiency of weight sharing, consider a toy calculation of an input volume of $3 \times 32 \times 32$ mapped to an output of size $10 \times 32 \times 32$:

| Metric | Fully Connected Layer ($W \times X$) | Convolutional Layer (10 filters, $5\times5$, $S=1, P=2$) |
| :--- | :--- | :--- |
| **Number of Output Neurons** | $10 \times 32 \times 32 = 10,240$ | $10 \times 32 \times 32 = 10,240$ |
| **Connection Scheme** | Global connectivity (unshared templates) | Local connectivity (shared template matching) |
| **Learnable Parameters** | $10,240 \times 3,072 = 31,457,280$ | $10 \times (3 \times 5 \times 5 + 1 \text{ bias}) = 760$ |
| **Total MAC Operations** | $10,240 \times 3,072 \approx 31.45 \times 10^6$ FLOPs | $10,240 \times 75 = 768,000$ FLOPs |

**Key Takeaway:** The Convolutional layer reduces the parameter footprint by **$41,391 \times$** and compute cost by **$41 \times$** compared to the fully connected equivalent, while strictly respecting spatial structure.

##### CNN Historical Milestones on ImageNet
*   **ImageNet Complexity:** Tracing the scale from Caltech-101 (unbalanced, small scale) to ImageNet's **15 million images** across **22,000 categories**.
*   **2012 AlexNet Moment:** Surpassed traditional hand-engineered pipelines (which plateaued at a ~30% top-5 error rate) by implementing Yann LeCun's 1998 backprop-optimized network scaled across parallelized GPUs, cutting the error nearly in half to **~15.3%**.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Critical Architecture Gotchas
*   **The Dimensionality Mismatch Trap:** Specifying invalid hyperparameters that do not divide evenly in the spatial sizing formula. For example, if $W=10, K=3, P=0, S=2$:
    $$W' = \frac{10 - 3 + 2(0)}{2} + 1 = 4.5$$
    In PyTorch, fractional dimension results will trigger a compile-time or runtime exception depending on the framework, demanding careful coordinate padding design.
*   **The Zero-Padding Boundary Artifact:** Appending zeros to the boundaries of an image ($P > 0$) causes convolutional kernels to consistently receive inactive, artificial values at borders. Deeper networks can learn to over-rely on these border artifacts, resulting in spatial localization biases.
*   **The Loss-of-Symmetry Pitfall:** Initializing multiple convolutional filters to identical or constant weights. Because the backpropagation gradient is uniform across symmetric nodes, the filters will compute the exact same gradient updates, failing to specialize and collapsing model capacity back to a single filter.

##### Graduate-Level Reflection Questions
1.  **Symmetry in Convolutional Stacking:** Prove that a stack of three $3\times3$ convolutions with stride 1 and padding 1 preserves the exact spatial dimensions of the input. What is the parameter savings ratio compared to a single $7\times7$ convolution if the channel depth is a constant $C$ throughout? Derive this parameter scaling ratio in terms of $C$.
2.  **Vanishing Gradients in Nonlinear Pooling:** While Max Pooling introduces pointwise non-linearities, it is a non-differentiable operator at its boundaries. Analyze the backward-pass gradient of a Max Pooling layer. What is the gradient of the loss with respect to non-maximal activation values in a pooled grid, and how does this affect gradient sparsity in extremely deep networks?
3.  **Equivariance vs. Invariance:** We mathematically proved that 2D convolutions are equivariant to spatial translation. However, image classification requires the final class scores to be strictly translation *invariant* (the class label remains 'cat' regardless of where the cat sits in the frame). What architectural mechanism at the end of a CNN breaks translation equivariance and transforms it into translation invariance?
