# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes
## Lecture 10: Video Understanding

### 0. Quick-Recall Summary
*   **4D Spatiotemporal Volume:** Video data adds a temporal dimension $T$, yielding a tensor of shape $C \times T \times H \times W$ which introduces massive storage and computational complexity ($1.5\text{ GB/min}$ for SD, $10\text{ GB/min}$ for HD uncompressed).
*   **Early vs. Late Fusion:** Late fusion processes frames independently via 2D CNNs and aggregates features at the end, losing fine-grained early motion cues. Early fusion concatenates frames into the input channel dimension $3T \times H \times W$ at the first layer, destroying temporal shift invariance.
*   **3D Convolutions (Slow Fusion):** Slow fusion gradually mixes space and time via 3D kernels of size $K_t \times K_h \times K_w$, preserving temporal shift equivariance and learning spatiotemporal patterns.
*   **Two-Stream Networks:** Decomposes video understanding into a spatial stream (appearance from static RGB frames) and a temporal stream (motion from computed dense optical flow fields $d_x, d_y$).
*   **I3D (Inflation Trick):** Converts pre-trained 2D CNN architectures (e.g., Inception) to 3D by copying 2D filters along the temporal axis $K_t$ times and dividing the weights by $K_t$, bootstrapping 3D learning with 2D image priors.
*   **Non-Local Networks:** Formulates spatiotemporal self-attention on $C \times T \times H \times W$ feature maps as a standalone block to model long-range global context in a single layer, avoiding sequential/receptive-field bottlenecks.

---

### 1. Core Concept & Learning Objectives
*   **Objective:** This lecture explores the expansion of deep learning architectures from static 2D images to the temporal dimension of video. It teaches the mathematical and structural trade-offs required to process time, handle high computational demands, explicitly represent motion (via optical flow), and scale model capacity using self-attention and weights pre-trained on large-scale image datasets.
*   **Lecture Category:** Specific vision task (Video Understanding) blended with Architecture Design (3D CNNs, Two-Stream, Non-Local Blocks).
*   **Builds on:** Builds on Lecture 5 & 6 (Image Classification with CNNs & Architectures), Lecture 7 (Recurrent Neural Networks), and Lecture 8 (Attention and Transformers).

---

### 2. Mathematical Foundations

#### A. Spatiotemporal Input Dimensions
A video clip is represented as a 4D tensor:
$$X \in \mathbb{R}^{C \times T \times H \times W}$$
where $C$ is the number of color channels, $T$ is the temporal length (number of frames), and $H, W$ are the spatial height and width.

#### B. Late Fusion Architectures
For $T$ frames, features are extracted independently per frame using a shared 2D CNN:
$$u_t = \text{CNN}\_{2D}(X_t) \in \mathbb{R}^{H' \times W' \times D} \quad \forall t \in \{1, \dots, T\}$$

1.  **Late Fusion by Concatenation:**
    Feature maps are flattened and concatenated across time:
    $$u\_{concat} = [u_1, u_2, \dots, u_T] \in \mathbb{R}^{T \cdot H' \cdot W' \cdot D}$$
    $$y = \text{MLP}(u\_{concat}) \in \mathbb{R}^C \quad (\text{High parameter footprint})$$
2.  **Late Fusion by Temporal Pooling:**
    Temporal aggregation is performed via average or max pooling across the temporal dimension:
    $$u\_{pool} = \frac{1}{T} \sum\_{t=1}^T u_t \in \mathbb{R}^{H' \times W' \times D}$$
    $$y = \text{Linear}(\text{Flatten}(u\_{pool})) \in \mathbb{R}^C$$

#### C. Early Fusion Architectures
The temporal sequence is collapsed directly into the channel dimension in the very first layer:
$$X\_{early} = \text{Reshape}(X) \in \mathbb{R}^{(C \cdot T) \times H \times W}$$
The first 2D convolution kernel $W\_{early} \in \mathbb{R}^{D \times (C \cdot T) \times K_h \times K_w}$ processes the entire temporal window at once:
$$h_1 = W\_{early} * X\_{early} + b \in \mathbb{R}^{D \times H' \times W'}$$
This destroys the temporal dimension immediately, preventing the model from achieving *temporal shift invariance*.

#### D. 3D Convolution (Slow Fusion)
A 3D convolution layer uses kernels that extend across both spatial and temporal dimensions. Let $K_t$ be the temporal kernel size, and $K_h, K_w$ be the spatial kernel sizes. For an input $X \in \mathbb{R}^{C\_{in} \times T \times H \times W}$, the pre-activation at a specific coordinate $(t, y, x)$ for output channel $c\_{out}$ is:
$$A(c\_{out}, t, y, x) = \sum\_{c=1}^{C\_{in}} \sum\_{i=-K'_t}^{K'_t} \sum\_{j=-K'_h}^{K'_h} \sum\_{k=-K'_w}^{K'_w} W(c\_{out}, c, i, j, k) \cdot X(c, t+i, y+j, x+k) + b(c\_{out})$$
where $K'_t = \frac{K_t - 1}{2}$, $K'_h = \frac{K_h - 1}{2}$, and $K'_w = \frac{K_w - 1}{2}$ (assuming odd kernel dimensions).
The output tensor has shape $C\_{out} \times T' \times H' \times W'$.

#### E. Dense Optical Flow
Optical flow measures the displacement vector field $(dx, dy)$ of pixels between two adjacent frames $I_t$ and $I\_{t+1}$:
$$I(x, y, t) = I(x + dx, y + dy, t + 1)$$
Assuming brightness constancy, the linearized optical flow constraint is:
$$\frac{\partial I}{\partial x} \frac{dx}{dt} + \frac{\partial I}{\partial y} \frac{dy}{dt} + \frac{\partial I}{\partial t} = 0$$
The horizontal flow component $d_x$ and vertical flow component $d_y$ are extracted as separate spatial channels and stacked across a sequence of $L$ frames to form a motion volume $F \in \mathbb{R}^{2L \times H \times W}$ which acts as input to the temporal stream.

#### F. Non-Local Block (Spatiotemporal Self-Attention)
Given an input feature map $X \in \mathbb{R}^{C \times T \times H \times W}$, queries $Q$, keys $K$, and values $V$ are computed via $1 \times 1 \times 1$ 3D convolutions:
$$Q(X) = W_q * X \in \mathbb{R}^{C' \times T \times H \times W}$$
$$K(X) = W_k * X \in \mathbb{R}^{C' \times T \times H \times W}$$
$$V(X) = W_v * X \in \mathbb{R}^{C' \times T \times H \times W}$$
The feature maps are reshaped into 2D matrices where $N = T \cdot H \cdot W$ is the total spatiotemporal position index:
$$\bar{Q} \in \mathbb{R}^{C' \times N}, \quad \bar{K} \in \mathbb{R}^{C' \times N}, \quad \bar{V} \in \mathbb{R}^{C' \times N}$$
The pairwise spatiotemporal affinity matrix is:
$$E = \bar{Q}^T \bar{K} \in \mathbb{R}^{N \times N}$$
$$\text{Attention}(X) = \text{Softmax}\left( \frac{\bar{Q}^T \bar{K}}{\sqrt{C'}} \right) \in \mathbb{R}^{N \times N}$$
The output $Y \in \mathbb{R}^{C \times T \times H \times W}$ incorporates a residual connection and a linear projection $W_z$:
$$Y = W_z * \text{Reshape}\left( \bar{V} \cdot \text{Attention}(X)^T \right) + X$$
where $W_z \in \mathbb{R}^{C \times C' \times 1 \times 1 \times 1}$ maps the channel dimension back to $C$.

---

### 3. Architecture / Algorithm Walkthrough

#### A. Structural Comparison of Fusions
1.  **Late Fusion:** Builds spatial receptive fields slowly through 2D conv/pooling. Temporal information is kept in disjoint parallel pathways and collapsed *all at once* in the final layers.
2.  **Early Fusion:** Collapses the temporal dimension in the *very first layer*, processing the temporal sequence as high-dimensional static channels. Spatial receptive fields are built up slowly.
3.  **Slow Fusion (3D CNN):** Gradually builds both spatial and temporal receptive fields. Kernels slide across both space and time, enabling temporal shift invariance/equivariance and slow spatiotemporal abstraction.

```
 Late Fusion:      [Frame 1] -> [2D CNN] ----\
                   [Frame 2] -> [2D CNN] ----+--> [Concatenate / Pool] -> [Classifier]
                   [Frame T] -> [2D CNN] ----/
 
 Early Fusion:     [F1, F2, ..., FT] (Concatenated) -> [1st 2D Conv] -> [Standard 2D CNN] -> [Classifier]
 
 Slow Fusion:      [F1, F2, ..., FT] -> [3D Conv] -> [3D Pool] -> [3D Conv] -> [AvgPool] -> [Classifier]
```

#### B. PyTorch Blueprint: 3D Convolution & Spatiotemporal Non-Local Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    """
    Illustrative 3D Convolutional Network showing progressive spatiotemporal fusion.
    """
    def __init__(self, num_classes=101):
        super().__init__()
        # Input shape: (B, C, T, H, W) -> (B, 3, 16, 112, 112)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, 
                               kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) # Downsamples T, H, W by 2
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.fc = nn.Linear(32 * 4 * 28 * 28, num_classes) # Assuming T=16, H=W=112 initially

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc(x)
        return logits

class SpatiotemporalNonLocalBlock(nn.Module):
    """
    PyTorch implementation of the Non-Local block for spatiotemporal self-attention.
    Input shape: (B, C, T, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        
        # 1x1x1 convolutions for Query, Key, and Value projections
        self.g_value = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.theta_query = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.phi_key = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        
        # Output projection back to input channels
        self.W_z = nn.Conv3d(self.inter_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, t, h, w = x.size()
        
        # Project and reshape to 2D matrices: (B, C', N) where N = T * H * W
        g_x = self.g_value(x).view(batch_size, self.inter_channels, -1) # (B, C', N)
        g_x = g_x.permute(0, 2, 1) # (B, N, C')
        
        theta_x = self.theta_query(x).view(batch_size, self.inter_channels, -1) # (B, C', N)
        theta_x = theta_x.permute(0, 2, 1) # (B, N, C')
        
        phi_x = self.phi_key(x).view(batch_size, self.inter_channels, -1) # (B, C', N)
        
        # Pairwise similarity matrix: (B, N, N)
        energy = torch.bmm(theta_x, phi_x) # Q * K^T
        
        # Scale dot-product to stabilize gradient flow
        scaled_energy = energy / (self.inter_channels ** 0.5)
        
        # Softmax over columns to normalize weights
        attention_map = F.softmax(scaled_energy, dim=-1) # (B, N, N)
        
        # Weighted combination of values: (B, N, C')
        out_matmul = torch.bmm(attention_map, g_x)
        out_matmul = out_matmul.permute(0, 2, 1).contiguous() # (B, C', N)
        
        # Reshape back to 5D: (B, C', T, H, W)
        out_spatial = out_matmul.view(batch_size, self.inter_channels, t, h, w)
        
        # Output projection + residual connection
        out = self.W_z(out_spatial) + x
        return out
```

---

### 4. Visual Intuition & Interpretability

#### A. The Point-Light Display Demonstration
*   **Demonstration:** A visualization of human movement shown purely through a few moving coordinates (points at joints, e.g., knees, elbows) with all appearance and background information removed.
*   **Insight:** Humans instantly recognize complex actions (e.g., walking, jumping) from pure sparse motion vectors. This proves that motion is a powerful, decoupled, and self-sufficient visual modality, inspiring two-stream architectures that separate appearance modeling from motion modeling.

#### B. 3D Filter Visualizations
*   **Methodology:** Optimizing input noise clips to maximize the activation scores of learned 3D CNN filters via gradient ascent (analogous to 2D deep visualization).
*   **Filter Types Revealed:**
    1.  **Static Feature Filters:** Act as spatial template matchers (edges, textures, color contrasts) that remain constant over the temporal window.
    2.  **Directional Motion Filters:** Capture temporal shifts (e.g., color transitioning from green to red, or edges rotating across time steps), showing that the model natively extracts velocity and acceleration cues.

#### C. Optimized Two-Stream Motion Field Visualization
*   **Appearance Stream vs. Temporal Stream:** Visualizations show the static spatial network focuses heavily on high-frequency object details and textures. 
*   **Flow Stream Activations:** The temporal stream highlights regions of extreme motion gradients (e.g., tracking the barbell's upward trajectory in a weightlifting clip), showing it is invariant to background context and purely sensitive to velocity fields.

---


<div id="plotly-cs231n-10-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 10 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

### 5. Visualization Blueprint (Conceptual Specification)
To debug a video classifier's temporal integration, we propose a conceptual **Interactive Spatiotemporal Receptive Field & Fusion Visualizer**:

*   **Visualization Type:** 3D Voxel/Volume Grid with slicing planes.
*   **Data Fields & Encoding:**
    *   **X, Y Axes:** Spatial dimensions of the video clip.
    *   **Z Axis:** Temporal dimension (frames $t=1 \dots T$).
    *   **Voxel Color:** Activation intensity (Red = High Positive, Blue = High Negative).
    *   **Interactive Slicing Plane:** Displays the corresponding 2D frame and its calculated dense optical flow vector field $(d_x, d_y)$ overlaid as arrows.
*   **Interactive Controls:**
    *   **Layer Depth Slider:** Move through the network layers (e.g., Conv3D_1, Pool3D_2, Conv3D_3).
    *   **Receptive Field Highlighter:** Clicking a single voxel in a deep layer highlights the corresponding 3D cone of voxels in the input space that contributed to its activation, demonstrating the exact temporal span and spatial region the network attended to.

---

### 6. Empirical Design Heuristics & Benchmark Results

#### A. Empirical Comparison of Fusion Techniques
On the Sports 1 Million dataset, Justin Johnson outlines the benchmark top-5 accuracies for different baseline models:
1.  **Single Frame 2D CNN Baseline:** $77.7\%$ top-5 accuracy. (Surprisingly strong, indicating that background context and static objects are powerful shortcuts for many video classification tasks).
2.  **Early Fusion:** Performs slightly *worse* than the single-frame baseline ($76.4\%$ top-5). (Collapsing all temporal frames in the first layer makes optimization extremely hard and overfits to specific spatial-temporal configurations).
3.  **Late Fusion:** Performs slightly *better* than the single-frame baseline ($78.5\%$ top-5).
4.  **3D CNN (Slow Fusion):** Achieves the best performance ($80.2\%$ top-5), providing a $2.5\%$ boost over late fusion by learning hierarchical spatiotemporal features.

#### B. C3D Network (VGG-16 for 3D Video)
*   **Design:** C3D uses $3 \times 3 \times 3$ 3D convolutions and $2 \times 2 \times 2$ Max Pooling layers (except the first layer which uses $1 \times 2 \times 2$ pooling to preserve temporal resolution initially).
*   **FLOPs Bottleneck:**
    *   **AlexNet:** $0.7\text{ G-FLOPs}$ per forward pass.
    *   **VGG-16 (2D):** $13.6\text{ G-FLOPs}$.
    *   **C3D (3D):** $39.5\text{ G-FLOPs}$ (nearly $3\times$ the computational footprint of VGG-16, despite having similar channel/layer proportions, due to the temporal convolution sliding cost).

#### C. Two-Stream Networks on UCF101
*   **Dataset:** UCF101 (101 action classes, 13,320 videos).
*   **Key Finding:** When evaluated on UCF101, the *Temporal (Motion) Stream* trained purely on stacked optical flow maps outperforms the *Spatial (Appearance) Stream* ($83.7\%$ vs. $73.0\%$ accuracy). Combining both streams via late prediction averaging provides a massive performance boost, reaching over $87.9\%$ accuracy, showing that appearance and motion are highly complementary.

---

### 7. Pitfalls, Debugging Tips & Reflection Questions

#### A. Gotchas & Silent Failure Modes
*   **The Dataset Rot Pitfall:** Large-scale video datasets like Sports 1 Million and Kinetics-400 were originally distributed as lists of YouTube URLs. Over time, uploaders delete, modify, or restrict access to their videos, causing the dataset to slowly "rot" (often losing over $50\%$ of original samples), leading to severe reproducibility issues.
*   **Vanishing Temporal Shift Invariance:** Early Fusion collapses temporal frames via $3T \times H \times W$ inputs in layer 1. This forces the network to learn separate spatial templates for motion transitions occurring at different absolute timestamps (e.g., a foot moving up at $t=1$ vs. $t=15$). It cannot shift kernels along the time axis, destroying temporal shift invariance and causing poor generalization on variable-speed videos.
*   **Memory Exhaustion in Recurrent Models:** Backpropagating through extremely long videos (e.g., minutes or hours) using a CNN + RNN/LSTM architecture leads to out-of-memory (OOM) errors.
    *   *Solution:* Freeze the CNN feature extractor, pre-compute and store feature tensors to disk, and backpropagate gradients strictly through the RNN layers.

#### B. Graduate-Level Reflection Questions
1.  *Prove why an Inflated 3D Network (I3D) initialized by copying 2D weights $K_t$ times must have its weights divided by $K_t$. What mathematically happens to the activations of a constant-video input ($X_1 = X_2 = \dots = X_T$) if the division by $K_t$ is omitted?*
2.  *Contrast the spatial-temporal modeling of a Non-Local Block with a standard 3D Convolution layer. Under what sequence lengths $T$ and spatial resolutions $H, W$ does the Non-Local block become computationally prohibitive compared to a stack of $3 \times 3 \times 3$ 3D convolutions? (Analyze the complexity of calculating the affinity matrix $E$).*
3.  *Why does the Temporal Stream of a Two-Stream network show a lower tendency to overfit the training dataset than the Spatial Stream, despite being trained on a relatively small video dataset like UCF101? Detail the architectural and data-driven reasons.*
