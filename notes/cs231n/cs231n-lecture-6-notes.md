# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 6: CNN Architectures

#### 0. Quick-Recall Summary
*   **The Depth Barrier Resolved:** Plain deep CNNs suffer from optimization degradation where stacked layers fail to learn even identity mappings. ResNets bypass this via residual skip connections ($H(x) = F(x) + x$), enabling successful optimization of 100+ layer architectures.
*   **Receptive Field & Parameter Efficiency:** Stacking three $3\times3$ convolutions with stride 1 has the identical effective receptive field as a single $7\times7$ convolution ($7\times7$), but uses $\approx 45\%$ fewer parameters and introduces three non-linearities instead of one.
*   **Normalizations Demystified:** Normalization techniques scale and shift activations after transforming them to unit Gaussians. **Layer Norm** computes statistics across channels, height, and width for each sample independently ($C \times H \times W$), whereas **Batch Norm** computes statistics across batch and spatial dimensions per channel ($N \times H \times W$).
*   **Kaiming Initialization:** To prevent vanishing or exploding activations in deep ReLU networks, weights must be initialized with a zero-mean Gaussian with variance $\sigma^2 = \frac{2}{D\_{in}}$, where $D\_{in}$ is the layer input dimensionality.
*   **In-Context Transfer Learning:** When target datasets are small, pre-trained ImageNet CNNs act as robust frozen feature extractors where only a newly initialized linear head is trained ("Linear Probing"). With more data, the entire network is end-to-end "fine-tuned" with a lower learning rate.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master the structural design of Convolutional Neural Networks, trace the evolutionary milestones from AlexNet to ResNet, and acquire the mathematical, initialization, and regularization mechanisms required to successfully train deep vision models.
*   **Lecture Category:** (b) Architecture Design and (c) Training/Optimization Practice (a balanced blend of spatial structural design and practical optimization heuristics).
*   **Builds on:** Lecture 5, extending localized 2D convolutional and pooling primitives into deep, interconnected global feature-extraction networks.

---

#### 2. Mathematical Foundations

##### Receptive Field Scaling Arithmetic
For a 1D or 2D grid with kernel size $K$, stride $S=1$, and padding $P$, stacking $L$ layers increases the effective receptive field ($RF$) of an activation map in layer $l$ relative to the input layer $l-1$ recursively:
$$RF_l = RF\_{l-1} + (K - 1)$$
With $RF_0 = 1$ (input pixel), the receptive field after $L$ layers is:
$$RF_L = 1 + L(K - 1)$$

##### Parameter Footprint: Stacked Convolutions vs. Single Large Kernel
Let $C$ be the number of input and output channels (assumed constant). 
*   **Stacked Convolutions ($L$ layers of $K \times K$):**
    $$\text{Params}\_{stack} = L \times (K \times K \times C \times C) = L \cdot K^2 \cdot C^2$$
*   **Equivalent Single Large Convolution ($1$ layer of $K\_{eff} \times K\_{effective}$ where $K\_{eff} = 1 + L(K-1)$):**
    $$\text{Params}\_{single} = K\_{eff}^2 \times C \times C = (1 + L(K-1))^2 \cdot C^2$$
*   *VGG Case Comparison ($L=3, K=3$):*
    $$\text{Params}\_{stack} = 3 \cdot (3^2) \cdot C^2 = 27 C^2$$
    $$\text{Params}\_{single} = (1 + 3(3-1))^2 \cdot C^2 = 7^2 \cdot C^2 = 49 C^2$$
    $$\text{Parameter Savings Ratio} = 1 - \frac{27 C^2}{49 C^2} \approx 44.9\%$$

##### Normalization Statistics
Let the input batch tensor be $X \in \mathbb{R}^{N \times C \times H \times W}$. Normalization transforms each activation $X\_{n,c,h,w}$ to $\hat{X}\_{n,c,h,w}$ and applies a learnable scale $\gamma$ and shift $\beta$:
$$\hat{X}\_{n,c,h,w} = \frac{X\_{n,c,h,w} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$Y\_{n,c,h,w} = \gamma \hat{X}\_{n,c,h,w} + \beta$$

The statistics $\mu$ and $\sigma^2$ are computed over different index subsets $\mathcal{S}$:
*   **Layer Normalization (LN):** Normalizes across channels and spatial dimensions for each batch sample independently.
    $$\mu\_{LN}(n) = \frac{1}{C \cdot H \cdot W} \sum\_{c=1}^C \sum\_{h=1}^H \sum\_{w=1}^W X\_{n,c,h,w}$$
    $$\sigma^2\_{LN}(n) = \frac{1}{C \cdot H \cdot W} \sum\_{c=1}^C \sum\_{h=1}^H \sum\_{w=1}^W (X\_{n,c,h,w} - \mu\_{LN}(n))^2$$
*   **Batch Normalization (BN):** Normalizes across the batch and spatial dimensions for each channel independently.
    $$\mu\_{BN}(c) = \frac{1}{N \cdot H \cdot W} \sum\_{n=1}^N \sum\_{h=1}^H \sum\_{w=1}^W X\_{n,c,h,w}$$
    $$\sigma^2\_{BN}(c) = \frac{1}{N \cdot H \cdot W} \sum\_{n=1}^N \sum\_{h=1}^H \sum\_{w=1}^W (X\_{n,c,h,w} - \mu\_{BN}(c))^2$$

##### Kaiming (He) Initialization
To preserve the variance of activations in deep architectures using ReLU activations, weights are initialized from a zero-mean Gaussian distribution with standard deviation $\sigma$:
$$W \sim \mathcal{N}\left(0, \sigma^2\right) \quad \text{where} \quad \sigma = \sqrt{\frac{2}{D\_{in}}}$$
*   For a Fully Connected layer: $D\_{in} = \text{fan\_in}$ (number of input nodes).
*   For a Convolutional layer: $D\_{in} = K_h \times K_w \times C\_{in}$ (kernel height $\times$ kernel width $\times$ input channel depth).

##### Dropout Mathematical Expectation
Let $M \in \{0, 1\}^{D}$ be a random mask vector where each element $M_i \sim \text{Bernoulli}(1-p)$, and $p$ is the dropout probability (the probability of setting an activation to zero).
*   **Training Time Forward Pass:**
    $$y\_{\text{train}} = x \odot M$$
*   **Test Time Activation Scaling:** Since $50\%$ (or $p\%$) of the neurons are dropped during training, the expected magnitude of activations at test time is preserved by scaling the inputs by the keep probability $1-p$ (or multiplying by $p$ if $p$ is the keep probability):
    $$y\_{\text{test}} = (1 - p) \cdot x$$
*   **Inverted Dropout (Modern Alternative):** PyTorch implements inverted dropout by scaling activations during *training* to avoid any scaling overhead at test time:
    $$y\_{\text{train}} = \frac{x \odot M}{1 - p}, \quad y\_{\text{test}} = x$$

---

#### 3. Architecture / Algorithm Walkthrough

##### VGG vs. ResNet Data Flow
The structural shift between plain feedforward stacking (VGG) and identity-routed shortcut skip propagation (ResNet) is illustrated below:

```
VGG Block (Plain Feedforward)               ResNet Block (Residual Identity)
         [Input X]                                     [Input X]
             │                                         │       ╲
             ▼                                         │        ╲ (Skip Connection)
      [3x3 Conv, Pad 1]                                │         │
             │                                         ▼         │
             ▼                                 [3x3 Conv, Pad 1] │
          [ReLU]                                       │         │
             │                                         ▼         │
             ▼                                      [ReLU]       │
      [3x3 Conv, Pad 1]                                │         │
             │                                         ▼         │
             ▼                                 [3x3 Conv, Pad 1] │
          [ReLU]                                       │         │
             │                                         ▼         │
             ▼                                      [Add] ◄──────┘
         [Output Y]                                    │
                                                       ▼
                                                    [ReLU]
                                                       │
                                                       ▼
                                                   [Output Y]
```

##### PyTorch Blueprint (VGG & ResNet Structural Primitives)
This self-contained script provides robust, modular blueprints for VGG blocks and residual blocks, incorporating Kaiming initialization and Layer/Batch Normalization layers:

```python
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    """
    A standard VGG-style plain block: Stack of 3x3 convolutions with constant channels,
    stride=1, padding=1 (preserves HxW), ending with MaxPool2d downsampling.
    """
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            # Conv2D preserves spatial size because K=3, P=1, S=1
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, 
                                    out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels)) # Normalizes N x H x W per channel
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # Halves H and W
        self.block = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        # Kaiming (He) normal initialization for stable ReLU gradients
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.block(x)


class ResNetBlock(nn.Module):
    """
    A standard ResNet residual block with a projection shortcut when spatial dimensions 
    change or channel sizes double.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Match spatial size via strided 1x1 conv and match channels via out_channels filters
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        for m in self.shortcut:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        identity = self.shortcut(x) # Identity skip connection path
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity # Element-wise sum of activations (tensor shapes must match)
        out = self.relu(out)
        return out
```

---

#### 4. Visual Intuition & Interpretability

##### The Plain Network Optimization Degradation
*   **The Observed Phenomenon:** Stacking plain convolutional layers results in a higher training error for deeper networks (e.g. 56 layers vs 20 layers). This is *not* overfitting (as both train and test errors increase) but rather an optimization failure where deep stacks struggle to preserve identity transformations during backpropagation.
*   **The ResNet Solution:** By modifying the architecture to learn $F(x) = H(x) - x$, the network learns a "delta" or a residual modification. If the optimal state is an identity function, the optimizer can easily decay the convolution weights to zero, automatically bypassing the block.

##### Weight Initialization Landscapes
*   **Under-initialized ($\sigma = 0.01$):** Activations decay exponentially toward zero as they pass deeper into the network, resulting in zero-magnitude gradients and leaving early layers completely untrained.
*   **Over-initialized ($\sigma = 0.05$):** Activations explode exponentially toward infinity, saturating ReLU bounds and causing numerical overflow (NaNs) or severe training instability.
*   **Kaiming (He) Balance:** Ensures variance is preserved across arbitrary depths, presenting a uniform, balanced activation magnitude at every layer.

---


<div id="plotly-cs231n-6-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 6 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)
To visually isolate the effects of normalizations and initialization on gradient flow, we propose an interactive **Activation Flow & Norm Histogram Visualizer**:
*   **Visualization Type:** Dynamic layer-by-layer distribution plots (histograms/boxplots) updated at each forward/backward pass.
*   **Data Fields & Encoding:**
    *   **X-axis:** Layer Index (1 to $L$).
    *   **Y-axis:** Value range of activation or gradient elements.
    *   **Color-encoding:** Normalization Type (Blue = No Norm, Orange = Batch Norm, Green = Layer Norm).
    *   **Histogram shape:** Represents activation density per layer.
*   **Interactive Controls:**
    *   **Initialization Standard Deviation Slider ($0.001 \rightarrow 0.5$):** Adjusts weight initialization scale. Watch the "No Norm" (Blue) activations either vanish into a vertical line at 0 or widen infinitely, while "Batch Norm" (Orange) maintains steady unit distributions.
    *   **Normalization Toggle:** Switches between Layer Norm, Batch Norm, and Instance Norm, highlighting the dimensions across which statistics are gathered.

---

#### 6. Empirical Design Heuristics & Benchmark Results

##### Pre-processing Statistics (ImageNet Defaults)
All input images must be normalized before model ingestion. The standard means and standard deviations computed over the ImageNet training set are:
$$\mu\_{\text{ImageNet}} = [0.485, 0.456, 0.406], \quad \sigma\_{\text{ImageNet}} = [0.229, 0.224, 0.225]$$
The normalized input pixel value is:
$$x\_{\text{norm}} = \frac{x\_{\text{raw}} - \mu}{\sigma}$$

##### Test-Time Augmentation (TTA)
For maximum competitive validation accuracy (such as winning Kaggle competitions or achieving top benchmark status), models use Test-Time Augmentation:
1.  Generate $M$ variations (different scales, crop positions, horizontal flips) of a single test image.
2.  Perform forward passes to obtain prediction logit vectors for all $M$ variations.
3.  Compute the average prediction: $\bar{y} = \frac{1}{M} \sum\_{m=1}^M y_m$.
*This consistently yields a $1\%\text{--}2\%$ reduction in absolute error rates.*

##### Transfer Learning Decision Matrix
The choice of transfer learning strategy is governed strictly by target dataset volume and domain similarity relative to the pre-training set (ImageNet):

| Dataset Size | Domain Similarity | Transfer Strategy |
| :--- | :--- | :--- |
| **Very Small** | High (e.g. daily objects) | **Linear Probe:** Freeze the feature extractor backbone, replace the final classification layer, and train only the new linear weights. |
| **Medium-Large** | High (e.g. daily objects) | **Full Fine-Tuning:** Initialize with pre-trained weights, replace the final layer, and train the entire network using a very small learning rate (e.g. $0.1 \times \text{lr}\_{initial}$). |
| **Very Small** | Low (e.g. Mars rover) | **Backbone Feature Extractor:** Freeze early layers, but try training a linear probe on top of intermediate layers (which contain more generic visual primitives like Gabor edges) instead of late semantic layers. |
| **Medium-Large** | Low (e.g. Mars rover) | **Full/Partial Fine-Tuning:** Fine-tune deep blocks while freezing early visual layers, or train from scratch if compute constraints allow. |

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Training Pitfalls
*   **The Inverted Dropout Scaling Mismatch:** If dropout is implemented without scaling activations by $\frac{1}{1-p}$ (or $1-p$ depending on the representation) during training, the magnitude of inputs to the next layer at test time will be artificially high, leading to catastrophic prediction errors.
*   **Dead ReLU Collapse:** Poor initialization or extremely high learning rates can cause weights to update such that a ReLU unit never activates on any training sample. This "dead neuron" has a permanent local gradient of 0, meaning its weights will never update again.
*   **Pre-training Domain Discrepancy:** Reusing features pre-trained on everyday natural objects (like dogs and cars) on a completely different domain (like medical radiology or astronomical imagery) can cause severe feature representation bottlenecks, as the backbone has no pre-existing templates for those specialized textures.

##### Graduate-Level Reflection Questions
1.  **Symmetry and Optimization in initialization:** Why does initializing a multi-layer plain CNN with identical weights (e.g. all ones or constant values) collapse representational learning? What is the mathematical connection between random initialization, symmetry breaking, and SGD parameter divergence?
2.  **Representational Capacity of ResNet:** A ResNet block computes $H(x) = F(x) + x$. Prove mathematically why a 56-layer ResNet can theoretically represent any function that a shallower 20-layer ResNet can model, and explain why this architecture resolves the optimization "degradation" trap that plain stacked architectures fall into.
3.  **Covariate Shift in Layer Norm vs. Batch Norm:** For recurrent sequence models or variable-length inputs, Batch Normalization is highly unstable because batch statistics are heavily skewed by varying lengths. Explain how Layer Normalization circumvents this sequence length dependency, and trace why LN is mathematically superior for dynamic input structures.
