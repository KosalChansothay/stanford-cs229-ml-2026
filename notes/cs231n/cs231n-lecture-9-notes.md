# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 9: Object Detection, Image Segmentation, Visualizing

#### 0. Quick-Recall Summary
*   **Semantic Segmentation Paradigms:** Transitioned from slow patch-based classification to Fully Convolutional Networks (FCNs) that process entire images at once. U-Net preserves fine spatial details in the decoder by using direct skip connections to copy high-resolution feature maps from the encoder.
*   **Upsampling Mechanics:** Unpooling can be non-parametric (Nearest Neighbor, Bed of Nails) or coordinate-cached (Max Unpooling, which reuses indices saved during Max Pooling). Parametric upsampling is performed using **Transposed Convolution**, which reverses the spatial downsampling of standard convolutions.
*   **Object Detection Scaling:** Multitask loss extends classification to single-object localization. Multiple objects require region proposals (R-CNN's slow patch cropping) or single-stage regression (YOLO's grid-based $S \times S$ spatial collapse).
*   **Transformer-Based Detection:** DETR (DEtection TRansformer) models object detection as a direct set prediction problem. It uses a CNN backbone with a Transformer Encoder-Decoder to process a fixed number of trainable "object queries," optimizing a bipartite matching loss.
*   **Interpretability Foundations:** Saliency maps visualize gradients of class scores with respect to input pixels. Class Activation Mapping (CAM) is limited to networks with global average pooling; **Grad-CAM** generalizes this by using gradients as weights to blend the penultimate layer's feature maps.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master the spatial and semantic scaling of computer vision models beyond simple image-level classification. This lecture details how models classify every pixel (segmentation), localize multiple instances with bounding boxes (detection), and how researchers mathematically trace decisions back to visual evidence (interpretability).
*   **Lecture Category:** (d) Specific Vision Task (Object Detection & Image Segmentation) and (c) Visual Interpretability.
*   **Builds on:** Lecture 8 (Attention and Transformers) and Lectures 5 & 6 (CNN Architectures).

---

#### 2. Mathematical Foundations

##### A. Pixel-Wise Cross-Entropy Loss (Semantic Segmentation)
For an image with height $H$ and width $W$, semantic segmentation is trained by treating each pixel as an independent classification problem:
$$L\_{\text{seg}} = -\frac{1}{H \cdot W} \sum\_{h=1}^{H} \sum\_{w=1}^{W} \log\left( \frac{e^{s\_{h, w, y\_{h,w}}}}{\sum\_{c=1}^{C} e^{s\_{h, w, c}}} \right)$$
where $s\_{h, w, c}$ is the unnormalized logit predicted for pixel $(h, w)$ for class $c$, and $y\_{h,w} \in \{1, \dots, C\}$ is the ground-truth class label at that pixel coordinate.

##### B. Multitask Joint Localization & Classification Loss
Single-object localization uses a dual-head loss combining softmax classification and regression of bounding box offsets $(x, y, w, h)$:
$$L\_{\text{multitask}} = L\_{\text{softmax}}(p, y) + \lambda \cdot \mathbb{I}(y \geq 1) \cdot \sum\_{j \in \{x, y, w, h\}} \|t_j - t^*_j\|_2^2$$
where $p$ is the predicted class probability distribution, $y$ is the ground-truth class label ($y \ge 1$ represents foreground objects, filtering out background), $t = (t_x, t_y, t_w, t_h)$ is the predicted box coordinate offset vector, $t^*$ represents ground-truth coordinates, and $\lambda$ is a balancing hyperparameter.

##### C. Transposed Convolution Arithmetic
Transposed convolution (sometimes called fractionally strided convolution) is the mathematical transpose of a standard convolution matrix operator. For an input with stride $S$, kernel size $K$, and padding $P$, the output dimension is upsampled as follows:
$$O = S \cdot (I - 1) + K - 2P$$
Where standard convolution maps $1 \times K$ patches to a single scalar, transposed convolution multiplies a single input scalar by a $K \times K$ filter, writing the scaled filter to the output grid and summing overlapping regions.

##### D. Bipartite Matching Loss (Hungarian Matching in DETR)
DETR avoids non-maximum suppression (NMS) by predicting a fixed-size set of $N$ predictions and computing a bipartite matching via the Hungarian algorithm to find a permutation of $N$ elements $\sigma \in \mathfrak{S}_N$ that minimizes matching cost:
$$\hat{\sigma} = \arg\min\_{\sigma \in \mathfrak{S}_N} \sum\_{i=1}^{N} \mathcal{L}\_{\text{match}}(y_i, \hat{y}\_{\sigma(i)})$$
where the matching cost combines class probability and bounding box spatial alignment (such as L1 and generalized IoU losses).

##### E. Pixel Saliency Maps via Backpropagation
To isolate which pixels are responsible for a specific class score $S_c(I)$ for image $I$, we compute the gradient of the unnormalized score with respect to the input image pixels:
$$\text{Saliency}(I) = \max\_{c \in \{\text{channels}\}} \left| \frac{\partial S_c(I)}{\partial I\_{x, y}} \right|$$
This represents the first-order Taylor expansion approximation of the image pixels' influence on the model's confidence.

##### F. Grad-CAM Formulation (Penultimate Feature Blending)
Grad-CAM computes a spatial heat map by weighting penultimate convolutional feature maps $A^k \in \mathbb{R}^{H \times W}$:
1.  **Gradient Weight Computation:** Global average pooling of the gradients of the class score $S_c$ with respect to the feature map $A^k$:
    $$\alpha_c^k = \frac{1}{Z} \sum\_{i=1}^{H} \sum\_{j=1}^{W} \frac{\partial S_c}{\partial A\_{i, j}^k}$$
    where $Z = H \times W$ is the spatial area.
2.  **Weighted Combination and ReLU:**
    $$L\_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum\_{k} \alpha_c^k A^k \right)$$
    The $\text{ReLU}$ is strictly applied to retain only features that positively correlate with the class of interest, ignoring features that contribute to other categories.

---

#### 3. Architecture / Algorithm Walkthrough

##### A. Upsampling & Segmentation Topologies
The lecture outlines the historical progression of upsampling paths in encoder-decoder networks:

```
[Nearest Neighbor Unpooling]  [Bed of Nails]          [Max Unpooling]            [Transposed Convolution]
   1 2  ──>  1 1 2 2            1 2  ──>  1 0 2 0        1 2  ──>  0 0 2 0          Input x Filter,
   3 4       1 1 2 2            3 4       0 0 0 0        3 4       0 3 0 0          sum overlapping
             3 3 4 4                      3 0 4 0                  (uses indices)   regions
```

##### B. PyTorch Blueprint: Modular Segmentation and Grad-CAM Hooks
This blueprint provides:
1.  A standard segmentation layer mapping encoding-decoding unpooling loops with coordinate indices tracking.
2.  A functional Grad-CAM hook implementation to capture intermediate feature maps and their corresponding backward gradients on the fly.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateTrackingDecoder(nn.Module):
    """
    Illustrative implementation of a coordinate-cached upsampling network (MaxUnpool2D).
    It reuses the spatial coordinates (indices) saved during Max Pooling to preserve
    boundary sharpness in decoders, a technique pioneered by SegNet/U-Net lineages.
    """
    def __init__(self):
        super(CoordinateTrackingDecoder, self).__init__()
        # Encoder Module
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # Cache indices
        
        # Decoder Module
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 10, kernel_size=3, padding=1) # 10-class segmentation output

    def forward(self, x):
        # Forward Pass (Encoder)
        h1 = F.relu(self.conv1(x))
        pooled_h1, pool_indices = self.pool1(h1) # Caches exact spatial coordinates
        
        # Forward Pass (Decoder)
        unpooled = self.unpool1(pooled_h1, pool_indices) # Restores coordinates
        out_logits = self.conv2(unpooled)
        return out_logits


class GradCAMWrapper:
    """
    Functional wrapper implementing Grad-CAM. It registers PyTorch forward and backward 
    hooks to automatically extract penultimate activation tensors and backpropagated gradients.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook registrations
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach() # Penultimate feature map A^k

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach() # Penultimate gradients dS/dA^k

    def generate_heatmap(self, class_idx, logits):
        """
        Computes the Grad-CAM localization map using cached activations and gradients.
        """
        if self.activations is None or self.gradients is None:
            raise ValueError("Must execute forward and backward passes before generating Grad-CAM.")
        
        # Global Average Pool gradients to calculate filter weights alpha_c^k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # [Batch, Channels, 1, 1]
        
        # Compute weighted sum of activations across channels
        cam = torch.sum(weights * self.activations, dim=1) # [Batch, Height, Width]
        
        # Apply ReLU to retain only positive contributions to the class score
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def release_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
```

---

#### 4. Visual Intuition & Interpretability

##### A. Saliency Maps vs. Grad-CAM Interpretability
The lecture contrasts classical interpretability techniques with spatial activation maps:
*   **Saliency Maps:** Often produce noisy, fine-grained pixel-level attributions. While they demonstrate which pixels are sensitive to score wiggles, they lack class specificity and often highlight generic edge boundaries rather than holistic semantic parts.
*   **Grad-CAM:** Generates coarse, localized heat maps that cleanly isolate the semantic coordinates of distinct classes. For example, in a medical setting classifying "tumor" vs. "normal," Grad-CAM directly localizes the boundaries of the pathology within the raw tissue scan.
*   **ViT Attention Visualization:** Since Transformers rely on self-attention, visualizing the raw attention matrices provides direct interpretability out of the box, mapping exactly which image patches attend to the `[CLS]` token.

##### B. Spatial Coordinate Preservation (SegNet/U-Net vs. FCN)
*   **Fully Convolutional Networks (FCNs):** Fall victim to boundary blurring. Because they compress the spatial dimension to a bottleneck and then aggressively upsample via bilinear interpolation, the model loses sharp contours (e.g., boundaries of a cat vs. background).
*   **Max Unpooling / U-Net Skip Routing:** Resolves boundary fade. By utilizing max-unpooling coordinate caches (recording exact max index grids) or copy-pasting high-resolution encoder layers directly to the decoder, boundaries remain mathematically crisp and spatially accurate.

##### C. Bounding Box Anchoring and Bipartite Collapse
*   **Anchor/Grid-Based Detectors (YOLO):** Fall victim to overlapping duplicate detections. Because the grid cell regression is decoupled, multiple nearby cells often fire on the same object, necessitating non-maximum suppression (NMS) to prune overlapping boxes.
*   **DETR Set Matching:** Completely avoids NMS by forcing a one-to-one bipartite matching between the $N$ object queries and ground-truth boxes.

---


<div id="plotly-cs231n-9-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 9 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)

To analyze the spatial upsampling path of segmentation models, we propose a **Interactive Max Unpooling & Feature Reconstruction Heatmap Visualizer**:

*   **Visualization Type:** Two-panel linked grid animation.
*   **Data Fields & Encoding:**
    *   **Left Panel (Encoder Grid):** Represents the $H \times W$ feature map. Pixels are colored by activation intensity (linear colormap: cold blue to hot red). A white highlight indicates the coordinate selected during Max Pooling.
    *   **Right Panel (Decoder Grid):** Represents the upsampled $2H \times 2W$ feature map.
*   **Interactive Controls:**
    *   **Hyperparameter Toggle (Nearest Neighbor vs. Max Unpooling):** 
        *   When set to *Nearest Neighbor*, the visualizer demonstrates the $2 \times 2$ duplication of every source activation.
        *   When set to *Max Unpooling*, the visualizer demonstrates how only the exact coordinate index receives the source activation value, while the other 3 neighboring slots are set to zero.
    *   **Receptive Field Slider:** Drags a virtual kernel across the decoder grid, demonstrating the spatial smoothing and "boundary sharpening" effect as successive 2D transposed convolutions are applied on top of the unpooled sparse tensor.

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **Symmetry and Hyperparameters in Transformers:** When using Vision Transformers (ViTs) for classification, omitting the special `[CLS]` token and replacing it with Global Average Pooling (GAP) across all patch tokens is empirically shown to yield comparable accuracy while stabilizing optimization.
*   **Hungarian Matching Cardinality:** DETR is typically configured with exactly $N = 100$ object queries, meaning it seeks a maximum of 100 object instances per image.
*   **FCN Segmentation Loss Scaling:** Summing pixel-wise cross-entropy losses across high-resolution inputs (e.g., $1024 \times 1024$) can explode gradient magnitudes. Normalizing the loss by $H \cdot W$ is a critical training heuristic.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas
*   **The Saliency Magnitude Trap:** Saliency maps are highly sensitive to high-frequency noise. A model can exhibit perfect pixel saliency on edges that actually have zero semantic relevance to the classification, leading researchers to over-interpret model "understanding".
*   **Grad-CAM Penultimate Selection:** Grad-CAM assumes the target penultimate convolutional layer contains rich spatial representations. If applied to a layer that is too early in the network, the gradients will highlight generic texture boundaries instead of class-specific objects.
*   **Loss Explosion in Transposed Convolutions:** Overlapping regions in transposed convolutions accumulate values via addition. If the stride is smaller than the kernel size, unnormalized additions will cause checkboard artifacts and gradient explosions.

##### Graduate-Level Reflection Questions
1.  **Fully Convolutional vs. Coordinate-Cached Decoders:** Prove why a standard FCN decoder relying solely on transposed convolutions struggles to reconstruct fine-grained structures like thin lines or sharp corners, and explain how Max Unpooling bypasses this capacity bottleneck without adding learnable parameters.
2.  **DETR Object Query Orthogonality:** DETR maps learnable, constant "object queries" to target predictions. How does the model prevent multiple queries from collapsing onto the same prominent foreground object during backpropagation, and what role does self-attention in the decoder play in enforcing this spatial separation?
3.  **The ReLU Constraint in Grad-CAM:** Explain why Grad-CAM applies a ReLU activation to the final weighted sum of feature maps. Mathematically, what class of features is discarded by the ReLU, and why is this filtering step crucial for class-specific localization?
