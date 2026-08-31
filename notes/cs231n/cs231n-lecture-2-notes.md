# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 2: Image Classification with Linear Classifiers

#### 0. Quick-Recall Summary
*   **The Semantic Gap:** The fundamental challenge in computer vision is the discrepancy between a human's high-level semantic understanding of an image and a computer's representation of it as a raw, grid-based 3D tensor of pixel integers $[H \times W \times C]$.
*   **Computational Inversion of $k$-NN:** Nearest Neighbor classifiers have a computational complexity of $O(1)$ for training (lazy memorization) but an unacceptable $O(N)$ for prediction (testing). Practical systems require the inverse: expensive offline training but near-instantaneous online inference.
*   **Isotropic vs. Axis-Aligned Distance:** $L1$ (Manhattan) distance is coordinate-frame dependent and behaves as an axis-aligned shape (diamond/square) under rotation, whereas $L2$ (Euclidean) distance is isotropic, rotationally invariant, and forms a circle.
*   **The Parametric Paradigm Shift:** Parametric classifiers summarize all training data knowledge into a fixed weight matrix $W$ and bias vector $b$, mapping high-dimensional pixel inputs to class scores. Once trained, the raw training data can be entirely discarded.
*   **Softmax Loss Initialization Boundary:** When weight matrices are initialized to small random values, the initial logits are approximately equal ($s_j \approx 0$), resulting in equal class probabilities ($1/C$). This establishes a critical debugging check where the initial cross-entropy loss must equal $\log(C)$.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To formalize the task of image classification through data-driven approaches, analyze the limitations of non-parametric algorithms like $k$-NN, and introduce the parametric linear classifier as the fundamental building block of deep neural networks.
*   **Lecture Category:** (a) Mathematical Foundations and (b) Core Paradigms (the transition from raw pixel distance heuristics to parameterized score optimization).
*   **Builds on:** Lecture 1, translating the biological visual processing hierarchy and historical end-to-end architectures into formal, clean mathematical score mappings and objective functions.

---

#### 2. Mathematical Foundations

##### Distance Metrics
Used to measure the similarity between two image tensors ($I_1, I_2$) in a flattened pixel space:
*   **L1 (Manhattan) Distance:**
    $$d_1(I_1, I_2) = \sum\_{p} |I_1^p - I_2^p|$$
    where $p$ indexes each individual pixel value (e.g., $32 \times 32 \times 3 = 3072$ coordinates).
*   **L2 (Euclidean) Distance:**
    $$d_2(I_1, I_2) = \sqrt{\sum\_{p} (I_1^p - I_2^p)^2}$$
    which measures the straight-line distance in Euclidean space.

##### Parametric Score Mapping
Maps raw flattened pixel vectors to class scores using a linear combination of learned parameters:
$$f(x, W, b) = W x + b$$
*   $x \in \mathbb{R}^{D \times 1}$ is the input image flattened into a column vector (for CIFAR-10, $D = 32 \times 32 \times 3 = 3072$).
*   $W \in \mathbb{R}^{C \times D}$ is the weight matrix, where each row acts as a high-dimensional template for a specific class ($C$ classes; e.g., $10$ for CIFAR-10).
*   $b \in \mathbb{R}^{C \times 1}$ is the bias vector, which scales and shifts individual class scores independently of the input features.

##### Softmax Function (Multinomial Logistic Regression)
Maps unbounded, raw class scores (logits) $s = f(x_i, W)$ to a normalized probability distribution over $C$ classes:
$$P(Y = k \mid X = x_i) = \frac{e^{s_k}}{\sum\_{j=1}^{C} e^{s_j}}$$

##### Cross-Entropy Loss
Quantifies the discrepancy (or "unhappiness") between predicted class probabilities and the target ground-truth label $y_i$ using negative log-likelihood:
$$L_i = -\log P(Y = y_i \mid X = x_i) = -\log \left( \frac{e^{s\_{y_i}}}{\sum\_{j=1}^{C} e^{s_j}} \right)$$
The total dataset loss is the average loss over all $N$ training examples:
$$L = \frac{1}{N} \sum\_{i=1}^{N} L_i$$

---

#### 3. Architecture / Algorithm Walkthrough

##### Data-Driven Classification Logic
Unlike traditional procedurally coded sorting algorithms with strict `if-then-else` logical pathways, image classification relies on a three-stage machine learning paradigm:
1.  **Collect:** Gather a labeled dataset consisting of images $X$ and their true categories $Y$.
2.  **Train:** Pass the dataset to a `train` function to construct a model correlating the visual space to the labels.
3.  **Predict:** Implement a `predict` function evaluating completely unseen test images using the learned model.

```
[Raw Pixel Input (3072x1)] ──> [Matrix Multiply (W: 10x3072)] ──> [Add Bias (b: 10x1)] ──> [Raw Logits s (10x1)]
                                                                                              │
                                                                                              ▼
[Cross-Entropy Loss (Li)] <── [Negative Log (-log)] <── [Target Class Prob] <── [Softmax Normalization]
```

##### PyTorch Blueprint (Linear Classifier and Cross-Entropy)
This minimal PyTorch implementation maps the mathematical forward pass and classification loss described in the lecture:

```python
import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, in_features: int = 3072, num_classes: int = 10):
        """
        Algebraic blueprint of a parametric linear score function f(x, W, b) = Wx + b.
        Matches CIFAR-10 dimensions: 32x32x3 flattened input and 10 output logits.
        """
        super().__init__()
        # Defines the weight matrix W (shape: [num_classes, in_features]) and bias b
        self.linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input tensor shape: [Batch_Size, 3, 32, 32]
        # Flatten spatial and channel dimensions to a single vector: [Batch_Size, 3072]
        x_flat = torch.flatten(x, start_dim=1)
        # Compute scores (logits) via linear mapping
        logits = self.linear(x_flat)
        return logits

# Illustrative training loss calculation step
if __name__ == "__main__":
    # Create mock batch: 4 images of size 32x32x3
    mock_images = torch.randn(4, 3, 32, 32)
    mock_targets = torch.randint(0, 10, (4,))  # Ground-truth class labels (shape: [4])
    
    # Initialize linear classifier
    model = LinearClassifier()
    
    # Forward pass to obtain raw, unbounded scores (logits)
    scores = model(mock_images)
    
    # Softmax regression + Negative Log Likelihood is computed natively in PyTorch's CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(scores, mock_targets)
    
    print(f"Computed Logits Shape: {scores.shape}") # Should be [4, 10]
    print(f"Average Batch Cross-Entropy Loss: {loss.item():.4f}")
```

---

#### 4. Visual Intuition & Interpretability

##### The Three Viewpoints of the Linear Classifier
1.  **Algebraic Viewpoint:** Each class score is calculated independently as the inner dot product of a specific row in weight matrix $W$ with the flattened input vector $x$, offset by the corresponding class bias in $b$.
2.  **Visual Viewpoint (Templates):** By unflattening each row of the learned weight matrix $W$ back into the original image dimensions $[H \times W \times C]$, we can visualize what the classifier is searching for. For example:
    *   *The "Car" template* resembles a generic red, front-facing automobile template.
    *   *The "Horse" template* is often learned as a green-blob background with a vague brown shape in the middle, revealing a visual bias where the classifier associates the background "grass" directly with the horse class.
3.  **Geometric Viewpoint (Decision Boundaries):** High-dimensional pixel space is partitioned by a set of linear hyperplanes. The decision boundary for any class represents the hyperplane where the class score equals zero:
    $$W_k x + b_k = 0$$
    The classifier can only separate classes if they are linearly separable in the input space.

##### Visual Failure Modes of Raw Distance Heuristics
*   **The Semantic Gap & Pixel-wise Shifts:** A $k$-NN operating in pixel space fails to recognize semantic identity because its distance calculations are easily fooled. For example, shifting a target image by just one pixel to the right creates a massive pixel-wise L1/L2 distance from the original, even though the semantic content remains identical to a human observer.
*   **Occlusion & Background Domination:** If a target object (e.g., a green frog) is placed on a brown background, pixel-wise L1/L2 distances will match the background colors rather than the semantic category, causing the model to misclassify a green frog on grass as a dog.

---


<div id="plotly-cs231n-2-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 2 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)

To help students build intuition around the structural difference between distance metrics, we propose an interactive **Isotropic vs. Axis-Aligned Distance Visualizer**:

*   **Visualization Type:** 2D Coordinate Grid with Contour Contour plots.
*   **Data Fields & Encoding:**
    *   **X and Y axes:** Represent two continuous feature values ($x_1, x_2$) extracted from an image (e.g., intensity of pixel A vs. pixel B).
    *   **Distance Contours:** Lines of equal distance from the origin ($0,0$).
    *   **Color-mapping:** Gradient heat-map showing distance magnitude (Brighter/Redder = larger distance; Darker/Bluer = closer to origin).
*   **Interactive Controls:**
    *   **Metric Toggle (L1 vs. L2):** Toggles between Manhattan and Euclidean metrics. When set to $L1$, the contour lines form an axis-aligned diamond/square. When set to $L2$, the contours form concentric circles.
    *   **Coordinate Rotation Slider ($0^\circ$ to $90^\circ$):** Rotates the coordinate axes. Sliding the rotation shows that the circular $L2$ contours remain perfectly unchanged (isotropic), while the diamond $L1$ contours distort relative to the new feature axes, demonstrating why $L1$ is coordinate-frame dependent.

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **The Baseline of Random Guessing:** On the CIFAR-10 dataset (which has 10 balanced categories), a completely random classifier achieves **10.0% accuracy**.
*   **Pixel-space nearest-neighbor performance:** Applying $1$-NN or $k$-NN directly to raw CIFAR-10 pixel tensors yields a peak accuracy of only **~28.0% to 29.0%** (optimized at $K = 7$ using 5-fold cross-validation).
*   **Modern Supervised Limits:** While raw pixel-wise distance methods stall below 30%, modern deep representation learning has essentially solved the CIFAR-10 task, pushing peak accuracy up to **99.7%**.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Silent Failures
*   **The Non-Linear Separability Trap:** Linear classifiers fail silently on highly non-linear geometric configurations of data in pixel space, such as parity (XOR) patterns, concentric circles (ring distribution), or multimodal clusters of the same class.
*   **Equal Probability Logit Leak:** If a softmax model's loss at the very first step of training does not equal $-\log(1/C)$ (e.g., $\approx 2.3$ for 10 classes), it indicates a bug in weight initialization or loss computation.

##### Graduate-Level Reflection Questions
1.  **Computational Inversion:** Explain why nearest neighbor classifiers are computationally impractical for edge-device computer vision deployments, and explain how the parametric linear score function $f(x,W,b)$ mathematically inverts this trade-off.
2.  **Coordinate-Frame Sensitivity:** Under what physical assumptions about the underlying feature representations would you choose an $L1$ distance metric over an $L2$ metric, and how does this relate to their geometric contours under coordinate-frame rotation?
3.  **Template Singularity:** If a category contains massive intra-class appearance variation (e.g., a green car, a red car, and a blue car), how does a single-layer linear classifier's visual viewpoint template represent this, and why does this constrain its capacity compared to a multi-layer neural network?
