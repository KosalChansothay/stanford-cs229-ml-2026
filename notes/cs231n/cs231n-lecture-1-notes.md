# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 1: Introduction

#### 0. Quick-Recall Summary
*   **Cornerstone of Intelligence:** Visual intelligence is the evolutionary catalyst for biological and artificial intelligence, originating from the "Cambrian Explosion" 540 million years ago with the onset of photosensitive cells.
*   **The Three Converging Forces:** The 2012 deep learning rebirth (AlexNet) was unlocked by the convergence of: (1) high-capacity algorithms (Backpropagation/CNNs), (2) GPU compute (parallel processors), and (3) large-scale annotated data (ImageNet).
*   **Hierarchical Processing:** Biological vision (Hubel & Wiesel, 1959) and CNNs are unified by hierarchical representation: early stages detect simple edges/lines, while deeper layers compose them into complex, abstract semantic objects.
*   **Ill-Posed Reconstruction:** Recovering a 3D physical world from 2D retinal projections is mathematically ill-posed; biological vision resolves this through triangulation (stereopsis) and contextual priors.
*   **Human Cognitive Speed:** Humans categorize complex natural scenes in under 150 milliseconds (measured via EEG), relying on dedicated cortical pathways (e.g., fusiform face area, place area) optimized over evolutionary timescales.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To trace the historical, biological, and technical evolution of computer vision and define its role as a fundamental pillar of artificial general intelligence. It frames the transition from hand-engineered visual feature rules to high-capacity, data-driven deep neural networks trained end-to-end.
*   **Lecture Category:** (a) Historical Foundations and (b) Core Paradigms (the transition from manual feature extraction to end-to-end representation learning).
*   **Builds on:** Introduces the entire quarter's curriculum, highlighting the mathematical, algorithmic, and computational landscape of deep learning for visual recognition.

---

#### 2. Mathematical Foundations
*   **Linear/Parametric Mapping (Introductory Formalism):**
    While Lecture 1 is a high-level conceptual introduction, it introduces the core parametric mapping of computer vision:
    $$f(x, W, b) = W x + b$$
    where $x$ represents the raw pixel tensor flattened into a vector, $W$ represents the weight matrix (parameters), and $b$ is the bias vector.
*   **Top-1 vs. Top-5 Error Rate:**
    The historical benchmark metric used in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC):
    $$\text{Top-5 Error Rate} = \frac{1}{N} \sum\_{i=1}^{N} \mathbb{I}(y_i \notin \hat{Y}\_{i, 1..5})$$
    where $y_i$ is the ground-truth label, $\hat{Y}\_{i, 1..5}$ represents the five most confident class predictions from the model, and $\mathbb{I}(\cdot)$ is the indicator function.
*   **The Calculus of Backpropagation (Conceptual):**
    First introduced mathematically in 1986 (Rumelhart, Hinton, Williams), backpropagation utilizes the derivative chain rule to compute parameter updates based on an error-correcting objective function:
    $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}$$

---

#### 3. Architecture / Algorithm Walkthrough
The lecture traces the architectural lineages of early seeing systems:

```text
[Hubel & Wiesel (1959)] ──> [Fukushima's Neocognitron (1980)] ──> [LeCun's LeNet (1998)] ──> [AlexNet (2012)]
      (Biological)                     (Hand-designed CNN)               (Backprop Trained)         (GPU Accelerated)
```

##### Comparison of Classical Paradigms
1.  **Fukushima's Neocognitron (1980):** A multi-layer architecture incorporating convolution and pooling-like operations inspired by simple and complex cells. However, it had no automatic learning rule; all parameters were meticulously hand-designed.
2.  **Yann LeCun's LeNet (1990s):** Incorporated a backpropagation learning rule to optimize 2D convolutional and pooling parameters. Highly successful for digit classification (US Postal service zip-code readers) but failed to scale to complex natural images due to limited data and compute.
3.  **AlexNet (2012):** Expanded LeNet's 2D convolutional architecture to 8 layers, parallelized across 2 Nvidia GPUs, and trained on the 1.2-million-image ILSVRC subset.

##### PyTorch Blueprint (LeNet/AlexNet Architectural Ancestry)
This minimal skeleton represents the transition from Fukushima's hand-designed visual pathways to Yann LeCun's end-to-end trained CNN, laying the baseline for AlexNet:

```python
import torch
import torch.nn as nn

class AncestralCNN(nn.Module):
    """
    Illustrative blueprint modeling the classical 2D Convolution-Pooling hierarchy 
    introduced by LeNet and scaled by AlexNet.
    """
    def __init__(self, num_classes=1000):
        super(AncestralCNN, self).__init__()
        # Early layers: localized simple-cell receptive fields (edges/lines)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # Receptive field: 11x11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                 # Spatial downsampling
            
            # Deeper layers: complex cell abstractions (parts/structures)
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Spatial reduction to fixed vector representation
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # High-level semantic templates mapping to class logits
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(384 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes), # Output logits for classes (e.g., 1000 for ImageNet)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

---

#### 4. Visual Intuition & Interpretability

The lecture relies heavily on psychological, biological, and visual illustrations of visual processing constraints and model behavior:

##### Receptive Field Hierarchy
*   **Hubel & Wiesel's Cat Experiment (1959):** Visual processing begins in the primary visual cortex (V1) where individual neurons fire only in response to edges/slits at specific orientations and moving lines. As signals propagate deeper, receptive fields become wider, mapping simple edges into corners, then parts, and finally objects.

##### Human Visual Biases & Heuristics
*   **Adelson's Checkerboard Shadow Illusion:** Squares $A$ and $B$ have the identical grayscale luminance value, yet the human brain perceives square $B$ as significantly lighter because evolution has pre-wired human vision to normalize local color estimates against global environmental context, physics priors (shadow casting), and expected illumination sources.
*   **The Stroop Test:** Measures interference in the reaction time of a task. Differentiating the color of a written word as opposed to reading the word itself triggers cognitive conflict, illustrating that high-level reading pathways compete with and dominate lower-level color classification pathways.
*   **Point-Light Display Experiment (Johansson):** A video played at 10 Hz consisting of only moving dots (attached to human joints) allows humans to instantly identify complex dynamic activities (e.g., walking, jumping) in under 100ms. This highlights that temporal motion is decoupled from static appearance and acts as a powerful, standalone visual modality.

---


<div id="plotly-cs231n-1-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 1 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)
To represent the core historical and performance transition in computer vision, we propose an interactive **Model Architecture and Dataset Co-evolution Plot**:

*   **Visualization Type:** 3D Scatter Plot with interactive sliders.
*   **Data Fields & Encoding:**
    *   **X-axis:** Number of Parameters / Capacity (Log Scale).
    *   **Y-axis:** Dataset Scale (Number of annotated images, from Caltech-101 to ImageNet-22k).
    *   **Z-axis (represented by marker size & color):** Top-5 Accuracy Rate (larger/brighter = higher accuracy).
    *   **Shape:** Representing model type (Triangle = Hand-designed filters like SIFT, Circle = Deep Learning CNNs, Square = Transformers).
*   **Interactive Controls:**
    *   **Timeline Slider (1960–2025):** Toggles active models on the plot. Dragging the slider from 1966 (MIT Summer Vision) to 2012 (AlexNet) visually displays the "AI Winter" gap where models had high theoretical capacity but data was lacking, culminating in the vertical accuracy explosion of 2012.
    *   **Compute Budget Toggle (G-FLOPs):** Color-codes models by the floating-point compute required, illustrating hardware's role as a converging force.

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **The Power of Scale:** The ImageNet dataset contains over **15 million images** organized across **22,000 categories** based on cognitive psychology hierarchies.
*   **ILSVRC Challenge Receptive Milestones:**
    *   **2010 Baseline:** Classical computer vision algorithms (hand-engineered SIFT features + SVM classifiers) yielded a **~30% Top-5 error rate**.
    *   **2012 (AlexNet):** Deep learning cut the error nearly in half to **~15.3%**, using a deep CNN trained on 2 Nvidia GPUs for several days.
    *   **Human Performance Baseline:** Psychologists establish standard human visual categorization error on the 1000-class ImageNet subset at **~3.0%**.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions
*   **The AI Winter Trap:** Designing highly complex, mathematically elegant architectures (like rod-and-cylinder decompositions in the 1970s) without considering how to automatically learn parameters from data or run them on parallel hardware.
*   **The "Incidental" Text Trap:** Scraping uncurated internet data to train multimodal models often fails because web text is subjective/incidental rather than descriptive of physical scene layouts (e.g., people write their feelings rather than stating "a lamp is to the left of the bed").

##### Graduate-Level Reflection Questions
1.  **The Ill-Posed Inverse Problem:** Why is recovering 3D visual structures from a single 2D projection mathematically ill-posed, and how do modern biological vision systems use stereopsis and structural priors to bypass this constraint?
2.  **Fukushima vs. LeCun:** Fukushima's Neocognitron (1980) had the structural primitives of convolution and pooling, yet Yann LeCun's LeNet (1998) is heralded as the modern CNN pioneer. What theoretical breakthrough in 1986 enabled this transition, and why is structural hierarchy alone insufficient for generalization?
3.  **The "Bitter Lesson" of ImageNet:** Why did high-capacity networks like CNNs stall in the early 2000s when applied to datasets like Caltech-101 or Pascal VOC, and what does this reveal about the relationship between parameter capacity, dataset cardinality, and generalization bounds?
