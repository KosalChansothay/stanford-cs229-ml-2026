# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 18: Human-Centered AI

#### 0. Quick-Recall Summary
*   **The Evolutionary Timeline:** Biological vision evolved over 540 million years starting from the "Cambrian Explosion", serving as the visual wetware that inspires deep architectural hierarchies.
*   **The Beerman Number:** Psychologists conjecture that human children by age six or seven can recognize $30,000$ to $100,000$ distinct visual categories. This inspired the scale of ImageNet ($22,000$ classes, $15\text{M}$ images).
*   **Structured scene Graphs:** Moving beyond isolated object classification to scene graphs ($G = (V, E)$) containing directional relationships, enabling zero-shot recognition of uncommon combinations (e.g., "horse wearing a hat").
*   **Human Visual Limitations:** Humans suffer from massive attentional and cognitive constraints, illustrated by change blindness (flicker tests), the Stroop conflict, and Adelson's checkerboard luminance illusion.
*   **The Embodied Generalization Gap:** The BEHAVIOR benchmark ($1,000$ everyday tasks, $10,000+$ objects) reveals that state-of-the-art closed-loop robotic policies get $0\%$ success in unprivileged "in-the-wild" environments, highlighting a massive research frontier.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To examine deep learning and computer vision through a human-centered lens, tracing the shift from replicating biological visual capabilities to augmenting human physical and cognitive limitations. It frames AI as an assistive, closed-loop cooperative tool (medical monitoring, privacy-preserving cameras, and embodied assistive robotics) rather than a replacement for human labor.
*   **Lecture Category:** (a) Specific Vision Tasks (grounding, scene understanding, action recognition) and (b) Systems/Applications (Embodied AI/Robotics, Privacy-preserving hardware, Ambient Healthcare).
*   **Builds on:** Integrates the deep foundations of previous lectures (transformers, CNNs, self-supervised pre-training, and video understanding) into real-world, closed-loop human-interactive domains.

---

#### 2. Mathematical Foundations
While Lecture 18 is a high-level conceptual, historical, and philosophical talk rather than an algorithmic derivation class, the systems described rely on several key mathematical formalisms:

*   **Scene Graph Representation:**
    A visual scene graph is defined as a directed, labeled graph:
    $$G = (V, E)$$
    where $V = \{o_1, o_2, \dots, o_n\}$ represents the set of object entity nodes (with predicted class labels and spatial bounding boxes), and $E = \{r\_{1,2}, \dots\}$ represents the set of directed, labeled relationship edges mapping spatial, semantic, or action dependencies between entities.
*   **The Perspective Projection Bottleneck (Mathematical Ill-Posedness):**
    Recovering 3D scene coordinate geometry $\mathbf{X} \in \mathbb{R}^3$ from a single 2D pixel coordinate $\mathbf{x} \in \mathbb{R}^2$ is mathematically ill-posed due to the projective collapse of depth:
    $$\mathbf{x} \propto \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \mathbf{X}$$
    where $\mathbf{K}$ is the camera intrinsic matrix, and $[\mathbf{R} \mid \mathbf{t}]$ represents extrinsic rotation and translation. Depth is collapsed to a scalar multiplier, requiring stereoscopic triangulation or learned structural depth priors to invert.
*   **Ambient Multi-Class Action Classification (Healthcare Monitoring):**
    Action classifiers in clinical spaces (e.g., hand hygiene compliance) process raw temporal video frames $X \in \mathbb{R}^{T \times C \times H \times W}$ to output a categorical probability distribution over $K$ actions:
    $$P(y = k \mid X) = \frac{e^{\mathbf{w}_k^T \phi(X)}}{\sum\_{j=1}^{K} e^{\mathbf{w}_j^T \phi(X)}}$$
    where $\phi(X)$ is a temporal convolutional or spatio-temporal attention feature extractor.

---

#### 3. Architecture / Algorithm Walkthrough

##### Algorithmic Logic: Closed-Loop open-Vocabulary Robotic Manipulation (VLM-LLM-MPC Pipeline)
To allow a physical robot arm to execute un-pre-trained, "in-the-wild" commands (e.g., *"Open the top drawer, but watch out for the vase"*), the system couples high-level semantic reasoning with low-level physical planning:

```
[Human Command] ──> [LLM (Writes Python APIs)] 
                          │
                          ▼
                    [VLM (Grounds Coordinates)] ──> [Generate Spatial Heatmaps]
                                                          │
                                                          ▼
                                                    [MPC / Motion Planner]
                                                          │
                                                          ▼
                                                    [Robot Trajectory]
```

1.  **Semantic Decomposition (LLM):** The natural language instruction is processed by an LLM that translates the command into executable Python code blocks invoking vision-language modules.
2.  **Visual Grounding (VLM):** The VLM processes the real-time camera feed to detect targets ("top drawer", "handle") and obstacles ("vase"), generating continuous pixel bounding boxes and segmentations.
3.  **Cost Map Generation:** Grounded coordinates are translated into 2D/3D cost maps represented as heatmaps. Targets yield positive basins of attraction, while obstacles ("vase") project high-cost obstacle avoidance gradients.
4.  **Motion Planning (Model Predictive Control / MPC):** A sampling-based trajectory planner continuously optimizes low-level joint torques to minimize the cost map landscape, executing closed-loop collision-free paths.

##### PyTorch Blueprint: Vision-Language Action Planner
This illustrative blueprint shows how to compose a pre-trained VLM detector with a mock MPC motion planning objective to adjust joint movements based on real-time visual obstacles:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VLAActionPlanner(nn.Module):
    """
    Illustrative blueprint modeling a closed-loop vision-language action 
    planner that dynamically generates obstacle cost fields to guide robotic control.
    """
    def __init__(self, visual_dim=512, hidden_dim=256):
        super(VLAActionPlanner, self).__init__()
        # Visual feature projection (mapping patch descriptors to planning latents)
        self.visual_projector = nn.Linear(visual_dim, hidden_dim)
        
        # Heatmap generator (predicts attractor and obstacle fields in spatial coordinates)
        self.cost_field_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1) # Channel 0: Attractor Field, Channel 1: Obstacle Cost Field
        )

    def forward(self, visual_features, current_state):
        """
        Args:
            visual_features: Tensor of shape (B, C, H, W) representing pre-trained VLM patch states.
            current_state: Tensor of shape (B, 3) representing the robot's current (x, y, theta) pose.
        Returns:
            optimal_action: Planned velocity delta to avoid high-cost obstacle regions.
        """
        # Project visual descriptors to latent space
        B, C, H, W = visual_features.shape
        flat_feats = visual_features.permute(0, 2, 3, 1).reshape(-1, C)
        latent_feats = self.visual_projector(flat_feats).view(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Predict spatial attractors and obstacle cost fields
        cost_fields = self.cost_field_head(latent_feats)
        attractor_map = cost_fields[:, 0, :, :] # Encouraged destinations
        obstacle_map = torch.sigmoid(cost_fields[:, 1, :, :])   # Avoidance coordinates

        # Simple Receding Horizon Optimization (Mock MPC Sample Loop)
        B, H_grid, W_grid = obstacle_map.shape
        best_action = torch.zeros(B, 2) # (dx, dy) velocity updates
        
        for b in range(B):
            grid_y, grid_x = current_state[b, 1], current_state[b, 0]
            
            # Sample candidate actions in a local neighborhood
            candidates = torch.randn(50, 2) * 2.0 
            costs = torch.zeros(50)
            
            for i, cand in enumerate(candidates):
                next_x = torch.clamp(grid_x + cand[0], 0, W_grid - 1).long()
                next_y = torch.clamp(grid_y + cand[1], 0, H_grid - 1).long()
                
                # Path Cost = Distance to Attractor - Obstacle Safety Cost
                safety_penalty = obstacle_map[b, next_y, next_x] * 10.0
                progress_reward = -attractor_map[b, next_y, next_x] # Min value represents max alignment
                costs[i] = progress_reward + safety_penalty
                
            # Select action with the lowest cost
            best_idx = torch.argmin(costs)
            best_action[b] = candidates[best_idx]
            
        return best_action, obstacle_map
```

---

#### 4. Visual Intuition & Interpretability

##### Visual Illusion & Attention Gaps
*   **The Stroop Conflict:** Differentiating word colors vs. semantic text characters forces a processing bottleneck, visually demonstrating that high-level language routing pathways compete with and delay lower-level visual pathways.
*   **Change Blindness:** Humans fail to perceive massive changes in sequential scenes (e.g., an airplane's entire jet engine disappearing during visual breaks), exposing the highly selective, sparse, and object-oriented memory constraints of human vision.
*   **Adelson's Checkerboard Shadow Illusion:** Squares $A$ and $B$ possess identical luminance values, yet the brain processes square $B$ as lighter. This visual failure exposes how biological vision uses contextual priors (shadow-casting expectations and constant illumination assumptions) to make sense of the world, rather than measuring absolute pixel values.

##### Privacy-Preserving Cameras
*   **Hardware/Software Co-Design Lens:** Blurring faces in software is vulnerable to reconstruction hacks and info leaks. A physical, hand-crafted optical lens is designed to distort visual wavelengths *before* hitting the digital sensor, protecting privacy at the physical layer while allowing a software decoder to extract human skeleton kinematics.

---


<div id="plotly-cs231n-18-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 18 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)

To represent the gap between human preferences and modern robotic performance across diverse environments, we propose an interactive **Household Tasks Generalization Map**:

*   **Visualization Type:** Hierarchical Sunburst Chart with interactive accuracy indicators.
*   **Data Fields & Encoding:**
    *   **Inner Ring:** Top-level human-centered preferences (e.g., Kitchen, Laundry, Yard, Cleaning) extracted from user surveys.
    *   **Middle Ring:** Sub-tasks (e.g., Folding Shirts, Shoveling Snow, Washing Dishes, Boiling Pasta).
    *   **Outer Ring:** Simulation Success Rate vs. Real-World Success Rate of leading closed-loop robotic models (colored dynamically: Green = $80\%+$, Yellow = $30\text{--}70\%$, Red = $0\text{--}10\%$).
    *   **Size Scaling:** Scaled proportionally to human preference intensity (tasks humans *most* want automated are larger).
*   **Interactive Controls:**
    *   **Privilege Level Slider:** Toggles model access to privileged simulator states (perfect memory, magic motion) vs. raw sensory inputs. Sliding the bar to "No Privilege" visually collapses the outer success ring to zero, highlighting the generalization bottleneck of modern algorithms.
    *   **Sim-to-Real Domain Randomization Slider:** Adjusts simulated visual distortion (lighting, friction, clutter), showing how virtual test realisms match the physical world.

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **The Biederman Number:** Evolutionary psychologists establish human visual capacity at **$30,000$ to $100,000$ categories** by age six or seven. This baseline bounded the creation of ImageNet ($22,000$ object classes, $15\text{M}$ images).
*   **Human Visual Categorization Speed:** Measured via EEG brainwave signals, humans distinguish and categorize complex visual stimuli (animals vs. non-animals) within **$150$ milliseconds** after image exposure.
*   **Surgical Item Retrieval Bottleneck:** In clinical operating rooms, search times for misplaced surgical sponges or gauze average **$1$ hour**, introducing high infection and bleeding risks, and providing a major target for computer vision tracking systems.
*   **BEHAVIOR Benchmark Scaling:** Compiles **$1,000$ tasks** preferred by humans across **$50$ fully reconstructed, photorealistic 3D virtual environments** (offices, apartments, restaurants) using **$10,000+$ articulated 3D object assets** integrated with Nvidia Omniverse.
*   **Embodied Baseline Limits:** Under unprivileged, raw sensory observations, today's leading robotic policies score **$0\%$ success rates** on core BEHAVIOR tasks, revealing the limitations of current out-of-distribution physical generalize strategies.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Silent Failures
*   **The Software-Blurring Privacy Leak:** Relying solely on software blurring or pixel mask overlays is highly insecure; deep classification networks can often decode raw actions or identify sensitive attributes from the surrounding context or residual pixel margins.
*   **The "In-Distribution Simulation" Mirage:** Getting a robotic policy to achieve $99\%$ success inside a specific physics simulator is a false success indicator; minor discrepancies in friction coefficients, material elasticity, or camera-lighting positions cause complete policy collapse in the physical world (Sim-to-Real gap).
*   **Co-occurrence Bias Trap:** Image captioning and VLM models often generate descriptions based on background co-occurrence rather than grounded evidence (e.g., predicting "surfboard" on any beach photo or "mouse" whenever a human hand is cupped).

##### Graduate-Level Reflection Questions
1.  **Inductive Bias vs. Representation Learning:** Pre-deep learning attempts at object recognition relied heavily on hand-designed geometric primitives (e.g., Rodney Brooks' "generalized cylinders"). Why did these mathematically elegant representations fail to generalize compared to high-capacity, unconstrained convolutional networks trained end-to-end?
2.  **The Sim-to-Real Locomotion/Manipulation Asymmetry:** Why does extreme domain randomization successfully bridge the sim-to-real gap for legged quadrupedal locomotion, yet consistently fail for precise physical manipulation tasks such as shirt folding or dumpling preparation?
3.  **Visual grounding vs. Autoregressive Hallucination:** How does adding spatial visual grounding outputs (like pointing/bounding coordinates in Momo) mathematically or structurally bound an LLM's next-token autoregressive generation to mitigate hallucinations?
