# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 17: Robot Learning

#### 0. Quick-Recall Summary
*   **The Closed-Loop Interaction Axiom:** Unlike static computer vision tasks that map inputs $X$ to outputs $Y$, robot learning operates in a dynamic, closed-loop system where actions $a_t$ directly alter the physical environment, feeding back novel, out-of-distribution observations $s\_{t+1}$ and rewards $r_t$.
*   **The Three Core Pillars of Learning:** Robot control is categorized into three paradigms: (1) **Reinforcement Learning** (trial-and-error reward maximization), (2) **Model-Based Planning** (learning a world model to predict and plan transitions), and (3) **Imitation Learning** (distilling expert demonstrations into policies).
*   **Active and Situated Perception:** Robotic vision is embodied; agents act as active perceivers that choose *where* to look, *how* to perturb the environment (e.g., pushing stacked objects to resolve semantic instance ambiguity), and focus exclusively on task-relevant regions.
*   **Cascading Error (Covariate Shift):** The primary failure mode of basic behavior cloning (imitation learning) is error compounding. Tiny one-step prediction errors alter the state trajectory, driving the agent into unobserved, out-of-distribution regions where the policy has no training signal.
*   **Robotic Foundation Models (VLAs):** Modern Vision-Language-Action (VLA) models (e.g., $\pi_0$) leverage pre-trained visual-language backbones co-fine-tuned on action-prediction objectives. This transfers high-level semantic reasoning and cross-modal open-vocabulary capabilities to real-world control.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To introduce the algorithmic formalisms and systems challenges of robot learning, focusing on closing the loop between perceptual inputs and physical actions. The lecture traces the evolution from game-theoretic model-free learning to physics-grounded world models, visual imitation policies, and modern multi-embodiment VLA foundation models.
*   **Lecture Category:** (e) Systems/Scaling and (d) Specific Vision Task (Robot Control and Interaction).
*   **Builds on:** Combines 2D ConvNets (Lecture 5 & 6) and Multimodal VLMs (Lecture 16) to translate perceptual tokens into continuous low-level action spaces (velocities and joint torques).

---

#### 2. Mathematical Foundations

##### Markov Decision Process (MDP) and Reinforcement Learning Formulation
Robot control is formally modeled as a Markov Decision Process defined by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
*   **$\mathcal{S}$:** State space describing the physical system.
*   **$\mathcal{A}$:** Action space (e.g., continuous joint torques or 6-DoF end-effector velocities).
*   **$\mathcal{P}(s\_{t+1} | s_t, a_t)$:** Environmental transition probability distribution (stochastic dynamics).
*   **$\mathcal{R}(s_t, a_t)$:** Reward function mapping state-action pairs to scalar performance metrics.
*   **$\gamma \in [0, 1)$:** Discount factor modeling the preference for immediate over delayed rewards.

The global optimization objective is to find a policy $\pi_\theta(a_t | s_t)$ that maximizes expected cumulative discounted return:
$$\max\_{\theta} \mathbb{E}\_{\tau \sim \pi_\theta} \left[ \sum\_{t=0}^{T} \gamma^t \mathcal{R}(s_t, a_t) \right]$$
where $\tau = (s_0, a_0, s_1, a_1, \dots, s_T, a_T)$ represents the trajectory rolled out under policy $\pi_\theta$.

##### Bellman Optimality Equation (Q-Learning)
Deep Q-Networks (DQNs) model the state-action value function $Q^*(s, a)$, representing the expected future discounted reward when executing action $a$ in state $s$ and acting optimally thereafter:
$$Q^*(s, a) = \mathcal{R}(s, a) + \gamma \mathbb{E}\_{s' \sim \mathcal{P}(\cdot | s, a)} \left[ \max\_{a\'} Q^*(s\', a\') \right]$$
In practice, the network parameter $\theta$ is updated by minimizing the Mean Squared Bellman Error (MSBE):
$$\mathcal{L}(\theta) = \mathbb{E}\_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max\_{a\'} Q\_{\theta^-}(s\', a\') - Q_\theta(s, a) \right)^2 \right]$$
where $\theta^-$ represents the weights of a lagging, target network used to stabilize optimization.

##### Model-Based Planning and Receding Horizon Control (MPC)
Model-based agents learn a forward dynamics model (world model) $f_\phi(s_t, a_t) \approx s\_{t+1}$ by minimizing prediction error:
$$\mathcal{L}(\phi) = \sum\_{t=0}^{T-1} \| f_\phi(s_t, a_t) - s\_{t+1} \|_2^2$$
To plan, the agent solves an online trajectory optimization problem over a finite temporal horizon $H$ to reach a target state $s^*$:
$$\min\_{a\_{t..t+H}} \sum\_{\tau=t}^{t+H} \| \hat{s}\_{\tau+1} - s^* \|_2^2 \quad \text{subject to} \quad \hat{s}\_{\tau+1} = f_\phi(\hat{s}_\tau, a_\tau)$$
Using **Model Predictive Control (MPC)**, the agent executes only the first planned action $a_t$, receives the true feedback state $s\_{t+1}$ from the physical environment, and re-optimizes the sequence over the shifted horizon.

##### Imitation Learning and Behavior Cloning
Given a dataset of expert demonstrations $\mathcal{D} = \{(o_i, a_i)\}\_{i=1}^N$, **Behavior Cloning (BC)** optimizes the policy $\pi_\theta$ via supervised Maximum Likelihood Estimation:
$$\mathcal{L}\_{BC}(\theta) = -\sum\_{(o_i, a_i) \in \mathcal{D}} \log \pi_\theta(a_i | o_i)$$
For continuous action spaces, this collapses to Mean Squared Error (MSE) minimization under a Gaussian policy assumption:
$$\mathcal{L}\_{BC}(\theta) = \sum\_{(o_i, a_i) \in \mathcal{D}} \| \pi_\theta(o_i) - a_i \|_2^2$$

---

#### 3. Architecture / Algorithm Walkthrough

##### Closed-Loop Perception-Action Pipelines
The flow of physical information and gradients contrasts model-free, model-based, and imitation loops:

```
1. Model-Free RL (Trial & Error):
   [State s_t] ──> [Policy π_θ] ──> [Action a_t] ──> [Physical World] ──> [Reward r_t, s_t+1]
        ▲                                                                        │
        └─────────────────────────── [Update via Q-Learning/PPO] ────────────────┘

2. Model-Based Planning (Internal Imagined Rollout):
   [State s_t] ──> [Sampling/Optimizer (CEM/MPPI)] ──> [Action Candidates]
                         ▲                                    │
                         │                                    ▼
                 [Backprop Gradients] ─── [World Model f_ϕ] ◄─┘
                         │                       │ (Unroll H steps)
                         │                       ▼
                 [Minimize Distance] ◄─── [Predicted States s_t+1..t+H]

3. Behavior Cloning / VLA Foundation:
   [Multimodal Context: (Images, Text)] ──> [VLA Policy (e.g., π_0)] ──> [Low-Level Actuations]
```

##### PyTorch Blueprint (Closed-Loop Imitation Policy with ConvNet Backbone)
This modular blueprint implements a continuous behavior cloning policy. It ingests multi-view RGB observations, flattens spatial activations into a joint embedding, and outputs a 7-dimensional continuous control action (6-DoF velocity vector + gripper state).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotImitationPolicy(nn.Module):
    """
    Continuous imitation learning policy matching the behavior cloning formulation.
    Ingests multi-view visual observations and outputs low-level robot actions.
    """
    def __init__(self, action_dim=7, hidden_dim=256):
        super(RobotImitationPolicy, self).__init__()
        
        # Shared visual feature extractor representing simple-to-complex V1 pathways
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), # Receptive field projection
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # Dimensionality bottleneck matching physical constraints
        )
        
        # Policy head mapping visual representations to continuous action values
        self.fc_policy = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1), # Regularization to mitigate visual overfitting
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim) # Output: [dx, dy, dz, droll, dpitch, dyaw, gripper_state]
        )

    def forward(self, observation):
        """
        Args:
            observation (Tensor): Batch of raw camera frames [B, C, H, W]
        Returns:
            actions (Tensor): Continuous robotic control signals [B, action_dim]
        """
        features = self.conv_backbone(observation)
        features_flat = torch.flatten(features, 1)
        actions = self.fc_policy(features_flat)
        
        # Gripper state squeezed to range [-1, 1] using tanh activation
        actions[:, -1] = torch.tanh(actions[:, -1])
        return actions

class MSEActionLoss(nn.Module):
    """
    Mean Squared Error loss to minimize difference between policy outputs
    and expert demonstrations.
    """
    def __init__(self):
        super(MSEActionLoss, self).__init__()

    def forward(self, predicted_actions, expert_actions):
        return F.mse_loss(predicted_actions, expert_actions)
```

---

#### 4. Visual Intuition & Interpretability

##### Active Perception to Resolve Scene Ambiguity
*   **The Semantic Occlusion Boundary:** In passive computer vision, overlapping objects (e.g., stacked blocks or nested bowls) produce ambiguous visual boundaries. 
*   **Perturbation Dynamics:** An embodied agent uses active perception: it physically nudges, slides, or shakes the visual cluster. By observing how parts move relative to one another over time (motion boundaries), the agent resolves visual ambiguity, segmenting single objects from composite structures.

##### Compounding Trajectory Error (Covariate Shift Visualization)
The fundamental divergence between supervised static classification and closed-loop imitation learning is represented below:

```
Expert Trajectory (Safe Zone):
───────────────────────────────────────────────────────────────► (Goal Met)
           ▲             ▲             ▲             ▲
           │             │             │             │
        (Action 1)    (Action 2)    (Action 3)    (Action 4)
           │             │             │             │
Policy Rollout with Compounding Error:
───────────┴─────────────┼─────────────┼─────────────┼─────────► (Catastrophic Failure)
                         ▼             ▼             ▼
                    Small Error   Large Shift   Out-of-Distribution
```
A small initial prediction error pushes the robot into a state slightly off the demonstration path. Because the training data contains no correction maneuvers for this new state, the policy makes an even larger error, compounding exponentially until physical system failure.

##### Particle Dynamics vs. Analytical Mechanics
*   **Physics Simulator Limitations:** Traditional Material Point Method (MPM) or analytical rigid-body mechanics struggle to model the highly complex, non-rigid, and history-dependent behaviors of highly deformable objects like dough or fluids.
*   **Visual World Models:** By tracking objects as thousands of interacting spatial particles and training a deep graph neural network on real-world interactions, the model learns complex topological deformations (e.g., folding and cutting dough) that bypass analytical simulator failures.

---


<div id="plotly-cs231n-17-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 17 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)
To isolate and understand compounding covariate shift under different imitation learning algorithms, we propose an interactive **Robotic Policy Trajectory Divergence Simulator**:

*   **Visualization Type:** 3D Path Renderer with dynamic state-space distributions.
*   **Data Fields & Encoding:**
    *   **X & Y dimensions:** Physical coordinates of the robot end-effector on a table.
    *   **Z dimension:** Temporal sequence steps.
    *   **Color-coded lines:** Green line = Expert demonstrations ($\mathcal{D}$); Red line = Rolled-out policy trajectory ($\hat{\tau}$).
    *   **Covariate Shading (Heatmap Overlay):** Shading intensity maps the distance to the nearest training sample. Bright regions are in-distribution; dark regions represent unknown state spaces.
*   **Interactive Controls:**
    *   **Algorithm Selector:** Toggle between (a) Pure Behavior Cloning, (b) DAgger (interactive online correction), and (c) Diffusion Policy.
    *   **Noise Injection Slider:** Artificially inject rotational or translational perturbations to watch how quickly pure Behavior Cloning trajectories escape the bright "safe zone" and cascade into failure, while DAgger or Diffusion policies execute corrective trajectories back to the green path.

---

#### 6. Empirical Design Heuristics & Benchmark Results

##### System and Dataset Characteristics
*   **The Power of Extreme Domain Randomization:** In locomotion tasks, Eth-Zurich successfully crossed the sim-to-real barrier by randomizing friction parameters, joint torques, mass, and surface slope angles by massive margins within simplified rigid body simulations, showing that highly robust policies generalize directly to unmodeled, slippery, and uneven real-world environments (snow, mud, loose rocks).
*   **The Manipulation Bottleneck:** Unlike locomotion (which is robustly solved via model-free RL and domain-randomized simulations), robot manipulation faces critical sim-to-real constraints. Small mismatches in tactile shear forces or gripper surface contact physics cause continuous objects to slip, slide, or catastrophically fly apart.
*   **Teleoperation Inefficiencies:** Robotic foundation models are bottlenecked by data collection pipelines. Humans teleoperating physical robots (e.g., using aloha rigs) are significantly slower than direct human hands due to joint-mapping latency and camera-perspective occlusions.

##### Model and Policy Baselines
*   **Model-Free RL Sample Complexity:** Model-free algorithms are exceptionally expensive; AlphaGo Zero required the equivalent of **3,000 years of human play** simulated in 40 days to discover superhuman tactics.
*   **PI-0 Foundations:** By initializing policies with pre-trained visual-language encoders and performing joint co-fine-tuning (action token prediction + visual-question answering), the base network inherits general visual-semantic priors from internet-scale datasets. This reduces task-specific data demands down to small, task-specific aloha demonstrations during post-training fine-tuning.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Silent Failure Modes
*   **The One-Step Loss Optimization Mirage:** A low Mean Squared Error (MSE) on the training set does not guarantee closed-loop trajectory success. A model can easily memorize static 1-step mapping combinations while remaining incapable of recovering from tiny physical perturbations over long temporal horizons.
*   **The Blind Spot of Passively Collected Datasets:** Standard computer vision datasets contain no causal understanding of actions. If a model only observes static images or passive video frames, it cannot learn the physical transition boundaries that result from deliberate interactions.
*   **The Unified Controller Collapse:** Attempting to represent all complex physical actions (cleaning, folding laundry, setting tables) under a single, non-structured policy fails. High-level task symbolic structures (scene graphs) must be decoupled from the low-level motor policy parameters to enable long-horizon task completion.

##### Graduate-Level Reflection Questions
1.  **Covariate Shift vs. Static Generalization:** Explain why a static image classification network trained on out-of-distribution (OOD) test sets degrades gracefully, whereas an imitation learning robot policy experiencing a tiny execution drift suffers from catastrophic, exponential trajectory collapse. How does online human-in-the-loop data aggregation resolve this?
2.  **Explicit vs. Implicit Actions:** Traditional behavior cloning outputs a deterministic single action vector $\pi(o) = a$. Explain how modeling the policy *implicitly* using energy-based models or diffusion frameworks enables the robot to handle highly multi-modal expert demonstrations (e.g., deciding whether to reach for a block from the left or right side).
3.  **The "Bitter Lesson" of Sim-to-Real:** Why does extreme domain-randomization succeed in transferring quadrupedal locomotion from simple simulation environments to real physical terrain, but fails to generalize robustly to fine-grained, tactile manipulation tasks like peeling a potato or folding loose cloth? Detail the physical trade-offs involved.