# CS336 Lecture 16: Post-Training — RLVR (Verifiable Rewards)

## 0. Quick-Recall Summary
- **The Limit of preference RLHF**: Preferences are highly hackable and suffer from overoptimization, capping model reasoning improvements.
- **RLVR (Reinforcement Learning from Verifiable Rewards)**: Leverages objective, external verifiers (compilers, math answer parsers) to provide non-hackable, ground-truth rewards.
- **GRPO (Group Relative Policy Optimization)**: Eliminates the large value-model network entirely, estimating relative advantages directly from group rollout z-scores.
- **System-Level RL Scheduling**: Generates multiple rollouts concurrently on dedicated GPU inference servers and streams sequences to training servers to coordinate off-policy updates.
- **Reward Hacking in Compilers**: RLVR agents can find exploits to hack external checkers, such as manipulating git history or Lean compiler flags to verify broken proofs.

## 1. Core Paradigm & Systems Overview
- **Objective**: Unleash infinite reinforcement learning compute scaling on logical and mathematical tasks by replacing fallible human preference models with exact programmatic verification.
- **Primary Bottleneck**: Synchronization and heterogeneous workloads. Pipelining high-temperature rollout generations (inference-bound) and gradient backpropagation updates (compute-dense training) introduces severe latency bubbles.
- **Builds on**: Preference Optimization (Lecture 15) — expands policy gradients and KL regularization to verifiable environments.

## 2. Theoretical & Mathematical Primitives

### GRPO Advantage Estimation
For a prompt $x$, GRPO draws a group of $G$ output completions $[o_1, o_2, \dots, o_G]$ from the policy $\pi_\theta$. The advantage $A_i$ for each rollout $o_i$ is computed as:

$$A_i = \frac{r(o_i) - \mu}{\sigma + \epsilon}$$

Where:
- $\mu = \frac{1}{G} \sum\_{g=1}^G r(o_g)$
- $\sigma = \sqrt{\frac{1}{G} \sum\_{g=1}^G (r(o_g) - \mu)^2}$
- $\epsilon$: A tiny stabilizer (typically $10^{-4}$) to prevent division by zero.

### GRPO Clipped Surrogate Loss
The complete GRPO objective optimizes the clipped surrogate reward with token-wise KL regularization:

$$\mathcal{L}\_{\text{GRPO}}(\theta) = \frac{1}{G} \sum\_{g=1}^G \sum\_{t=1}^{T_g} \left( \min \left( \rho_t(\theta) A_g, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_g \right) - \beta \mathbb{D}\_{\text{KL}}(\pi_\theta \| \pi\_{\text{ref}}) \right)$$

Where the probability ratio is defined token-wise as:

$$\rho_t(\theta) = \frac{\pi_\theta(o\_{g, t} | x, o\_{g, <t})}{\pi\_{\theta\_{\text{old}}}(o\_{g, t} | x, o\_{g, <t})}$$

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### GRPO Step Execution Workflow
1. Sample prompt $x$ and generate $G$ parallel completions via temperature sampling: $o_1, \dots, o_G \sim \pi_\theta$.
2. Pass each completion to the verifiable environment (compiler/calculator) to compute rewards $r(o_1), \dots, r(o_G)$.
3. Compute group mean $\mu$ and standard deviation $\sigma$, and z-score the advantages $A_i$.
4. Evaluate forward log-likelihoods for both $\pi_\theta$ and reference model $\pi\_{\text{ref}}$.
5. Compute the clipped GRPO loss and take gradient steps.

### PyTorch/Pythonic Blueprint (Educational GRPO Step)
```python
import torch

def grpo_loss_step(policy, ref_policy, prompt_ids, rollouts_list, rewards_list, beta=0.01, clip_eps=0.2):
    # G = len(rollouts_list)
    rewards = torch.tensor(rewards_list, dtype=torch.float32)
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-4
    advantages = (rewards - mean_r) / std_r  # Group Relative Z-Score
    
    total_loss = 0.0
    for i, (rollout, adv) in enumerate(zip(rollouts_list, advantages)):
        # Compute token-wise log probs
        log_pi = policy.get_log_probs(prompt_ids, rollout)
        log_ref = ref_policy.get_log_probs(prompt_ids, rollout)
        
        # Approximate KL divergence token-wise
        kl = torch.exp(log_ref - log_pi) - (log_ref - log_pi) - 1.0
        
        # Probability ratio (off-policy correction)
        ratio = torch.exp(log_pi - log_pi.detach())
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        
        loss = -(torch.min(surr1, surr2) - beta * kl).mean()
        total_loss += loss
        
    return total_loss / len(rollouts_list)
```

## 4. Hardware Realities & Compute/Memory Accounting
- **Value Model Memory Elimination**: Standard actor-critic PPO requires running two synchronized, parameter-matched frontier networks (the Actor Policy and the Critic Value Model). Since GRPO estimates advantages directly from rollout statistics, it completely eliminates the Critic Value Model. This saves **50% of weight memory overhead**, freeing up HBM to double rollout batch sizes.

## 5. Hyperparameter Heuristics

<div id="plotly-cs336-16-grpo-advantage" class="plotly-chart" aria-label="Interactive Plotly chart: GRPO Advantage Normalization"></div>

<p><em>Figure: GRPO Advantage Normalization ($A_i = \frac{r_i - \mu}{\sigma}$) across group completions eliminates the need for a critic model in RLVR.</em></p>

<div id="plotly-cs336-16-test-time-compute" class="plotly-chart" aria-label="Interactive Plotly chart: Test-Time Compute Scaling"></div>

<p><em>Figure: Test-Time Compute Scaling — Sampling multiple rollouts and voting/verifying unlocks substantial accuracy gains on reasoning benchmarks.</em></p>

- **Format Rewards**: Under GRPO, training stability requires a composite reward function: $R = R\_{\text{accuracy}} + R\_{\text{format}}$. Format rewards penalize the policy if its chain-of-thought does not properly start with `<think>` and end with `</think>` tags.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **Straggler Node Bottlenecks**: During parallel rollout generation, high-temperature sampling results in highly variable response lengths (the "long thinking" tail). In a synchronous execution grid, the entire cluster is bottlenecked by the single slowest rank chugging on a maximal-token rollout.

### Conceptual Reflection Questions
1. *Why does outcome supervision successfully induce logical reasoning behaviors without requiring intermediate step grading?*
   **Answer**: Outcome supervision rewards only final correctness. By avoiding intermediate human rubrics (process supervision), the model is free to explore arbitrary logical paths. The RL gradient update naturally reinforces whichever search steps successfully yield the correct final answer, allowing complex reasoning patterns to emerge organically.

2. *Explain why the standard deviation term in the denominator of the GRPO advantage estimation must be stabilized with an added epsilon.*
   **Answer**: On highly uniform tasks (such as simple math questions where all rollouts return identical rewards, e.g., all 0s or all 1s), the rollout variance is exactly zero. Without adding a stabilizer $\epsilon$, dividing by a zero standard deviation would trigger numerical overflow, causing gradient explosion.
