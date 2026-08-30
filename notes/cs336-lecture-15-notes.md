# CS336 Lecture 15: Post-Training (SFT & Preference Optimization)

## 0. Quick-Recall Summary
- **SFT Mode Extraction**: Supervised Fine-Tuning acts as a "mode extraction" mechanism to pull instruction-following capabilities already hidden in the pre-trained weights, requiring very few high-signal examples.
- **Tail Knowledge Overfitting**: SFT on facts that the model does not already possess forces memorization of formal formats (e.g. citation tags) and actively triggers hallucinations.
- **Mid-Training Decay Mixing**: Modern pre-training pipelines mix SFT conversational, math, and code data directly into the final pre-training decay/annealing phase.
- **DPO (Direct Preference Optimization)**: Bypasses the active reward model and reinforcement learning loop entirely by mathematically reformulating preference loss in terms of the generator policy.
- **Preference Length Hacking**: Preference models (including LLM-as-a-judge) are easily gamed by verbose formatting, necessitating length-controlled debiasing regressions.

## 1. Core Paradigm & Systems Overview
- **Objective**: Adapt base model completions into helpful, harmless, and instruction-compliant chat agents using supervised demonstrations and pairwise human preference signals.
- **Primary Bottleneck**: Style-capability trade-off and reward hacking. Optimizing win-rates often leads to models that output bloated, sycophantic text rather than actually smarter completions.
- **Builds on**: Evaluation Metrics (Lecture 12) — relies on robust alignment scoring and debiased judges to evaluate SFT and preference updates.

## 2. Theoretical & Mathematical Primitives

### Bradley-Terry Preference Model
Given prompt $x$ and completions $y_w, y_l$ (where $y_w$ is preferred over $y_l$), the probability that $y_w \succ y_l$ under reward model $r(x, y)$ is:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + \exp(r(x, y_l) - r(x, y_w))}$$

### KL-Regularized RLHF Objective
The standard RLHF objective seeks to maximize expected reward while penalizing policy drift from reference model $\pi\_{\text{ref}}$:

$$\max\_{\pi_\theta} \mathbb{E}\_{x \sim \mathcal{D}, y \sim \pi_\theta(y | x)} [r(x, y)] - \beta \mathbb{D}\_{\text{KL}}(\pi_\theta(y | x) \| \pi\_{\text{ref}}(y | x))$$

### DPO Objective Derivation
The closed-form analytical solution to the KL-regularized objective (for an unconstrained nonparametric policy $\pi^*$) is:

$$\pi^*(y | x) = \frac{1}{Z(x)} \pi\_{\text{ref}}(y | x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

Rearranging to solve for the implicit reward $r(x, y)$:

$$r(x, y) = \beta \ln \frac{\pi^*(y | x)}{\pi\_{\text{ref}}(y | x)} + \beta \ln Z(x)$$

Substituting this implicit reward back into the Bradley-Terry preference loss cancels out the partition function $Z(x)$, resulting in the pure **DPO Loss**:

$$\mathcal{L}\_{\text{DPO}}(\pi_\theta; \pi\_{\text{ref}}) = -\mathbb{E}\_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi\_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi\_{\text{ref}}(y_l | x)} \right) \right]$$

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### DPO Gradient Step Workflow
1. Sample a prompt $x$, preferred response $y_w$, and dispreferred response $y_l$ from preference dataset $\mathcal{D}$.
2. Compute forward pass log-probabilities on the current policy: $\log \pi_\theta(y_w | x)$ and $\log \pi_\theta(y_l | x)$.
3. Compute forward pass log-probabilities on the frozen reference model: $\log \pi\_{\text{ref}}(y_w | x)$ and $\log \pi\_{\text{ref}}(y_l | x)$.
4. Compute the implicit rewards and their difference.
5. Compute Sigmoid loss and backward pass gradients to update $\theta$.

### PyTorch/Pythonic Blueprint (Educational DPO Loss)
```python
import torch
import torch.nn.functional as F

def compute_dpo_loss(policy_model, ref_model, prompt_ids, preferred_ids, dispreferred_ids, beta=0.1):
    # Helper to extract token log-likelihoods
    def get_log_probs(model, inputs, targets):
        outputs = model(inputs)
        log_probs = F.log_softmax(outputs, dim=-1)
        # Gather along target indices
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(-1)

    # 1. Compute current policy log probs
    log_pi_w = get_log_probs(policy_model, prompt_ids, preferred_ids)
    log_pi_l = get_log_probs(policy_model, prompt_ids, dispreferred_ids)
    
    # 2. Compute reference model log probs (frozen)
    with torch.no_grad():
        log_ref_w = get_log_probs(ref_model, prompt_ids, preferred_ids)
        log_ref_l = get_log_probs(ref_model, prompt_ids, dispreferred_ids)
        
    # 3. Formulate DPO loss
    policy_ratio = log_pi_w - log_pi_l
    ref_ratio = log_ref_w - log_ref_l
    
    loss = -F.logsigmoid(beta * (policy_ratio - ref_ratio)).mean()
    return loss
```

## 4. Hardware Realities & Compute/Memory Accounting
- **Dual-Model Memory Wall**: DPO requires keeping *two* full-scale models (the trainable policy $\pi_\theta$ and the frozen reference model $\pi\_{\text{ref}}$) in active device memory. This doubles the weight memory footprint, making pipeline sharding and parameter offloading (FSDP) essential to prevent Out-Of-Memory (OOM) failures.

## 5. Systems Warnings, Pitfalls, & Reflection Questions

<div id="plotly-cs336-15-dpo-loss" class="plotly-chart" aria-label="Interactive Plotly chart: DPO Loss and Gradient Scale"></div>

<p><em>Figure: Direct Preference Optimization (DPO) Loss & Gradient Scale — Gradient vanishes when the model correctly prefers the chosen response.</em></p>

<div id="plotly-cs336-15-sft-mode" class="plotly-chart" aria-label="Interactive Plotly chart: SFT Mode Extraction"></div>

<p><em>Figure: Supervised Fine-Tuning acts as mode extraction, concentrating diffuse pre-training probabilities into structured assistant formats.</em></p>


### Gotchas & Common Bugs
- **The Citation Hallucination Spiral**: SFT on dense bibliographies teaches the model the *format* of citations rather than the *knowledge*. At test time, when asked about rare topics, the model will output plausible-looking, completely hallucinated URLs and ISBNs to satisfy the formatting constraint.

### Conceptual Reflection Questions
1. *Why does DPO successfully optimize preferences without requiring a reward model or active sampling?*
   **Answer**: Because there is a 1-to-1 analytical mapping between any reward function and its optimal policy. DPO leverages this relationship to substitute the reward model with the ratio of the policy and reference probabilities, directly optimizing the model on preference pairs and eliminating the unstable RL policy gradient step entirely.

2. *Why is pure SFT insufficient for calibration of model certainty?*
   **Answer**: SFT uses standard cross-entropy loss, forcing the model to fit every token in the demonstration dataset with equal confidence. It does not penalize the model for generating plausible lies. Preference optimization and RL verify model boundaries by explicitly penalizing false outputs, forcing the policy to shift probability mass to "I don't know" when confidence is low.
