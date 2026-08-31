# CS336 Lecture 12: Evaluation (Methodology & Contamination)

## 0. Quick-Recall Summary
- **Evaluation Validity**: Evaluating models involves mapping abstract constructs (reasoning, safety) to concrete, graded metrics on benchmarks.
- **Standard Exams (MMLU/GPQA)**: Hendricks' MMLU (high school/college multiple choice) and GPQA (PhD-level Google-proof questions) serve as traditional benchmarks but suffer from rapid saturation.
- **Chatbot Arena (ELO)**: Evaluates open-ended human-aligned preferences by collecting blind A/B model generation votes on crowdsourced prompts to fit global ELO ratings.
- **Agentic Evaluation (SWE-bench)**: Measures code-base interaction capabilities by asking models to resolve real GitHub issues, verified strictly via execution unit tests.
- **Data Contamination**: Models trained on crawled web documents frequently encounter test set questions, leading to inflated, artificial benchmark performance.

## 1. Core Paradigm & Systems Overview
- **Objective**: Rigorously measure and compare language model capabilities across diverse dimensions (perplexity, knowledge, open-ended instruction following, agentic tool-use, reasoning) without falling victim to benchmark gaming or data contamination.
- **Primary Bottleneck**: Evaluation shelf-life and contamination. As frontier models are trained on larger slices of the public web, standard test suites are swallowed into the training corpus, rendering them useless as clean holdout measures.
- **Builds on**: Post-training (Lecture 15) — once models are instruction-tuned, evaluations must transition from simple perplexities to conversational and agentic testing harnesses.

## 2. Theoretical & Mathematical Primitives

### Perplexity Metric
Perplexity is the exponentiated cross-entropy loss of the model over a validation dataset $D = [x_1, x_2, \dots, x_N]$:

$$\text{PPL}(D) = \exp\left( -\frac{1}{N} \sum\_{i=1}^N \log P(x_i | x\_{<i})  \right)$$

At optimal convergence where the model distribution $P$ matches the true distribution $T$, perplexity equals the exponentiated Shannon entropy of the true data distribution:

$$\text{PPL}\_{\text{opt}} = \exp\left( H(T)  \right)$$

### Bradley-Terry Preference Modeling (ELO)
To fit ELO rankings from Chatbot Arena A/B comparison trials, the probability of model $A$ beating model $B$ is modeled as a sigmoid function of their latent scores $r_A, r_B$:

$$P(A \succ B) = \frac{1}{1 + 10^{(r_B - r_A)/400}} = \sigma\left( \frac{\ln(10)}{400} (r_A - r_B)  \right)$$

We estimate the rating parameters $\mathbf{r}$ by maximizing the log-likelihood of all observed comparisons $Y\_{ij}$:

$$\mathcal{L}(\mathbf{r}) = \sum\_{(i, j) \in \text{Comparisons}} \left( Y\_{ij} \log P(i \succ j) + (1 - Y\_{ij}) \log P(j \succ i)  \right)$$

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### Detecting Data Contamination (Deduplication / Order-Permutation test)
1. **Benchmark Permutation**: Benchmarks often have multiple-choice options in random orders.
2. **Order Bias Test**: If a model has memorized the test set, it will assign high probability to the exact option labels matching the original online files (e.g. always predicting "C" if that's the online key), regardless of options shuffling.
3. **Log-Likelihood Discrepancy**: Compare the perplexity of the test questions under normal vs. corrupted/shuffled configurations to identify statistical anomalies indicating training set leakage.

### PyTorch/Pythonic Blueprint (Educational ELO Rating Solver)
```python
import numpy as np
from scipy.optimize import minimize

# Educational solver for Bradley-Terry preference ratings
def compute_elo_ratings(num_models, comparisons):
    # comparisons: list of tuples (model_A_idx, model_B_idx, A_won_bool)
    def neg_log_likelihood(ratings):
        # Anchor first model rating to 1200
        full_ratings = np.zeros(num_models)
        full_ratings[1:] = ratings
        full_ratings[0] = 1200.0
        
        nll = 0.0
        for A, B, A_won in comparisons:
            prob_A_wins = 1.0 / (1.0 + 10 ** ((full_ratings[B] - full_ratings[A]) / 400.0))
            prob_A_wins = np.clip(prob_A_wins, 1e-12, 1.0 - 1e-12)
            nll -= np.log(prob_A_wins) if A_won else np.log(1.0 - prob_A_wins)
        return nll

    initial_guess = np.ones(num_models - 1) * 1200.0
    res = minimize(neg_log_likelihood, initial_guess, method="BFGS")
    final_ratings = np.zeros(num_models)
    final_ratings[1:] = res.x
    final_ratings[0] = 1200.0
    return final_ratings
```

## 4. Hardware Realities & Compute/Memory Accounting
- **Agentic Sandbox Execution**: Agentic evaluations (such as SWE-bench, TerminalBench, or cybersecurity capture-the-flag environments) require running untrusted, model-generated code in isolated sandboxes. This adds massive systems overhead compared to static perplexity checks: each verification step involves instantiating Docker containers, setting up complex dependency environments, compiling repositories, and executing test suites asynchronously across cluster workers.

## 5. Hyperparameter Heuristics

<div id="plotly-cs336-12-bradley-terry" class="plotly-chart" aria-label="Interactive Plotly chart: Bradley-Terry Win Probability"></div>

<p><em>Figure: Chatbot Arena Bradley-Terry Curve — Elo rating differences map non-linearly to expected win probabilities.</em></p>

<div id="plotly-cs336-12-judge-length-bias" class="plotly-chart" aria-label="Interactive Plotly chart: LLM Judge Length Bias"></div>

<p><em>Figure: LLM-as-a-Judge exhibits severe length bias on identical answers, requiring length-controlled calibration.</em></p>

- **LLM-as-a-Judge De-biasing**: Autoregressive judges (like GPT-4 in AlpacaEval) exhibit severe **length bias** (preferring longer responses) and **positional bias** (preferring whichever option is presented first in the prompt context). These are mitigated by shuffling presentation orders and using length-controlled regression models to subtract length correlations from final win rates.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **Goodhart's Law in RLHF**: Running RLHF against a frozen reward model without strict KL regularization causes the generator to find adversarial shortcuts (e.g. appending bloated sycophantic politeness markers) that maximize reward score while destroying actual response utility.

### Conceptual Reflection Questions
1. *Why does LLM-as-a-Judge evaluation exhibit high correlation with human ratings, yet remain highly vulnerable to programmatic exploitation?*
   **Answer**: LLM-as-a-Judge aligns well with human averages because model behavior mimics the distribution of human text on the web. However, because the judge is itself a model, it suffers from the same structural blind spots: it is easily gamed by superficial styling cues like excessive length, formal vocabulary, and structured markdown lists, even when the underlying factual accuracy of the response is flawed.

2. *How do private holding-test sets mitigate the limits of out-of-distribution evaluations?*
   **Answer**: Standard out-of-distribution evaluations become contaminated over time as their web URLs are ingested by crawlers. Private holdout tests (like those in HLE) are never published on the web, preventing crawlers from accessing them. This maintains clean holdout status, forcing model developers to rely on genuine feature generalization rather than verbatim pre-training memorization.
