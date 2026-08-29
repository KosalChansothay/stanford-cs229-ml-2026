# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 6: Dataset Split, ML Advice

### 1. Summary
This lecture addresses the central, motivating question of machine learning: given a finite, noisy sample, how do we select a model that will generalize successfully to future, unseen data? The instructor explores the fundamental tension between **underfitting (high bias)** and **overfitting (high variance)**, mapping out the classical **bias-variance decomposition** of statistical risk. To combat high variance, **regularization** techniques (using **Ridge Regression** as a primary case study) are introduced to trade a slight increase in bias for a substantial reduction in variance. 

The lecture then bridges these classical frameworks with **modern machine learning paradigms**, examining **Double Descent**—where overparameterized models with zero training error continue to generalize well past the classical interpolation threshold. It also reviews empirical findings on test set robustness, such as the **ImageNet-V2** replication, which dispels concerns of severe **adaptive overfitting**. Finally, practical workflows for **model selection** are introduced, including **$k$-fold cross-validation** and **Hyperband**, a highly compute-efficient algorithm for hyperparameter tuning.

---

### 2. Key Concepts & Definitions
- **The Generalization Goal**: Given a finite, noisy sample of training data, the objective is to choose a model that behaves stably and generalizes to future draws from the same underlying population.
- **Underfitting (High Bias)**: Occurs when a model class is too simple or lacks the expressive capacity to capture the underlying data-generating process (e.g., trying to fit a quadratic curve with a straight line).
- **Overfitting (High Variance)**: Occurs when a model class is excessively expressive relative to the dataset size, causing it to fit random noise rather than the true underlying signal (e.g., interpolating a 9-degree polynomial through noisy quadratic points).
- **Interpolation**: The regime where a model fits the training dataset perfectly, achieving exactly zero training loss.
- **Statistical Risk**: The expected squared error of a hypothesis $h_S$ evaluated on a fresh, randomly drawn test point.
- **Unavoidable Noise ($\sigma^2$)**: The intrinsic variance of the target variable $y$, representing measurement error or unmodeled factors that cannot be eliminated by any learning algorithm.
- **Inductive Bias**: The set of assumptions a learning algorithm uses to predict outputs for unseen inputs, guiding model selection (e.g., assuming a smooth, lower-degree polynomial over a highly oscillatory one).
- **Regularization**: The process of adding a soft penalty to the objective function (or using dynamic optimization techniques like dropout) to constrain parameters, shrink their magnitude, and dramatically reduce variance.
- **$k$-fold Cross-Validation**: A data-efficient model evaluation protocol where the training set is partitioned into $k$ equal folds; the model is trained $k$ times on $k-1$ folds and validated on the remaining fold, allowing hyperparameter tuning without leaking test set information.
- **Double Descent**: A modern statistical phenomenon where increasing model capacity beyond the interpolation threshold causes the test error to peak and then descend again, often reaching generalization levels superior to those of the classical under-parameterized regime.
- **Adaptive Overfitting**: The risk that collective, iterative benchmarking on a public test set (like ImageNet) implicitly leaks test set characteristics back into model designs, leading to artificially inflated accuracy.

---

### 3. Mathematical Formulations & Derivations

#### The Data-Generating Process
We assume there exists a true, hidden function $h^{\*}(x)$ that generates the relationship in the real world. When we observe a data point, it is corrupted by zero-mean, independent and identically distributed (IID) Gaussian noise:
$$y = h^{\*}(x) + \epsilon, \quad \mathbb{E}[\epsilon] = 0, \quad \text{Var}(\epsilon) = \mathbb{E}[\epsilon^2] = \sigma^2$$

#### The Bias-Variance Decomposition
To analyze how a hypothesis $h_S$ trained on a random dataset $S$ generalizes, we evaluate the **expected test error** at a fixed test point $x$ over the joint randomness of the training set $S$ and the test noise $\epsilon$:
$$\text{Expected Test Error} = \mathbb{E}\_{S, \epsilon} \left[ (h_S(x) - y)^2 \right]$$

Substitute the definition of the target $y = h^{\*}(x) + \epsilon$ into the expectation:
$$\mathbb{E}\_{S, \epsilon} \left[ (h_S(x) - y)^2 \right] = \mathbb{E}\_{S, \epsilon} \left[ (h_S(x) - h^{\*}(x) - \epsilon)^2 \right]$$
Expanding this quadratic expression yields:
$$\mathbb{E}\_{S, \epsilon} \left[ (h_S(x) - h^{\*}(x))^2 - 2\epsilon(h_S(x) - h^{\*}(x)) + \epsilon^2 \right]$$
Because the test noise $\epsilon$ is independent of the training dataset $S$ and has a mean of zero ($\mathbb{E}[\epsilon] = 0$), the middle cross-term vanishes:
$$\mathbb{E}\_{S, \epsilon} \left[ -2\epsilon(h_S(x) - h^{\*}(x)) \right] = -2 \mathbb{E}[\epsilon] \cdot \mathbb{E}_S[h_S(x) - h^{\*}(x)] = 0$$
This simplifies the risk to:
$$\mathbb{E}\_{S, \epsilon} \left[ (h_S(x) - y)^2 \right] = \mathbb{E}_S \left[ (h_S(x) - h^{\*}(x))^2 \right] + \sigma^2$$
We now focus on the first term. Let $\bar{h}(x) = \mathbb{E}_S[h_S(x)]$ represent the **long-run average predictor** across all possible training set draws $S$ of a fixed size. We add and subtract $\bar{h}(x)$ inside the square:
$$\mathbb{E}_S \left[ (h_S(x) - h^{\*}(x))^2 \right] = \mathbb{E}_S \left[ \left( (h_S(x) - \bar{h}(x)) + (\bar{h}(x) - h^{\*}(x)) \right)^2 \right]$$
Expanding this expression gives:
$$\mathbb{E}_S \left[ (h_S(x) - \bar{h}(x))^2 + 2(h_S(x) - \bar{h}(x))(\bar{h}(x) - h^{\*}(x)) + (\bar{h}(x) - h^{\*}(x))^2 \right]$$
We take the expectation of each term with respect to $S$. Since $(\bar{h}(x) - h^{\*}(x))$ is a constant with respect to the random draw of $S$, the expectation of the middle term is:
$$\mathbb{E}_S \left[ 2(h_S(x) - \bar{h}(x))(\bar{h}(x) - h^{\*}(x)) \right] = 2(\bar{h}(x) - h^{\*}(x)) \cdot \mathbb{E}_S[h_S(x) - \bar{h}(x)]$$
By definition, $\mathbb{E}_S[h_S(x) - \bar{h}(x)] = \mathbb{E}_S[h_S(x)] - \bar{h}(x) = 0$. Thus, the cross-term is exactly zero. This leaves:
$$\mathbb{E}_S \left[ (h_S(x) - h^{\*}(x))^2 \right] = \mathbb{E}_S \left[ (h_S(x) - \bar{h}(x))^2 \right] + (\bar{h}(x) - h^{\*}(x))^2$$

Recombining all terms, we obtain the complete **Bias-Variance-Noise Decomposition**:
$$\mathbb{E}\_{S, \epsilon} \left[ (h_S(x) - y)^2 \right] = \sigma^2 + \text{Bias}(h_S(x))^2 + \text{Var}(h_S(x))$$
Where:
- **Unavoidable Noise**: $\sigma^2 = \mathbb{E}[\epsilon^2]$
- **Bias Squared**: $\text{Bias}(h_S(x))^2 = (\bar{h}(x) - h^{\*}(x))^2$
- **Variance**: $\text{Var}(h_S(x)) = \mathbb{E}_S \left[ (h_S(x) - \bar{h}(x))^2 \right]$

---

### 4. Step-by-Step Optimization & Regularization

#### Ridge Regression (L2 Regularization)
To prevent overfitting and reduce model variance, we add an $L_2$ regularization penalty (scaled by hyperparameter $\rho > 0$) to the Ordinary Least Squares (OLS) objective. Let $X \in \mathbb{R}^{n \times d}$ be the design matrix and $y \in \mathbb{R}^n$ be the label vector:
$$J(\theta) = \frac{1}{2} \|X\theta - y\|_2^2 + \frac{\rho}{2} \|\theta\|_2^2$$

#### Analytical Derivation of the Ridge Estimator
To find the optimal parameter vector $\theta$, we compute the gradient of $J(\theta)$ and set it to zero:
$$\nabla_\theta J(\theta) = X^T(X\theta - y) + \rho \theta = 0$$
$$(X^T X + \rho I)\theta = X^T y$$
$$\theta\_{\text{ridge}} = (X^T X + \rho I)^{-1} X^T y$$

#### Analysis of Numerical Stability and Variance Reduction
1. **The Singularity Problem**: In the underdetermined case where there are fewer training examples than features ($n < d$), the matrix $X^T X$ is singular (non-invertible) with a non-empty null space. Any vector from this null space can be added to $\theta$ without altering predictions, leading to infinite valid parameters and unstable, high-variance estimates.
2. **Eigenvalue Shifting**: Since $X^T X$ is symmetric and positive semidefinite, its eigenvalues are non-negative ($\lambda_i \ge 0$). Adding the term $\rho I$ shifts the eigenvalues of the system to:
   $$\lambda_i \to \lambda_i + \rho$$
   Since $\rho > 0$, the minimum eigenvalue of $(X^T X + \rho I)$ is strictly bounded away from zero ($\lambda\_{\text{min}} \ge \rho$).
3. **Variance Control**: The variance of the parameter estimator is directly scaled by the inverse of the eigenvalues. If $\lambda_i \approx 0$, small perturbations in the training data cause massive, unstable swings in the parameter updates. By replacing $\frac{1}{\lambda_i}$ with $\frac{1}{\lambda_i + \rho}$, Ridge Regression suppresses these extreme fluctuations, bringing stability back to the optimization path at the cost of a controlled, small amount of bias.

```
[Insert diagram: The Bias-Variance Trade-Off Curve. A plot showing Model Complexity (x-axis) vs. Error (y-axis). The training error decreases monotonically to 0. The test error is a U-shaped curve that reaches a local minimum at the sweet spot before rising in the classical overfitting regime. Red curve: Bias^2 (falling). Blue curve: Variance (rising). Black dotted curve: Expected Test Error (U-shape).]
```

```
[Insert diagram: The Double Descent Curve. An extension of the classical trade-off. To the left of the 'Interpolation Threshold' (where number of parameters d equals sample size n), we see the classical U-curve. Exactly at the threshold, variance explodes. To the right of the threshold (the overparameterized regime), the curve slopes downward again, showing that extremely large models generalize exceptionally well.]
```

---

### 5. Practical Workflows & Tuning Algorithms

#### A. $k$-fold Cross-Validation Workflow
When data is limited and we cannot afford a dedicated, large development set, $k$-fold cross-validation is used to evaluate model generalizations and select hyperparameters (such as $\rho$):
1. **Split**: Randomly partition the dataset $D$ into $k$ equal-sized, mutually exclusive folds: $\{F_1, F_2, \dots, F_k\}$.
2. **Iterate**: For each fold $i \in \{1, \dots, k\}$:
   - Treat fold $F_i$ as the validation/dev set.
   - Treat the union of the remaining $k-1$ folds as the training set $D \setminus F_i$.
   - Train the model configuration on the training set to obtain parameters $\theta^{(i)}$.
   - Calculate the loss (error) of $\theta^{(i)}$ on the validation fold $F_i$.
3. **Average**: Compute the mean validation score across all $k$ iterations:
   $$\text{CV Error} = \frac{1}{k} \sum\_{i=1}^k \text{Error}(F_i; \theta^{(i)})$$
4. **Select**: Run this process over a grid of candidate hyperparameters and select the one minimizing the CV Error. Once selected, train the final model on the entire dataset $D$.

#### B. The Hyperband Algorithm (Successive Halving)
Hyperband is a compute-efficient hyperparameter tuning algorithm that aggressively prunes poorly performing configurations early in the training process, reserving resources for promising candidates.

```
Hyperband Pseudocode (Simplified Successive Halving Round):
------------------------------------------------------------
Input: M (Set of randomized hyperparameter configurations, e.g., different values of rho)
       R (Initial compute budget, e.g., 5 epochs per configuration)

While size(M) > 1:
    1. Run all configurations in M for R steps of training
    2. Evaluate each configuration on the Dev/Validation set
    3. Sort configurations by Dev loss
    4. Prune the bottom 50% (the worst performing configurations)
    5. Double the step budget for the survivors: R := 2 * R
    6. M := Survivors (top 50%)

Return the single remaining configuration
```

---

### 6. Modern Twists & Deep Learning Advice

- **Why Double Descent Occurs**: In classical statistics, fitting a model past the interpolation threshold (where parameters $d >$ samples $n$) is considered a recipe for catastrophic variance. However, in modern deep neural networks, the optimization algorithms (like SGD) exhibit an **implicit bias**. Among the infinite possible parameter configurations that can fit the training data with zero error, the optimizer implicitly selects the "smoothest" or "minimum norm" solution. This implicit regularization ensures that overparameterized models do not fit to the high-frequency training noise, allowing them to generalize exceptionally well.
- **The ImageNet-V2 Lesson**: To test if decades of collective tuning on the ImageNet benchmark had caused "adaptive overfitting" (cheating by design), researchers spent years rebuilding the ImageNet test set from scratch (ImageNet-V2). They discovered that while absolute accuracy dropped by a constant ~11% across all models (due to shifts in modern camera technology and distributions), the **relative ranking of the models remained completely identical**. The best models on the old set were still the best models on the new set, proving that machine learning models learn robust, generalizable representations of the world rather than overfitting to specific public test set quirks.

---

### 7. Applications
- **Infrastructure Infrastructure Capital Expenditure (CapEx) Planning**: When choosing parameters for massive foundation models, hyperparameter tuning cannot be run naively due to cost. Standard tuning algorithms like Hyperband are used by systems engineers to run thousands of parallel training schedules for only a few steps, killing off poor configurations before wasting millions of dollars in compute credits.
- **Medical Diagnostics and Drug Trials**: In clinical statistics, strict trial registration and holdout policies are enforced to prevent $p$-hacking (adaptive overfitting to the validation cohort). Machine learning models deployed in healthcare use $k$-fold cross-validation and rigorous regularizers to ensure that diagnostic predictions are driven by stable, biological signals rather than singular training anomalies.

---

### 8. Reflection Questions
1. **The Role of the Dev Set**: Why is it mathematically and methodologically invalid to select the regularization parameter $\rho$ by choosing the value that minimizes training loss? What would happen to the selected value of $\rho$ if you did this?
2. **Double Descent vs. Classical Trade-Off**: Explain how a modern overparameterized neural network with 10 billion parameters can achieve perfect zero loss on a training set of 1 million images and still outperform a smaller model of 500,000 parameters on a test set. What is the optimizer doing implicitly to achieve this?
3. **The Robustness of ImageNet Rankings**: If a model experiences an 11% drop in accuracy when evaluated on ImageNet-V2, has it overfit to the original ImageNet? Why is the preservation of model rankings across both test sets a reassuring result for the machine learning community?

---

### 9. Further Reading & Resources
- **Rethinking Generalization**: The seminal paper by Benjamin Recht et al. that catalyzed the modern study of deep learning generalization and double descent.
- **ImageNet-V2 Robustness Literature**: Research papers detailing the reconstruction of the ImageNet test cohort and the mathematics of distribution shifts.
- **Trevor Hastie's Statistical Learning Books**: Recommended for mastering the mathematics of the Lasso path, Ridge Regression, and $k$-fold cross-validation theory.
