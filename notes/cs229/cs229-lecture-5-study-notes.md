# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 5: Gaussian Discriminant Analysis

### 1. Summary
This lecture introduces the concept of **Generative Learning Algorithms**, marking a major paradigm shift from the discriminative classifiers studied in previous lectures. While discriminative algorithms like linear and logistic regression directly model the posterior probability $P(y|x)$ of a label given the inputs, generative algorithms model the joint probability $P(x, y)$ by learning the class-conditional feature distribution $P(x|y)$ and the class prior $P(y)$. The instructor, Chris, contrasts these two paradigms using **Gaussian Discriminant Analysis (GDA)** for continuous features and **Naive Bayes** for discrete text data. He highlights how generative modeling allows us to estimate optimal model parameters analytically in closed form without needing iterative gradient-based optimization like Stochastic Gradient Descent. The lecture concludes by exploring the theoretical connections between GDA and logistic regression and discussing how the generative philosophy underpins modern AI breakthroughs, including Diffusion Models and Large Language Models.

---

### 2. Key Concepts & Definitions
- **Discriminative Algorithms**: Classifiers that learn a direct mapping or decision boundary from the input space $\mathcal{X}$ to the label space $\mathcal{Y}$ to estimate $P(y|x)$.
- **Generative Algorithms**: Classifiers that model how the data is generated for each class, estimating $P(x|y)$ and $P(y)$. Prediction is performed by inverting this forward model using Bayes' rule to compute $P(y|x)$.
- **Multivariate Gaussian Distribution**: A multidimensional generalization of the normal distribution for a vector-valued random variable $x \in \mathbb{R}^d$. It is fully parameterized by a mean vector $\mu \in \mathbb{R}^d$ and a symmetric, positive definite covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$.
- **Covariance Matrix ($\Sigma$)**: A symmetric matrix representing the pairwise covariances between features. Being positive definite, its eigenvalues are strictly positive, controlling the spread, shape, and orientation of the elliptical probability contours.
- **Precision Matrix ($\Sigma^{-1}$)**: The inverse of the covariance matrix. In graphical models, a zero entry in the precision matrix ($\Sigma^{-1}_{ij} = 0$) reveals that feature $i$ and feature $j$ are conditionally independent given all other features.
- **Gaussian Discriminant Analysis (GDA)**: A generative model for continuous features where the class-conditional distributions $P(x|y)$ are modeled as multivariate Gaussians that share a single, identical covariance matrix $\Sigma$, but have different class-specific mean vectors $\mu_0$ and $\mu_1$.
- **Quadratic Discriminant Analysis (QDA)**: A generalized version of GDA where the assumption of a shared covariance matrix is relaxed, allowing each class to have its own covariance matrix ($\Sigma_0 \neq \Sigma_1$). This results in a quadratic decision boundary instead of a linear one.
- **Naive Bayes Classifier**: A generative model designed for discrete feature vectors (such as word occurrences in document classification). It relies on the **Naive Bayes Assumption**: that features are conditionally independent of one another given the class label $y$.
- **Laplace Smoothing (Additive Smoothing)**: A technique that adds a small pseudo-count (typically $+1$) to feature occurrences during parameter estimation. This prevents the model from predicting zero probabilities for unseen features, which would otherwise zero out the entire product of probabilities.

---

### 3. Mathematical Formulations & Derivations

#### Multivariate Gaussian Probability Density Function (PDF)
For a $d$-dimensional random variable $x \in \mathbb{R}^d$:
$$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$
where $|\Sigma|$ denotes the determinant of the covariance matrix and $(x - \mu)^T \Sigma^{-1} (x - \mu)$ acts as a Mahalanobis distance metric.

#### The GDA Probabilistic Setup
For binary classification where $y \in \{0, 1\}$ and features $x \in \mathbb{R}^d$ are continuous:
1. **Class Prior ($P(y)$)**:
   $$P(y) = \phi^y (1 - \phi)^{1 - y}$$
2. **Class-Conditional Densities ($P(x|y)$)**:
   $$P(x|y=0) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right)$$
   $$P(x|y=1) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right)$$

#### Closed-Form GDA Parameter Estimation (Joint MLE)
Unlike discriminative models that maximize the conditional likelihood $P(y|x)$, generative models maximize the **joint log-likelihood** of the training data:
$$\ell(\phi, \mu_0, \mu_1, \Sigma) = \ln \prod_{i=1}^n P(x^{(i)}, y^{(i)}) = \sum_{i=1}^n \left( \ln P(x^{(i)} | y^{(i)}) + \ln P(y^{(i)}) \right)$$

By setting the partial derivatives of the joint log-likelihood to zero, we obtain closed-form estimators:
1. **Prior ($\phi$)**:
   $$\phi = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(y^{(i)} = 1)$$
2. **Class Means ($\mu_0, \mu_1$)**:
   $$\mu_0 = \frac{\sum_{i=1}^n \mathbb{I}(y^{(i)} = 0) x^{(i)}}{\sum_{i=1}^n \mathbb{I}(y^{(i)} = 0)}$$
   $$\mu_1 = \frac{\sum_{i=1}^n \mathbb{I}(y^{(i)} = 1) x^{(i)}}{\sum_{i=1}^n \mathbb{I}(y^{(i)} = 1)}$$
3. **Shared Covariance Matrix ($\Sigma$)**:
   $$\Sigma = \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}}) (x^{(i)} - \mu_{y^{(i)}})^T$$

---

### 4. Step-by-Step Explanations & Algorithmic Workflows

#### A. Derivation of the GDA Linear Decision Boundary
To prove that GDA with shared covariance results in a linear decision boundary, we find the surface of equal probability where the log-odds ratio equals zero:
$$\ln \frac{P(y=1|x)}{P(y=0|x)} = \ln \frac{P(x|y=1)P(y=1)}{P(x|y=0)P(y=0)} = 0$$

Substituting the Gaussian PDFs and priors:
$$\ln \left( \frac{\exp \left( -\frac{1}{2} (x-\mu_1)^T \Sigma^{-1} (x-\mu_1) \right) \cdot \phi}{\exp \left( -\frac{1}{2} (x-\mu_0)^T \Sigma^{-1} (x-\mu_0) \right) \cdot (1-\phi)} \right) = 0$$

Expanding this expression:
$$-\frac{1}{2}(x-\mu_1)^T \Sigma^{-1} (x-\mu_1) + \ln \phi + \frac{1}{2}(x-\mu_0)^T \Sigma^{-1} (x-\mu_0) - \ln(1-\phi) = 0$$

Expanding the quadratic terms reveals:
$$(x-\mu)^T \Sigma^{-1} (x-\mu) = x^T \Sigma^{-1} x - 2 \mu^T \Sigma^{-1} x + \mu^T \Sigma^{-1} \mu$$

Substitute these expansions back into the equation:
$$-\frac{1}{2} \left[ x^T \Sigma^{-1} x - 2 \mu_1^T \Sigma^{-1} x + \mu_1^T \Sigma^{-1} \mu_1 \right] + \frac{1}{2} \left[ x^T \Sigma^{-1} x - 2 \mu_0^T \Sigma^{-1} x + \mu_0^T \Sigma^{-1} \mu_0 \right] + \ln \frac{\phi}{1-\phi} = 0$$

The quadratic term $x^T \Sigma^{-1} x$ cancels out completely from both sides because the covariance matrix $\Sigma$ is shared. Simplifying the remaining linear and constant terms yields:
$$(\mu_1 - \mu_0)^T \Sigma^{-1} x - \frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 + \frac{1}{2} \mu_0^T \Sigma^{-1} \mu_0 + \ln \frac{\phi}{1-\phi} = 0$$

This can be written in the standard linear/affine form:
$$\theta^T x + \theta_0 = 0$$
where:
$$\theta = \Sigma^{-1} (\mu_1 - \mu_0)$$
$$\theta_0 = -\frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 + \frac{1}{2} \mu_0^T \Sigma^{-1} \mu_0 + \ln \frac{\phi}{1-\phi}$$

<div id="plotly-gda-vs-qda" class="plotly-chart" aria-label="Interactive Plotly chart: GDA versus QDA decision boundaries with Gaussian class contours"></div>

<p><em>Figure: GDA vs. QDA Decision Boundaries — with a shared covariance matrix (GDA), the quadratic terms cancel and the boundary is a straight line. Relaxing to per-class covariances (QDA) produces curved, quadratic boundaries that can wrap around elliptical class distributions.</em></p>

#### B. Naive Bayes Classification with Laplace Smoothing
For a text document represented as a binary feature vector $x \in \{0, 1\}^d$ over a vocabulary of size $d$:
1. **The Naive Bayes Assumption**:
   $$P(x_1, \dots, x_d | y) = \prod_{j=1}^d P(x_j | y)$$
2. **Prior Estimation**:
   $$\phi_y = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(y^{(i)} = 1)$$
3. **Conditional Parameter Estimation (with Laplace $+1$ Smoothing)**:
   $$\phi_{j|y=1} = P(x_j = 1 | y=1) = \frac{\sum_{i=1}^n \mathbb{I}(x_j^{(i)} = 1 \land y^{(i)} = 1) + 1}{\sum_{i=1}^n \mathbb{I}(y^{(i)} = 1) + 2}$$
   $$\phi_{j|y=0} = P(x_j = 1 | y=0) = \frac{\sum_{i=1}^n \mathbb{I}(x_j^{(i)} = 1 \land y^{(i)} = 0) + 1}{\sum_{i=1}^n \mathbb{I}(y^{(i)} = 0) + 2}$$
   *(Note: The $+2$ in the denominator accounts for the two possible binary values of $x_j \in \{0, 1\}$.)*
4. **Prediction**:
   $$\hat{y} = \arg\max_{y \in \{0,1\}} P(y) \prod_{j=1}^d P(x_j | y)$$

---

### 5. GDA vs. Logistic Regression: Structural Trade-offs

A highly elegant relationship connects generative and discriminative modeling. If $P(x|y)$ is modeled as multivariate Gaussian with shared covariance (GDA), the resulting posterior distribution $P(y=1|x)$ takes the exact form of a logistic (sigmoidal) function:
$$P(y=1|x) = \frac{1}{1 + \exp(-\theta^T x - \theta_0)}$$

This is also true if $P(x|y)$ comes from other distributions in the **Exponential Family** (e.g., Poisson or Multinomial) with shared parameters. However, the reverse is not true. This leads to key trade-offs:

| Metric / Dimension | Gaussian Discriminant Analysis (GDA) | Logistic Regression |
| :--- | :--- | :--- |
| **Model Nature** | Generative ($P(x, y)$) | Discriminative ($P(y \vert x)$) |
| **Assumption Strength** | **Stronger**: Assumes features are multivariate Gaussian within each class. | **Weaker**: Only assumes that the log-odds ratio $\ln \frac{P(y=1 \vert x)}{P(y=0 \vert x)}$ is linear. |
| **Data Efficiency** | **Highly Efficient**: Converges to its asymptotic limit faster with less training data *if* the Gaussian assumption is correct. | **Less Efficient**: Requires more training data to fit parameters robustly. |
| **Robustness** | **Fragile**: Performs poorly if the underlying data is strongly non-Gaussian (e.g., highly skewed or multi-modal). | **Robust**: Less sensitive to deviations from distributional assumptions since it does not model the features. |
| **Computation** | Closed-form analytic solution (extremely cheap, no iteration). | Iterative optimization needed (Stochastic Gradient Descent). |

---

### 6. Applications

- **Spam Filtering**: The classic use case for Naive Bayes. By treating an email as a bag of words or a binary indicator vector, the classifier identifies high-signal terms (like "buy" or "free") to route messages to spam folder.
- **Medical Diagnostics**: Modeling continuous health features (e.g., blood pressure, heart rate, hormone levels) for positive and negative disease classes as multivariate Gaussians (GDA/QDA) to classify patient health.
- **Pre-training and Generative AI Backbones**: Modern Large Language Models (LLMs) and Diffusion Models are conceptually generative, learning the underlying representation of unlabeled datasets. Diffusion models, for example, work by starting with Gaussian noise and iteratively reversing the noise addition process to generate high-resolution images.

---

### 7. Reflection Questions
1. **Shared vs. Independent Covariance (GDA vs. QDA)**: Derive why relaxing the shared covariance assumption in GDA—allowing $\Sigma_0 \neq \Sigma_1$—prevents the $x^T \Sigma^{-1} x$ terms from canceling out. What mathematical shape does the decision boundary take as a result?
2. **The Fragility of Generative Assumptions**: If you train a GDA model on a dataset where one of the features is binary (e.g., whether a house has a garage), why does the multivariate Gaussian assumption fail? How does GDA's performance compare to Logistic Regression in this scenario?
3. **Laplace Smoothing Scaling**: In multinomial Naive Bayes (modeling raw word counts instead of binary flags), if the vocabulary size is $V$ and we apply Laplace smoothing, we add $+1$ to each word count. Why must the denominator be adjusted by adding $+V$ instead of $+2$? How does this maintain a valid probability distribution?

---

### 8. Further Reading & Resources
- **Stanford CS229 Course Notes (Tenyu's Notes)**: The ultimate source of truth for the rigorous mathematical proofs of GDA and QDA parameter derivation.
- **Clifford-Hammersley Theorem**: Explores the deep connection between the zero patterns in precision matrices ($\Sigma^{-1}$) and conditional independence in graphical models.
- **Data Mixing & Pre-training Literature**: Papers exploring why pre-training models generatively on massive unlabeled datasets yields richer representations than direct discriminative training.
