# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 10: Gaussian Mixture Models (EM) and Principal Component Analysis (PCA)

### 1. Summary
This lecture unifies the treatment of unsupervised learning by exploring two foundational paradigms: probabilistic latent variable modeling and non-probabilistic linear dimensionality reduction. Shifting from the ad-hoc soft clustering introduced in GMMs, the instructor formally derives the **Expectation-Maximization (EM) algorithm**. By introducing the **Evidence Lower Bound (ELBO)** through the "Q-distribution trick" and utilizing **Jensen's Inequality** on concave structures, EM is shown to be a mathematically rigorous coordinate ascent technique that guarantees monotonic convergence to a local maximum of the marginal log-likelihood. The lecture then transitions to **Principal Component Analysis (PCA)**, a non-probabilistic workhorse algorithm. The mathematical derivation proves that maximizing projected variance is equivalent to minimizing perpendicular reconstruction residuals. This optimization problem is solved analytically using the **eigenvalue decomposition** of the empirical covariance matrix, revealing critical systems/data realities such as feature scaling, mean centering, and spectrum-separation instabilities.

---

### 2. Key Concepts & Definitions
- **Latent (Hidden) Variables ($z$)**: Unobserved variables that represent hidden structure in the data (e.g., the true origin cluster or light source of a photon).
- **Evidence Lower Bound (ELBO)**: A surrogate lower-bound function, denoted as $\mathcal{L}(Q, \theta)$, constructed via Jensen's Inequality to under-approximate the marginal log-likelihood $l(\theta)$.
- **Strict Monotonicity**: A convergence property of the EM algorithm ensuring that the marginal log-likelihood $l(\theta^{(t)})$ strictly increases or stays constant at each iteration, preventing local cycle oscillations.
- **Lagrange Multipliers**: A mathematical optimization tool used in the M-step to enforce constraints on parameters, such as ensuring the mixture prior vector $\phi$ resides on the probability simplex ($\sum\_{j=1}^k \phi_j = 1$).
- **Principal Component**: An orthonormal vector direction $u$ that captures the maximum variance of the projected dataset.
- **Mean Centering**: The preprocessing step of shifting a dataset to have zero empirical mean ($\mu = 0$), ensuring that PCA eigenvectors represent directions of variance rather than directions pointing toward the data centroid.
- **Rescaling (z-score Standardization)**: Normalizing each feature by dividing by its empirical standard deviation, preventing features with large raw numerical scales from disproportionately dominating PCA variance calculations.
- **Separated Spectrum Requirement**: The condition that eigenvalues of the empirical covariance matrix must be distinctly separated ($\lambda_j \gg \lambda\_{j+1}$) for the corresponding principal directions to be stable and identifiable under minor data perturbations.

---

### 3. Mathematical Formulations & Derivations

#### A. Constructing the ELBO (The Q-Distribution Trick)
We are given a dataset $\mathcal{D} = \{x^{(1)}, \dots, x^{(n)}\}$ and a latent variable model $p(x, z; \theta)$. The marginal log-likelihood is notoriously difficult to optimize directly because the sum over the latent space resides inside the logarithm:
$$l(\theta) = \sum\_{i=1}^n \log p(x^{(i)}; \theta) = \sum\_{i=1}^n \log \sum\_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta)$$

To establish a local surrogate, we examine a single data point (dropping the superscript $i$ for notational convenience). Let $Q(z)$ be any arbitrary probability distribution over the latent variable space such that $Q(z) \ge 0$ and $\sum\_{z} Q(z) = 1$:
$$\log p(x; \theta) = \log \sum_z p(x, z; \theta)$$
$$\log p(x; \theta) = \log \sum_z Q(z) \frac{p(x, z; \theta)}{Q(z)}$$

We can interpret this summation as a mathematical expectation under the distribution $Q(z)$:
$$\log p(x; \theta) = \log \mathbb{E}\_{z \sim Q} \left[ \frac{p(x, z; \theta)}{Q(z)} \right]$$

Since the natural logarithm $\log(t)$ is a strictly concave function, we apply **Jensen's Inequality** ($\log \mathbb{E}[X] \ge \mathbb{E}[\log X]$) to pull the logarithm inside the expectation:
$$\log p(x; \theta) \ge \mathbb{E}\_{z \sim Q} \left[ \log \frac{p(x, z; \theta)}{Q(z)} \right]$$
$$\log p(x; \theta) \ge \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)} = \mathcal{L}(Q, \theta)$$

Summing across all $n$ data points yields the global Evidence Lower Bound (ELBO):
$$l(\theta) \ge \sum\_{i=1}^n \sum\_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}$$

#### B. Proving ELBO Tightness (The E-Step Solution)
To make the lower bound $\mathcal{L}(Q, \theta)$ perfectly tight at our current parameter estimate $\theta^{(t)}$, we must satisfy the condition under which Jensen's Inequality becomes an equality. This occurs when the random variable inside the expectation is a constant with respect to $z$:
$$\frac{p(x, z; \theta^{(t)})}{Q(z)} = c \implies p(x, z; \theta^{(t)}) = c \cdot Q(z)$$

Since $Q(z)$ is a probability distribution and must sum to 1, we sum both sides over $z$ to solve for the constant $c$:
$$\sum_z p(x, z; \theta^{(t)}) = c \sum_z Q(z) \implies p(x; \theta^{(t)}) = c$$

Substituting $c$ back into our constant-ratio equation yields:
$$Q(z) = \frac{p(x, z; \theta^{(t)})}{p(x; \theta^{(t)})} = p(z \mid x; \theta^{(t)})$$

Thus, the lower bound is perfectly tight (touching the true log-likelihood curve) when $Q(z)$ is set to the posterior distribution of the latent variable $z$ given the observed data $x$ and parameters $\theta^{(t)}$.

<div id="plotly-em-geometry" class="plotly-chart" aria-label="Interactive Plotly chart: the EM step geometry with tangent lower bound, slider to advance iterations"></div>

<p><em>Figure: The Geometry of the EM Step — the green curve is the multi-modal log-likelihood $l(\theta)$; the orange curve is the tangent lower bound $\mathcal{L}(Q_t, \theta)$, touching $l(\theta)$ exactly at $\theta^{(t)}$ (tightness). Maximizing the orange curve yields $\theta^{(t+1)}$, a strictly higher point on the green curve — monotonic progress. Drag the slider to climb.</em></p>

<div id="plotly-pca-projection" class="plotly-chart" aria-label="Interactive Plotly chart: PCA first principal component with a slider to rotate the projection direction and watch variance change"></div>

<p><em>Figure: PCA as Maximum-Variance Projection — the slider rotates a candidate direction $u$; the projected points (faded onto the direction line) spread widest exactly when $u$ aligns with the first eigenvector of the covariance matrix. The readout shows the projected variance $u^T \Sigma u$ at each angle.</em></p>

#### C. PCA Derivation: Maximizing Projected Variance
Let $\mathcal{D} = \{x^{(1)}, \dots, x^{(n)}\}$ be a preprocessed dataset with $x^{(i)} \in \mathbb{R}^d$ that has been mean-centered ($\sum_i x^{(i)} = 0$). We want to find a unit direction vector $u \in \mathbb{R}^d$ ($u^T u = 1$) such that the projected coordinates of the data points along $u$ have maximum empirical variance.

The projection length of a data point $x^{(i)}$ onto the unit vector $u$ is given by the inner product:
$$\text{proj}(x^{(i)}) = x^{(i)T} u$$

Since the dataset is mean-centered, the mean of the projected coordinates is also 0. The empirical variance of the projected data is:
$$\sigma\_{\text{proj}}^2 = \frac{1}{n} \sum\_{i=1}^n (x^{(i)T} u)^2 = \frac{1}{n} \sum\_{i=1}^n (u^T x^{(i)})(x^{(i)T} u) = u^T \left( \frac{1}{n} \sum\_{i=1}^n x^{(i)} x^{(i)T} \right) u$$

Recognizing the term in parentheses as the empirical covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$:
$$\sigma\_{\text{proj}}^2 = u^T \Sigma u$$

To maximize this projected variance subject to the unit-norm constraint on $u$, we formulate the Lagrangian:
$$\mathcal{L}(u, \lambda) = u^T \Sigma u - \lambda(u^T u - 1)$$

Taking the vector gradient with respect to $u$ and setting it to 0:
$$\nabla_u \mathcal{L}(u, \lambda) = 2\Sigma u - 2\lambda u = 0 \implies \Sigma u = \lambda u$$

This is the standard **eigenvalue equation**. Left-multiplying by $u^T$ reveals the objective value:
$$u^T \Sigma u = \lambda u^T u = \lambda$$

Therefore, to maximize the projected variance, $u$ must be the eigenvector of the empirical covariance matrix $\Sigma$ associated with the **largest eigenvalue** $\lambda_1$. Subsequent principal components are orthogonal eigenvectors associated with sorted decreasing eigenvalues $\lambda_2, \lambda_3, \dots$.

---

### 4. Step-by-Step Optimization Workflows

#### General Expectation-Maximization (EM) Loop
The general EM algorithm alternates between the E-step and M-step to maximize the log-likelihood of a latent variable model:
1. **Initialize** the parameters $\theta^{(0)}$ randomly or using heuristics.
2. **Loop until convergence**:
   - **E-Step (Expectation)**: For each training example $i \in \{1, \dots, n\}$, set $Q_i$ to the posterior distribution of the latent variables given the data and current parameters:
     $$Q_i^{(t)}(z^{(i)}) := p(z^{(i)} \mid x^{(i)}; \theta^{(t)})$$
   - **M-Step (Maximization)**: Compute the new optimal parameter set $\theta^{(t+1)}$ by maximizing the expectation of the joint log-likelihood over the $Q_distribution$:
     $$\theta^{(t+1)} := \arg \max\_{\theta} \sum\_{i=1}^n \sum\_{z^{(i)}} Q_i^{(t)}(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i^{(t)}(z^{(i)})}$$

---

### 5. Practical Implementation & Examples

#### A. Expectation-Maximization for Gaussian Mixture Models (GMM)
In GMMs, we assume $z^{(i)} \sim \text{Multinomial}(\phi)$ and $x^{(i)} \mid (z^{(i)}=j) \sim \mathcal{N}(\mu_j, \Sigma_j)$.

##### Pseudocode for GMM EM Update:
```python
import numpy as np

def gmm_em(X, K, max_iter=100, tol=1e-4):
    n, d = X.shape
    # Initialize parameters
    phi = np.ones(K) / K
    mus = X[np.random.choice(n, K, replace=False)]
    sigmas = np.array([np.eye(d) for _ in range(K)])
    w = np.zeros((n, K))
    
    for iteration in range(max_iter):
        # E-Step: Compute soft memberships (responsibilities)
        for j in range(K):
            diff = X - mus[j]
            inv_sigma = np.linalg.inv(sigmas[j])
            det_sigma = np.linalg.det(sigmas[j])
            exponent = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
            # Compute multivariate Gaussian density numerator
            w[:, j] = phi[j] * (1.0 / np.sqrt(((2*np.pi)**d) * det_sigma)) * np.exp(exponent)
            
        # Normalize across classes for soft-assignment w_ij
        sum_w = np.sum(w, axis=1, keepdims=True)
        w /= sum_w
        
        # M-Step: Update parameters
        sum_w_j = np.sum(w, axis=0)
        phi = sum_w_j / n  # Prior update
        
        for j in range(K):
            # Update mean vectors (weighted averages)
            mus[j] = np.sum(w[:, [j]] * X, axis=0) / sum_w_j[j]
            # Update covariance matrices (weighted outer products)
            diff = X - mus[j]
            sigmas[j] = (diff.T @ (w[:, [j]] * diff)) / sum_w_j[j]
            
    return phi, mus, sigmas
```

#### B. Dimensionality Reduction via PCA
```python
def pca_reduction(X, k_components):
    # 1. Mean Centering
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Rescaling (Standardization)
    X_scaled = X_centered / np.std(X_centered, axis=0)
    
    # 3. Compute Empirical Covariance Matrix
    covariance_matrix = np.cov(X_scaled, rowvar=False)
    
    # 4. Eigenvalue Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_vectors = eigenvectors[:, sorted_indices[:k_components]]
    
    # 5. Project Data into Lower-Dimensional Subspace
    X_reduced = X_scaled @ top_vectors
    return X_reduced, eigenvalues[sorted_indices]
```

---

### 6. Applications
- **Astroparticle Emission Profiling**: Disentangling overlapping emission distributions from distinct cosmic sources (e.g., pulsars, quasars, and background photons) without explicit ground-truth classifications.
- **High-Dimensional Data Denoising**: Filtering random high-frequency measurement noise by projecting data onto a low-dimensional principal subspace and reconstruction.
- **Linguistic Vector Compression**: Condensing large document-term sparse vectors down to dense, latent semantic representations (e.g., Latent Semantic Analysis / Word2Vec pre-processing steps).

---

### 7. Reflection Questions
1. **The Role of Non-Negativity in Jensen's**: Why is it mathematically required that the surrogate distribution $Q(z)$ is non-negative ($Q(z) \ge 0$) and sums to 1 ($\sum Q(z) = 1$) for Jensen's Inequality to hold? How does this restrict the function spaces EM can explore?
2. **Instability of the Eigen-Basis**: If a dataset has an empirical covariance matrix where the top two eigenvalues are equal ($\lambda_1 = \lambda_2$), what does this imply about the uniqueness of the first principal component vector $u_1$? Why can this cause downstream linear classifiers to behave inconsistently across slightly perturbed datasets?
3. **Information Loss and the Covariance Trace**: How does computing the trace of the empirical covariance matrix $\Sigma$ allow us to calculate the exact percentage of total dataset variance preserved when projecting data down to $k$ components?

---

### 8. Further Reading & Resources
- **CS229 Friday Discussion Sections**: Highly recommended by the course staff for reviewing the underlying spectral theorem, matrix ranks, and Lagrangian formulations of constrained optimization.
- **Scikit-Learn Decomposition Module**: Documentation on the numerical scaling and convergence differences between full PCA, Randomized (thin) PCA, and GMM solvers in Python.
- **EM Convergence and Non-Convex Optimization Literature**: Classic papers analyzing the rate of convergence of EM compared to first-order gradient descent on latent variable loss landscapes.
