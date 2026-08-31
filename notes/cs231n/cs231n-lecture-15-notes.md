# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 15: 3D Vision

#### 0. Quick-Recall Summary
*   **Dimensional Inverse Problem:** 3D vision seeks to invert the mathematically ill-posed 2D perspective projection $3\text{D} \rightarrow 2\text{D}$ to recover structural, geometric, and appearance properties of the physical world.
*   **Explicit vs. Implicit Representations:** Explicit representations (point clouds, meshes, parametric splines) define coordinates directly, facilitating easy sampling but making inside/outside spatial queries hard. Implicit representations (signed distance functions, occupancy fields) define geometry as level sets, making spatial queries trivial but surface point sampling difficult.
*   **PointNet Symmetric Axiom:** To process unordered point clouds, PointNet enforces permutation invariance by projecting points independently into high-dimensional space via an MLP, followed by a symmetric pooling operator (e.g., max pool).
*   **Differentiable Volume Rendering:** NeRF (Neural Radiance Fields) optimizes continuous 5D scenes (3D coordinates + 2D viewing directions) using 2D images by integrating colors and densities along camera rays using differentiable quadrature formulations.
*   **3D Gaussian Splatting:** Bypasses NeRF's computationally expensive neural coordinate query loop by representing scenes as a collection of sparse, anisotropic 3D Gaussian ellipsoids, rasterizing them directly to 2D at real-time speeds (150+ FPS).

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master the representation, processing, and generation of 3D objects and scenes. The lecture traces the historical progression from early volumetric voxel-grid convolutions to native point cloud operators, continuous implicit fields, and modern real-time visual-appearance splatting.
*   **Lecture Category:** (a) Mathematical Foundations, (b) Architecture Design, and (d) Specific Vision Task (3D Reconstruction/Generation).
*   **Builds on:** Lecture 5 (2D Convolutions) and Lecture 14 (Generative Models), extending standard template-matching and likelihood optimizations from $C \times H \times W$ spatial grids to $C \times T \times H \times W$ volumes and unstructured, non-parametric 3D manifolds.

---

#### 2. Mathematical Foundations

##### Representational Formalisms
*   **Point Clouds (Explicit):** Represented as a non-parametric matrix of coordinates:
    $$P = \begin{bmatrix} x_1 & x_2 & \dots & x_N \\ y_1 & y_2 & \dots & y_N \\ z_1 & z_2 & \dots & z_N \end{bmatrix} \in \mathbb{R}^{3 \times N}$$
    Optionally augmented with surface normal vectors $N \in \mathbb{R}^{3 \times N}$ to calculate lighting-surface interactions.
*   **Implicit Level-Set Surfaces:** Defined as the zero-isocontour of a continuous scalar function $f: \mathbb{R}^3 \rightarrow \mathbb{R}$:
    $$\mathcal{S} = \{ \mathbf{x} \in \mathbb{R}^3 \mid f(\mathbf{x}) = 0 \}$$
    where $f(\mathbf{x}) < 0$ denotes the interior volume, and $f(\mathbf{x}) > 0$ represents the exterior space.
*   **Parametric Curves and Spheres:** Mapping low-dimensional parameters directly to 3D coordinate space. For example, a unit sphere parameterized by azimuth $u \in [0, 2\pi]$ and elevation $v \in [0, \pi]$:
    $$f(u, v) = \begin{bmatrix} \cos(u)\sin(v) \\ \sin(u)\sin(v) \\ \cos(v) \end{bmatrix} \in \mathbb{R}^3$$
    This explicit formulation simplifies surface point generation via forward sampling.

##### PointNet Formulation
To construct a function $F$ over an unordered point set $\{p_1, \dots, p_N\}$ that is mathematically invariant to the permutation of its inputs:
$$F(\{p_1, \dots, p_N\}) \approx g\left( h(p_1), h(p_2), \dots, h(p_N) \right)$$
where $h: \mathbb{R}^3 \rightarrow \mathbb{R}^D$ is a multi-layer perceptron (MLP) mapping individual points to high-dimensional embedding spaces, and $g: \mathbb{R}^D \times \dots \times \mathbb{R}^D \rightarrow \mathbb{R}^K$ is a symmetric pooling function (e.g., element-wise $\max$ or $\sum$) that is invariant to input ordering.

##### Point Cloud Reconstruction Losses
To compute backpropagatable reconstruction loss between a generated point cloud $S_1$ and a ground-truth cloud $S_2$:
*   **Chamfer Distance:** Measures the average nearest-neighbor squared $L_2$ distance symmetrically between both point sets:
    $$d\_{\text{CD}}(S_1, S_2) = \frac{1}{|S_1|} \sum\_{x \in S_1} \min\_{y \in S_2} \|x - y\|_2^2 + \frac{1}{|S_2|} \sum\_{y \in S_2} \min\_{x \in S_1} \|x - y\|_2^2$$
    This is computationally efficient $O(|S_1| \log |S_2|)$ but can be sensitive to uneven cluster densities.
*   **Earth Mover's Distance (EMD):** Solves the optimal transport problem by finding a strict one-to-one bijection $\phi: S_1 \rightarrow S_2$ (where $|S_1| = |S_2|$):
    $$d\_{\text{EMD}}(S_1, S_2) = \min\_{\phi: S_1 \rightarrow S_2} \sum\_{x \in S_1} \|x - \phi(x)\|_2$$
    This yields cleaner structural alignments but requires solving a costly Hungarian matching algorithm.

##### Neural Radiance Fields (NeRF)
NeRF represents a continuous 3D scene as a 5D function $f_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$ where $\mathbf{x} = (x, y, z)$ is the 3D position, $\mathbf{d} = (\theta, \phi)$ is the camera viewing direction, $\mathbf{c} = (r, g, b)$ is the emitted radiance (color), and $\sigma \in [0, \infty)$ is the volume density.
The color $C(\mathbf{r})$ of a pixel corresponding to camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ integrated from near bound $t_n$ to far bound $t_f$ is:
$$C(\mathbf{r}) = \int\_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt$$
where $T(t) = \exp\left(-\int\_{t_n}^t \sigma(\mathbf{r}(s)) ds\right)$ represents the accumulated transmittance along the ray.
Using numerical quadrature with $N$ stratified samples along the ray, the differentiable color approximation is:
$$\hat{C}(\mathbf{r}) = \sum\_{i=1}^N T_i \left( 1 - \exp\left(-\sigma_i \delta_i\right) \right) \mathbf{c}_i$$
where $T_i = \exp\left(-\sum\_{j=1}^{i-1} \sigma_j \delta_j\right)$ and $\delta_i = t\_{i+1} - t_i$ is the distance between adjacent sample points.

##### 3D Gaussian Splatting Covariance Formulation
Each 3D Gaussian is parameterized by a mean position $\mu \in \mathbb{R}^3$ and a 3D covariance matrix $\Sigma \in \mathbb{R}^{3 \times 3}$:
$$G(\mathbf{x}) = \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)$$
To ensure $\Sigma$ remains positive semi-definite during gradient descent, it is sharded and optimized via a scaling matrix $S$ and rotation matrix $R$:
$$\Sigma = R S S^T R^T$$
where $S = \text{diag}(s_x, s_y, s_z)$ and $R$ is represented by normalized quaternions.

---

#### 3. Architecture / Algorithm Walkthrough

The visual processing pipelines transition from explicit volumetric grids to coordinates and splatted ellipsoids:

```
Volumetric 3D CNN Pipeline:
[Voxel Input (V x V x V)] ──> [3D Convolutions] ──> [Latent Features] ──> [3D Deconvolutions] ──> [Reconstructed Voxels]

PointNet Permutation Invariance:
[Point Cloud (3 x N)] ──> [Shared MLPs (Point-wise h)] ──> [Embeddings (D x N)] ──> [Symmetric Max Pool (g)] ──> [Global Feature (D)]

Differentiable Volume Rendering (NeRF):
[Ray Casting r(t)] ──> [Stratified Sampling (t_i)] ──> [Query MLP f_theta] ──> [Density/Color (sigma_i, c_i)] ──> [Differentiable Quadrature Integration] ──> [Rendered Pixel Color C(r)]
```

##### PyTorch Blueprint (PointNet Classifier and Custom Chamfer Loss)
This blueprint implements a PointNet architecture incorporating point-wise shared MLPs, a symmetric global pooling layer, and a vectorized, differentiable Chamfer distance loss function for reconstruction:

```python
import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    """
    Illustrative implementation of the PointNet encoder mapping unordered 
    point sets to permutation-invariant global shape descriptors.
    """
    def __init__(self, embedding_dim=1024):
        super(PointNetEncoder, self).__init__()
        # Shared MLPs implemented as 1D convolutions with kernel size 1
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, embedding_dim, kernel_size=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input shape: [Batch, 3, N_points]
        x = self.mlp1(x)
        x = self.mlp2(x)
        
        # Symmetric global aggregation function: Max pooling across points
        global_features, _ = torch.max(x, dim=2, keepdim=False)
        # Output shape: [Batch, embedding_dim]
        return global_features

class ChamferDistanceLoss(nn.Module):
    """
    Differentiable Chamfer Distance Loss for explicit point cloud reconstruction.
    """
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()

    def forward(self, S1, S2):
        # S1 shape: [Batch, N, 3], S2 shape: [Batch, M, 3]
        # Calculate pairwise L2 squared distances: shape [Batch, N, M]
        r1 = torch.sum(S1 ** 2, dim=2, keepdim=True) # [B, N, 1]
        r2 = torch.sum(S2 ** 2, dim=2, keepdim=True).transpose(1, 2) # [B, 1, M]
        pairwise_dist = r1 - 2 * torch.bmm(S1, S2.transpose(1, 2)) + r2
        
        # Min distances along both sets
        min_dist_S1_to_S2, _ = torch.min(pairwise_dist, dim=2) # [B, N]
        min_dist_S2_to_S1, _ = torch.min(pairwise_dist, dim=1) # [B, M]
        
        # Average distances symmetrically
        loss = torch.mean(min_dist_S1_to_S2) + torch.mean(min_dist_S2_to_S1)
        return loss
```

---

#### 4. Visual Intuition & Interpretability

The lecture contrasts the rendering qualities and coordinate properties of the various representations:

##### Voxel Reconstructions
*   **Voxel Grid Quantization:** Resolving shape boundaries on 3D matrices results in blocky, low-resolution "staircase" structures. If a resolution is increased to $256^3$, memory overhead scales cubically $O(N^3)$, rapidly exceeding GPU capacity limits.

##### Point Clouds vs. Meshes
*   **Uneven Sampling Bias:** Raw scanners sample shapes non-uniformly, concentrating millions of points on forward-facing regions while leaving rear shadows or thin components under-sampled. This lacks clean topological boundaries.
*   **Surface Disconnectivity:** Unlike connected polygonal faces that yield smooth reflection gradients under Phong shading, point clouds lack surface connectivity, leading to sparse "speckled" visualizations containing internal background leakage.

##### Implicit Field Transformations
*   **Tug-of-War Slicing (AtlasNet):** Visualizes AtlasNet deforming 2D squares into 3D. It mimics folding a piece of paper, where a network struggles to generate closed manifolds without tearing or creating self-intersecting artifacts.
*   **Implicit Voxel Transitions:** Visualizing the transition from continuous implicit level-set fields to explicit binarized voxel grids, highlighting how positive/negative sign shifts identify exact geometric zero-boundaries.

##### NeRF vs. Gaussian Splatting Rendering
*   **Volume Ray Traversal:** NeRF ray tracing samples hundreds of points per ray, executing computationally heavy MLP evaluations across millions of pixels. This behaves like a dense visual fog where transmittance $T(t)$ gradually drops as density increases.
*   **Sparse Splatting Ellipsoids:** Gaussian splatting visualizes the scene as transparent anisotropic ellipsoids. It bypasses ray-marching entirely by splatting means and projected covariances directly to the camera plane. This isolates empty space and concentrates compute on physical interfaces.

---


<div id="plotly-cs231n-15-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 15 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)
To compare 3D scene representations, we propose a conceptual **Differentiable Representation and Receptive Query Analyzer**:

*   **Visualization Type:** Multi-Panel Interactive 3D Voxel and Ray-Marching Simulator.
*   **Data Fields & Encoding:**
    *   **3D Viewport:** Renders the target 3D model (e.g., Stanford Bunny) using three interchangeable formats: (1) $64^3$ Voxel Grid, (2) Point Cloud, or (3) Continuous NeRF Ray Slice.
    *   **Ray-Slicing Heatmap:** Maps sample point density ($\sigma_i$) along a cast camera ray to a 1D gradient bar (Color: Blue to Red = low to high opacity; Size: scaled by ray transmittance $T_i$).
*   **Interactive Controls:**
    *   **Representation Selector:** Dropdown to switch between Voxel, Point Cloud, Mesh, NeRF, and 3D Gaussian Splatting.
    *   **Ray Cast Angle Slider:** Rotates camera views. In real-time, the viewport traces active rays, highlighting intersection queries on the model and computing the integrated quadrature summation in an adjacent formula panel.
    *   **Grid Resolution Slider ($32^3 \rightarrow 256^3$):** Morphs the voxel model, dynamically plotting memory utilization cubically $O(N^3)$ vs. Octree hierarchical partitioning.

---

#### 6. Empirical Heuristics & Benchmark Results
*   **The 3D Data Scaling Gap:** Modern 3D core datasets are orders of magnitude smaller than 2D counterparts:
    *   *ShapeNet Core:* **50,000 synthetic CAD models** across **55 categories**.
    *   *ScanNet:* **1,500 reconstructed indoor rooms** (ScanNet++ scales to 2,000-3,000 rooms).
    *   *Co-Evolving 2D Baselines:* Multi-view 2D pre-training models leverage foundations like LAION-5B (**5 billion images**) to bridge the 3D data scarcity bottleneck.
*   **Rendering Speed Milestones:**
    *   *Vanilla NeRF:* Render time of **$\approx 20$ seconds per frame** due to dense $N$-point MLP query integration.
    *   *3D Gaussian Splatting:* Render time of **$150+$ FPS** (Real-Time) by replacing ray sampling with rasterized ellipsoids.
*   **Volumetric Classifiers:**
    *   *Princeton ModelNet (2015 Baseline):* Volumetric deep belief nets running $30^3$ voxels yielded basic shape classification.
    *   *Multi-View CNNs (2015):* Simply rendering 3D shapes from multiple 2D camera views and passing them to ImageNet pre-trained CNNs consistently outperformed volumetric models due to rich 2D image prior features.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas
*   **Voxel Memory Volatility:** Cubically scaling voxel resolutions to preserve fine boundaries (e.g. $128^3 \rightarrow 256^3$) immediately triggers Out-of-Memory (OOM) errors on standard GPUs.
*   **Point Cloud Permutation Mismatch:** Evaluating point clouds as sequential tensors (e.g., using LSTMs or standard linear layers) results in catastrophic representation drift when the index ordering is shuffled.
*   **Empty Space Query Wastage:** Naive Ray-Marching in NeRF wastes over $90\%$ of MLP evaluations querying empty space or dense interiors. Solid bounds must be tracked via voxel coarse-grids or Gaussian density anchors to maintain efficiency.

##### Graduate-Level Reflection Questions
1.  **PointNet Symmetry Collapse:** PointNet achieves permutation invariance by applying a symmetric max-pooling function $g$ over high-dimensional point embeddings $h(p_i)$. If $h$ lacks sufficient capacity or coordinate resolution, what mathematical failure occurs on the boundaries of $g$, and how does this affect PointNet's sensitivity to sampling density changes?
2.  **Explicit-Implicit Sampling Trade-off:** Explicit 3D representations like AtlasNet simplify surface sampling but struggle with inside/outside queries. Implicit functions like DeepSDF exhibit the inverse profile. Prove how NeRF utilizes differentiable volume rendering to bypass the implicit surface sampling constraint, and analyze why this relies on dense stratified ray sampling.
3.  **NeRF vs. Gaussian Splatting Coordinate Grids:** Vanilla NeRF encodes a scene globally inside MLP weights $\theta$. In contrast, 3D Gaussian Splatting assigns scene coordinates to explicit spatial ellipsoids. Analyze how this difference in spatial parameterization affects: (a) scene editing/compositionality, (b) rendering throughput, and (c) model size/storage tradeoffs.
