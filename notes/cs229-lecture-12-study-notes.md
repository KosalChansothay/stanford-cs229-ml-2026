# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 12: Representation Learning & Foundation Models

### 1. Summary
This lecture details a fundamental shift in the machine learning paradigm, moving from training isolated models for specific tasks to pre-training general-purpose **Foundation Models** on massive, unlabeled datasets. The instructor, Tenyu, introduces the concept of **Representation Learning**, where a model learns to embed high-dimensional, complex data into structured, semantic vector spaces. The lecture explores multiple adaptation pathways for applying pre-trained representations to downstream tasks, including **Linear Probing**, **Fine-Tuning**, and the combined state-of-the-art approach **LP-FT**. Finally, the session dives deep into the mathematical, computational, and systems engineering realities of **Low-Rank Adaptation (LoRA)**, highlighting how low-rank parameterized updates enable massive memory optimization and solve the hardware multi-tenancy bottleneck for hosting custom models in production.

---

### 2. Key Concepts & Definitions
- **Foundation Model**: A term coined by Stanford researchers (led by Percy Liang in 2020/2021) to describe a general-purpose model pre-trained on massive, highly diverse, and unlabeled datasets that can be adapted to an unlimited number of downstream tasks.
- **Representation ($\phi\_\theta(x)$)**: A structured feature vector (also called an embedding) in an $m$-dimensional space produced by passing a raw input $x$ through a neural network parameterized by $\theta$.
- **Linear Probing (LP)**: An adaptation technique where the representation network parameters $\theta$ are frozen, and a simple linear model (or "head") is trained on top of the fixed embeddings.
- **Fine-Tuning (FT)**: An adaptation technique where both the representation parameters $\theta$ (initialized to pre-trained weights) and the downstream head are optimized jointly.
- **Destructive Gradient Backpropagation**: The phenomenon in early fine-tuning where backpropagating gradients from a randomly initialized downstream head can corrupt or destroy the delicate, pre-trained semantic structures within the representation parameters.
- **LP-FT**: A two-phase adaptation procedure that first performs linear probing to align the head with the representation, and then fine-tunes both jointly, mitigating the destructive update effect.
- **Low-Rank Adaptation (LoRA)**: A parameter-efficient fine-tuning (PEFT) method that freezes the pre-trained weight matrices and parameterizes their updates as the product of two low-rank matrices.
- **Multi-Tenancy Bottleneck**: The systems challenge of serving customized models to millions of distinct users. Standard fine-tuning requires hosting a separate multi-gigabyte model per user; LoRA solves this by sharing a single base model and swapping tiny user-specific low-rank adapters.

---

### 3. Mathematical Formulations & Derivations

#### A. The Pre-training Paradigm
Let the pre-training dataset consist of a massive set of unlabeled examples $\mathcal{D}\_{\text{pre}} = \{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}$ where $N$ is exceptionally large (e.g., 1 trillion tokens in modern LLM regimes). The goal of pre-training is to minimize an unsupervised loss function (e.g., next-token prediction or self-supervised contrastive objectives) over parameters $\theta$:
$$
\mathcal{L}\_{\text{pre}}(\theta) = \frac{1}{N} \sum\_{i=1}^N \mathcal{L}\_{\text{unsup}}(x^{(i)}; \theta)
$$
The output of this pre-training phase is a fixed set of weights $\hat{\theta}$ representing the learned representation function $\phi\_{\hat{\theta}}(x)$.

#### B. Linear Probing Setup
Given a labeled downstream dataset $\mathcal{D}\_{\text{task}} = \{(x^{(1)}, y^{(1)}), \dots, (x^{(n)}, y^{(n)})\}$, where the dataset size $n$ is typically small (few-shot or low-resource regimes). 
In Linear Probing, the pre-trained weights $\hat{\theta}$ are kept strictly frozen. We introduce a linear head parameterized by $W \in \mathbb{R}^{k \times m}$ (for a $k$-class classification task).
The prediction for class $j$ given input $x$ is:
$$
h(x)\_j = \frac{e^{W\_j^T \phi\_{\hat{\theta}}(x)}}{\sum\_{l=1}^k e^{W\_l^T \phi\_{\hat{\theta}}(x)}}
$$
The objective is to optimize only the linear head $W$:
$$
\min\_{W} \frac{1}{n} \sum\_{i=1}^n \mathcal{L}\left(W^T \phi\_{\hat{\theta}}(x^{(i)}), y^{(i)}\right)
$$
#### C. Fine-Tuning Optimization and Initialization
In Fine-Tuning, both $W$ and $\theta$ are treated as free variables. Crucially, the optimization is initialized at the pre-trained weights:
$$
\theta^{(0)} = \hat{\theta}, \quad W^{(0)} \sim \mathcal{N}(0, \sigma^2 I)
$$
The joint objective is:
$$
\min\_{W, \theta} \frac{1}{n} \sum\_{i=1}^n \mathcal{L}\left(W^T \phi\_\theta(x^{(i)}), y^{(i)}\right)
$$
*Theoretical Note*: Because neural networks are overparameterized, there exist millions of global minima where the training loss is zero. Initializing $\theta$ with $\hat{\theta}$ ensures the optimizer converges to a basin of the loss landscape that generalizes exceptionally well to test data, whereas random initialization converges to a poor minimum that overfits.

#### D. The Mathematics of LoRA (Low-Rank Adaptation)
Consider a pre-trained linear weight layer (such as in an MLP or attention block) represented by a matrix $W\_0 \in \mathbb{R}^{d \times d}$. During fine-tuning, the weight matrix is updated to $W = W\_0 + \Delta W$. 
Rather than updating all $d^2$ parameters in $W$, LoRA restricts the update $\Delta W$ to a low-rank subspace:
$$
\Delta W = A B
$$
where:
- $W\_0 \in \mathbb{R}^{d \times d}$ is kept frozen.
- $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$ are trainable parameters.
- $r \ll d$ is the rank of the adaptation.

For an activation input vector $h\_{\text{in}} \in \mathbb{R}^d$, the forward pass computation is formulated as:
$$
h\_{\text{out}} = W\_0 h\_{\text{in}} + \Delta W h\_{\text{in}} = W\_0 h\_{\text{in}} + A B h\_{\text{in}}
$$
<div id="plotly-12-lora-diagram" class="plotly-chart" aria-label="Interactive Plotly chart: LoRA forward pass with frozen base and low-rank adapter branch"></div>

<p><em>Figure: LoRA forward pass with frozen base and low-rank adapter branch.</em></p>

#### E. LoRA Parameter Initialization
To ensure that the model behavior is exactly unmodified at the start of training (i.e., $\Delta W = 0$), the matrices are initialized as:
$$
A = 0, \quad B \sim \mathcal{N}(0, \sigma\_B^2 I)
$$
Thus, at iteration 0:
$$
\Delta W = A B = 0 \cdot B = 0
$$
which guarantees that the pre-trained representation is preserved at step-zero.

---

### 4. Step-by-Step Workflows

#### A. LP-FT Optimization Workflow
To bypass the destructive updates of standard fine-tuning, LP-FT uses a sequential strategy:
1. **Initialize representation**: Set $\theta = \hat{\theta}$ (the pre-trained model weights).
2. **Linear Probe (LP) Phase**: Keep $\theta$ frozen and optimize $W$ until convergence on the downstream training set:
   

$$W^{(0)} = \arg\min\_{W} \frac{1}{n} \sum\_{i=1}^n \mathcal{L}\left(W^T \phi\_{\hat{\theta}}(x^{(i)}), y^{(i)}\right)
$$
3. **Fine-Tuning (FT) Phase**: Initialize the joint parameters at $(W^{(0)}, \hat{\theta})$ and optimize both jointly:
   

$$(W\_{\text{final}}, \theta\_{\text{final}}) = \arg\min\_{W, \theta} \frac{1}{n} \sum\_{i=1}^n \mathcal{L}\left(W^T \phi\_\theta(x^{(i)}), y^{(i)}\right)
$$
   *Benefit*: Because $W^{(0)}$ is already aligned with the pre-trained features, the gradients backpropagated through $\theta$ in the second phase are small and non-destructive.

#### B. LoRA Training and Multi-Tenant Deployment
1. **Freeze Base Model**: Identify weight matrices in the base model (e.g., projection layers) and freeze their parameters $W\_0$.
2. **Attach Adapters**: For each selected layer, attach trainable low-rank adapters $A$ and $B$.
3. **Compute Gradients & Update**: For a downstream training dataset, compute the loss, perform backward propagation, and update only $A$ and $B$ (typically using the Adam optimizer).
4. **Deploy Multi-Tenant System**:
   - Keep a single instance of the massive base model $W\_0$ loaded in GPU High Bandwidth Memory (HBM).
   - Store tiny, specialized adapter weights $(A\_1, B\_1)$ for User 1, $(A\_2, B\_2)$ for User 2, etc..
   - At inference time, dynamically swap or route the user's input to pass through the shared base model and their specific low-rank adapter.

---

### 5. Practical Systems & Memory Analysis of LoRA

#### A. Parameter Reduction Analysis
If we adapt a single weight layer of size $d = 4096$ with a rank $r = 8$:
- **Full Fine-Tuning**: Modifies $d \times d = 4096^2 \approx 16.78 \text{ million parameters}$.
- **LoRA**: Modifies $2 \times d \times r = 2 \times 4096 \times 8 = 65,536 \text{ parameters}$.
This represents a **$99.6\%$ reduction** in trainable parameters.

#### B. Hardware Memory Savings (Optimizer States)
The primary memory bottleneck in training large models is not storing the parameters themselves, but storing the optimizer states:
- For **Adam**, each trainable parameter requires maintaining its first moment (momentum) and second moment (variance), consuming **8 bytes of memory per parameter** in FP32.
- By reducing the number of trainable parameters by $99.6\%$, LoRA dramatically shrinks the optimizer's HBM footprint. While the base model parameters $W\_0$ must still occupy memory, freezing them means we store zero optimizer states for them, freeing up gigabytes of GPU memory to allow for larger batch sizes or training on consumer hardware.

---

### 6. Applications
- **Multi-Tenant SaaS API Gateways**: Cloud providers (like OpenAI and Anthropic) use LoRA to serve custom fine-tuned models for millions of different customers efficiently, swapping megabyte-sized adapters on a shared gigabyte-sized base network on the fly.
- **Domain-Specific Medical Imaging**: In clinical diagnostic systems (e.g., oncology detection across different hospital scanners), medical institutions pre-train a massive visual model on general images, and apply linear probing to train extremely lightweight classification heads for distinct diseases using highly restricted local datasets.

---

### 7. Reflection Questions
1. **Destructive Updates in Fine-Tuning**: Why does starting with a randomly initialized linear head $W$ in standard fine-tuning damage pre-trained representation parameters $\theta$? How does the LP-FT protocol mathematically resolve this?
2. **LoRA Identity Guarantee**: Why is it mathematically essential that matrix $A$ in LoRA is initialized to $0$ while matrix $B$ is initialized randomly? What would happen on the very first training iteration if both matrices were initialized randomly?
3. **Memory Footprint of Optimization**: If the base model weights $W\_0$ must still reside in GPU High Bandwidth Memory (HBM) during training, explain why freezing them under the LoRA framework still leads to dramatic memory savings.

---

### 8. Further Reading & Resources
- **Stanford Foundation Models Survey (Percy Liang et al., 2021)**: The seminal 200+ page survey paper that coined the term "foundation model" and formalized the pre-training and adaptation paradigm.
- **LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)**: The foundational PEFT paper describing the mathematical formulation and systems advantages of low-rank updates.
- **LP-FT Generalization Analysis (Kumar et al., 2022)**: Highly recommended for analyzing the theoretical and empirical differences in generalization bound performance between linear probing, fine-tuning, and the two-step LP-FT.
