# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 14: Transformers, In-Context Learning

### 1. Summary
This lecture introduces the deep learning architectures and probabilistic paradigms that power modern **Large Language Models (LLMs)**. The instructor, Tenyu, systematically constructs the stack of autoregressive sequence modeling. Starting with raw text, the lecture details **tokenization** (explicitly comparing character, word, and subword Byte-Pair Encoding schemes) and maps how tokens are projected into continuous high-dimensional vector embeddings. It establishes the formal next-token classification setup, showing how language modeling represents a sequence of conditional softmax choices over a massive vocabulary, trained via **Negative Log-Likelihood (NLL) / Cross-Entropy Loss**. 

The core of the lecture focuses on the inner workings of the **Transformer architecture**. It highlights why a simple feedforward Multi-Layer Perceptron (MLP) cannot scale to sequential data and motivates **Self-Attention** as the fundamental mechanism for fusing semantic context across positions. After deriving the mathematical formulation of single-head causal attention and its matrix representation, the lecture introduces **Causal Masking** (adding $-\infty$ bounds to prevent future information leakage) and **Multi-Head Attention (MHA)**. Finally, the discussion addresses practical systems-level limits, analyzing the quadratic $O(T^2)$ computational and memory constraints of sequence length $T$ and outlining how contemporary hardware-aware tiling algorithms like **FlashAttention** mitigate these bottlenecks on modern GPU SRAM/HBM.

---

### 2. Key Concepts & Definitions
- **Autoregressive Sequence Modeling**: A probabilistic modeling paradigm where the joint probability of a sequence of tokens is modeled sequentially, generating one token at a time based on the history of previously generated tokens.
- **Subword Tokenization (BPE)**: A hybrid tokenization technique (such as Byte-Pair Encoding) that breaks down words into the most frequent subword segments. It solves the sequence length explosion of character-level tokenization and the out-of-vocabulary (OOV) / morphological parameter-sharing failures of word-level tokenization.
- **Beginning of Sentence (BOS) Token ($x\_0$)**: A fixed special token prepended to the start of every text sequence during training and inference to provide an initial conditioning state.
- **Logits**: The raw, unnormalized outputs of a transformer corresponding to the vocabulary size $|V|$. Passing these through a softmax function maps them to a valid probability distribution.
- **Causal Masking**: A triangular matrix operation applied to raw attention scores during training to block information flowing from future tokens ($j > t$) back to the current token ($t$), ensuring the model remains causal.
- **Isotropic Embeddings / Scaling Invariance**: Properties established via LayerNorm and RMSNorm to keep vector sizes and scaling factors consistent across deep stacks of hidden dimensions.
- **FlashAttention**: A hardware-aware tiling algorithm that computes exact attention on GPU SRAM blocks to avoid reading and writing the $T \times T$ attention matrix back and forth to high-latency GPU High Bandwidth Memory (HBM).

---

### 3. Mathematical Formulations & Derivations

#### A. Autoregressive Factored Joint Probability
Given a vocabulary $V$ containing subword segments, the total number of possible combinations of length $T$ is $|V|^T$. We model the joint probability distribution of a sequence of tokens $\mathbf{x} = (x\_1, x\_2, \dots, x\_T) \in V^T$ by factoring it into a chain of conditional distributions using the probability chain rule:
$$
P(x\_1, x\_2, \dots, x\_T) = \prod\_{t=1}^T P(x\_t \mid x\_1, x\_2, \dots, x\_{t-1})
$$
#### B. The Softmax Next-Word Predictor
For each time step $t$, the transformer model parameterized by $\theta$ maps the history of prior tokens $(x\_0, x\_1, \dots, x\_{t-1})$ to a vector of logits $U\_t \in \mathbb{R}^{|V|}$:
$$
U\_t = f\_\theta(x\_0, x\_1, \dots, x\_{t-1})
$$
Applying the softmax function entrywise produces a probability vector on the simplex of dimension $|V|$:
$$
P(x\_t = j \mid x\_1, \dots, x\_{t-1}; \theta) = \text{softmax}(U\_t)\_j = \frac{e^{U\_{t, j}}}{\sum\_{l=1}^{|V|} e^{U\_{t, l}}}
$$
#### C. Next-Token Prediction Loss (Negative Log-Likelihood)
At training time, the model is shown a complete sequence of tokens. We minimize the Negative Log-Likelihood (NLL) of predicting the correct next token $x\_t$ over all positions $t \in \{1, \dots, T\}$:
$$
\mathcal{L}(\theta) = -\log P(x\_1, \dots, x\_T \mid \theta) = -\sum\_{t=1}^T \log P(x\_t \mid x\_1, \dots, x\_{t-1}; \theta)
$$
$$
\mathcal{L}(\theta) = \sum\_{t=1}^T \left[ -U\_{t, x\_t} + \log \sum\_{l=1}^{|V|} e^{U\_{t, l}} \right]
$$
*Derivation Note*: This loss is equivalent to the categorical cross-entropy loss between the true next-token (represented as a one-hot target vector over $V$) and the model's predicted probability distribution.

---

### 4. Step-by-Step Optimization & Generation Algorithms

<div id="plotly-14-causal-attention" class="plotly-chart" aria-label="Interactive Plotly chart: Causal attention probability heatmap"></div>

<p><em>Figure: Causal attention probability heatmap.</em></p>

#### A. Autoregressive Decoding with Temperature Scaling
Once a model $\theta$ is trained, we generate text sequentially. To balance creativity and coherence, we apply temperature scaling $\tau > 0$ to the logits vector $U\_t$ before taking the softmax:
1. **Initialize**: Given a prompt sequence $(x\_1, \dots, x\_k)$, prepend the BOS token $x\_0$.
2. **Loop until termination** (e.g., generating an End of Sentence token or reaching max context length $T$):
   - Compute logits $U\_t = f\_\theta(x\_0, \dots, x\_{t-1})$.
   - Scale logits by temperature $\tau$ and apply softmax:

$$P(x\_t = j \mid x\_1, \dots, x\_{t-1}) = \frac{e^{U\_{t, j}/\tau}}{\sum\_l e^{U\_{t, l}/\tau}}
$$
   - **Temperature Properties**:
     - **$\tau \to 0$ (Greedy / Deterministic Decoding)**: The probability mass concentrates entirely on the token with the maximum logit.

$$P(x\_t = j) \to \begin{cases} 1 & j = \text{argmax}\_l U\_{t, l} \\ 0 & \text{otherwise} \end{cases}
$$
     - **$\tau > 1$ (High Stochasticity)**: The distribution flattens, increasing sample diversity by pulling from the long tail of the vocabulary.
   - **Top-$k$ Filtering**: Retain only the top-$k$ most probable tokens, zero out all other indices, and renormalize the distribution to prevent generating incoherent long-tail tokens.
   - **Sample** the next token $x\_t \sim P(x\_t \mid x\_{1 \dots t-1})$.
   - **Append** $x\_t$ to the context window and repeat.

#### B. Single-Head Causal Self-Attention
Self-attention maps an input sequence of hidden representation row vectors $H^{\text{in}} = [h\_1^{\text{in}}; \dots; h\_T^{\text{in}}] \in \mathbb{R}^{T \times d}$ to an output sequence of hidden states $H^{\text{out}} \in \mathbb{R}^{T \times d}$.
1. **Project Inputs to Queries, Keys, and Values**:
   We multiply the input row vectors on the right by trained weight matrices $W^Q \in \mathbb{R}^{d \times d\_h}$, $W^K \in \mathbb{R}^{d \times d\_h}$, and $W^V \in \mathbb{R}^{d \times d}$:

$$q\_t = h\_t^{\text{in}} W^Q \in \mathbb{R}^{1 \times d\_h}
$$
$$k\_t = h\_t^{\text{in}} W^K \in \mathbb{R}^{1 \times d\_h}
$$
$$v\_t = h\_t^{\text{in}} W^V \in \mathbb{R}^{1 \times d}
$$
2. **Compute Raw Causal Attention Scores with Masking**:
   For any position $t$, we compute the dot products between query $q\_t$ and keys $k\_1, \dots, k\_T$ to capture contextual relevance. To maintain causality, future keys ($j > t$) are mathematically deleted by adding negative infinity ($-\infty$) to their raw scores:

$$A\_{t, j} = \begin{cases} \frac{q\_t k\_j^T}{c} & j \le t \\ -\infty & j > t \end{cases}
$$
   where $c = \sqrt{d\_h}$ is a scaling constant preventing raw dot products from pushing softmax gradients into vanishing regimes.
3. **Normalize with Softmax**:

$$P\_t = \text{softmax}(A\_t \in \mathbb{R}^{1 \times T})
$$
   Because $e^{-\infty} = 0$, future token coefficients are nullified, ensuring $P\_{t, j} = 0$ for all $j > t$.
4. **Fleshing out the Value Aggregation**:

$$h\_t^{\text{out}} = \sum\_{i=1}^t P\_{t, i} v\_i
$$
   This weighted convex combination aggregates the values of all historically relevant vectors.

**Causal Matrix Formulation**:
Grouping all steps together, the vectorized attention block is written as:
$$
H^{\text{out}} = \text{softmax}\left( \frac{Q K^T}{\sqrt{d\_h}} + M \right) V
$$
where $M \in \mathbb{R}^{T \times T}$ is the causal mask matrix:
$$
M\_{i, j} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$
---

### 5. Architectural Building Blocks

#### A. Multi-Head Attention (MHA)
To enable the network to simultaneously attend to different semantic structures (such as grammatical parsing vs. entity sentiment), we run $n\_h$ parallel attention heads:
1. **Parallel Computations**: Each head $i \in \{1, \dots, n\_h\}$ uses separate parameter sets $\{W\_i^Q, W\_i^K, W\_i^V\}$ to produce its own output matrix $\text{Head}\_i \in \mathbb{R}^{T \times d\_{\text{head}}}$.
2. **Concatenation and Output Projection**:
   The outputs are concatenated column-wise and projected back to the hidden space $d$ using the trained matrix $W^O \in \mathbb{R}^{(n\_h d\_{\text{head}}) \times d}$:

$$\text{MHA}(H^{\text{in}}) = \text{concat}\left( \text{Head}\_1, \text{Head}\_2, \dots, \text{Head}\_{n\_h} \right) W^O
$$
#### B. Multi-Layer Perceptrons (MLPs) vs. Self-Attention
Within a Transformer block, layers are strictly divided:
- **Self-Attention**: Fuses information across different temporal positions. It is the *only* operator that communicates across different steps in the context window.
- **MLP Block**: A multi-layer feedforward neural network applied independently and in parallel to each position $t$. It has no temporal cross-talk and processes every token's hidden state separately with shared parameters.
*Theoretical Motivation*: A massive, fully connected MLP covering the entire context window would suffer from parameter explosion ($O(T^2 d^2)$) and fail to generalize across varying sequence lengths $T$. Interleaving context-fusing Attention with position-wise MLPs acts as a vital inductive bias to restrict parameter count while retaining representational capacity.

<div id="plotly-14-attention-scaling" class="plotly-chart" aria-label="Interactive Plotly chart: Attention memory scaling versus FlashAttention"></div>

<p><em>Figure: Attention memory scaling versus FlashAttention.</em></p>

---

### 6. Computational Complexity & Hardware Realities

#### A. Causal Attention Bottleneck
The computational and memory limits of modern language models are heavily constrained by sequence length $T$.
- **FLOPs Complexity**: Computing $Q K^T$ involves taking the inner product of $T$ queries with $T$ keys. This requires $T^2$ dot-product calculations of dimension $d\_h$, resulting in a complexity of:

$$\text{FLOPs} = O(T^2 d\_h)
$$
- **Memory Footprint Complexity**: Storing the raw and normalized attention matrix requires storing $T \times T$ values per layer, leading to:

$$\text{Space Complexity} = O(T^2)
$$
As $T$ grows to millions of tokens, these $O(T^2)$ bottlenecks become prohibitive, requiring the truncation or compacting of sequence history.

#### B. FlashAttention Optimization
In modern deep learning systems, the physical bottleneck of attention calculation is often memory bandwidth rather than compute capabilities.
- **The Issue**: In standard implementations, the large $T \times T$ attention matrix is repeatedly written to and read from the GPU's High Bandwidth Memory (HBM), which is extremely slow.
- **The Solution**: FlashAttention is a hardware-aware tiling algorithm. It partitions the queries, keys, and values into block tiles that fit inside the GPU's fast, local on-chip **SRAM (Static Random-Access Memory)**. It computes softmax online and performs matrix multiplication incrementally inside SRAM, writing to slow HBM only at the very end. This reduces the HBM memory access complexity from quadratic $O(T^2)$ to linear $O(T)$, enabling massive speedups without altering the mathematical output of attention.

---

### 7. Applications
- **Autoregressive Text Generators**: Foundations of conversational assistants (such as OpenAI's GPT models or Anthropic's Claude), generating tokens sequentially for general reasoning tasks.
- **Vision-Language-Action (VLA) Robotic Controllers**: Modern robotic systems leverage pretrained multi-modal Transformer backbones, converting vision data into tokenized embeddings and using autoregressive self-attention steps to output physical robotic trajectories and actions.

---

### 8. Reflection Questions
1. **Subword vs. Word-Level Tokenization**: How does the use of subword Byte-Pair Encoding (BPE) prevent "out-of-vocabulary" errors during model deployment when encountering completely novel, synthetic words (e.g., `LLMefication`), and how does this affect downstream representational capacity?
2. **The MLP Parameter Bottleneck**: Why would a single, massive Multi-Layer Perceptron (MLP) designed to take a concatenated vector of all sequence tokens ($H \in \mathbb{R}^{Td}$) be strictly inferior to interleaving self-attention with position-wise MLPs, both in terms of parameter scaling and handling variable sequence lengths?
3. **Causal Masking and Softmax Behavior**: Mathematically analyze what would occur to the softmax probability distribution $P\_t$ at step $t$ if causal masking was implemented by assigning $M\_{i, j} = 0$ instead of $M\_{i, j} = -\infty$ for future positions ($j > i$). How does the choice of $-\infty$ guarantee causal boundaries?

---

### 9. Further Reading & Resources
- **Vaswani et al. (2017) "Attention Is All You Need"**: The seminal paper establishing the Transformer architecture.
- **Dao et al. (2022) "FlashAttention"**: Highly recommended for understanding hardware-efficient tiling of online softmax on GPU SRAM blocks.
- **Qwen 3.5 & Claude Code Tokenizer Playground**: Online diagnostic playgrounds showing Byte-Pair Encoding and subword granularity shifts.
