# CS231N (Deep Learning for Computer Vision) Rigorous Study Notes

### CS231N Lecture 7: Recurrent Neural Networks

#### 0. Quick-Recall Summary
*   **Sequential Recurrence Paradigm:** Sequence modeling resolves variable-length input/output constraints by maintaining a dynamic hidden state vector $h_t = f_W(h\_{t-1}, x_t)$ utilizing parameter sharing (W) across all temporal ticks.
*   **The BPTT Bottleneck:** Backpropagation Through Time (BPTT) unrolls the computational graph over the sequence length, requiring repeated multiplication of the transition matrix $W\_{hh}$, which triggers exponential vanishing or exploding gradients.
*   **Gradient Cliff Mitigation:** Exploding gradients are solved via **gradient clipping** (clipping the norm above a threshold), while vanishing gradients are addressed by the **LSTM** architecture, which establishes an additive, linear cell-state "highway".
*   **Truncated BPTT:** To avoid massive GPU memory footprints ($O(T)$ activation caching) over long sequences, training is chunked into fixed temporal windows where gradients are zeroed at boundaries, but hidden states are forwarded.
*   **Multimodal Sequence Conditioning:** Tasks like image captioning inject high-level visual features (CNN feature map outputs) directly into the RNN hidden state initialization $h_0 = f_W(v)$, bypassing the raw pixel representation.

---

#### 1. Core Concept & Learning Objectives
*   **Objective:** To master sequence modeling paradigms capable of processing variable-length inputs and outputs (one-to-many, many-to-one, many-to-many). This lecture teaches the mechanics of maintaining temporal memory through hidden states, derives the mathematical failure modes of vanilla recurrence, and explains the structural solutions offered by gated architectures (LSTMs).
*   **Lecture Category:** (b) Architecture Design and (c) Training/Optimization Practice (gradient stability over long graphs).
*   **Builds on:** Lecture 4 (Neural Networks and Backpropagation) and Lecture 6 (CNN Architectures as pre-trained visual encoders).

---

#### 2. Mathematical Foundations

##### Recurrence and Output Formalisms (Vanilla RNN)
At each temporal step $t$, the hidden state vector $h_t \in \mathbb{R}^H$ is updated using the previous hidden state $h\_{t-1} \in \mathbb{R}^H$ and the current input $x_t \in \mathbb{R}^D$:
$$h_t = \tanh(W\_{hh} h\_{t-1} + W\_{xh} x_t + b_h)$$
where $W\_{hh} \in \mathbb{R}^{H \times H}$, $W\_{xh} \in \mathbb{R}^{H \times D}$, and $b_h \in \mathbb{R}^H$ are shared across all steps.
The network predicts an output vector $y_t \in \mathbb{R}^C$ at step $t$ via:
$$y_t = W\_{hy} h_t + b_y$$
where $W\_{hy} \in \mathbb{R}^{C \times H}$ and $b_y \in \mathbb{R}^C$.

##### The Mathematical Proof of Vanishing/Exploding Gradients
Consider a vanilla RNN unrolled for $T$ steps. Let a loss term $L$ be computed at step $T$ (e.g. $L = \mathcal{L}(y_T, \hat{y})$).
To calculate the gradient of the loss with respect to the hidden state at an early step $t$, we apply the calculus chain rule:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \frac{\partial h_T}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod\_{k=t+1}^{T} \frac{\partial h_k}{\partial h\_{k-1}}$$
The local Jacobian matrix mapping hidden state transitions is:
$$\frac{\partial h_k}{\partial h\_{k-1}} = \text{diag}\left(1 - \tanh^2(W\_{hh} h\_{k-1} + W\_{xh} x_k + b_h)\right) W\_{hh}^T$$
Since the final gradient contains a product of these Jacobians:
$$\frac{\partial h_T}{\partial h_t} = \prod\_{k=t+1}^{T} \text{diag}\left(1 - \tanh^2(\cdot)\right) W\_{hh}^T$$
*   **Exploding Gradients:** If the largest singular value (spectral radius) of the shared weight matrix $W\_{hh}$ is greater than $1$ ($\rho(W\_{hh}) > 1$), then as the temporal distance $(T - t)$ grows, the repeated matrix product $\left(W\_{hh}^T\right)^{T-t}$ scales exponentially, driving the gradient norm toward infinity.
*   **Vanishing Gradients:** If $\rho(W\_{hh}) < 1$, or because the derivative of the hyperbolic tangent is bounded by $\frac{d\tanh(z)}{dz} = 1 - \tanh^2(z) \in (0, 1]$ (which is almost always strictly less than 1 in active units), the repeated product of fractional scalar factors drives the gradient norm exponentially to $0$, blocking long-range temporal dependencies.

##### Gradient Norm Clipping Heuristic
To prevent numerical instability (NaNs) caused by exploding gradients, the gradient vector $g = \frac{\partial L}{\partial W}$ is scaled if its $L_2$ norm exceeds a threshold:
$$\hat{g} = \begin{cases} g & \text{if } \|g\|_2 \le \tau \ \frac{\text{threshold}}{\|g\|_2} g & \text{if } \|g\|_2 > \tau \end{cases}$$
where $\tau$ is a hyperparameter representing the maximum allowed gradient norm.

##### Gated Recurrent Formulations: Long Short-Term Memory (LSTM)
To resolve the vanishing gradient problem, the LSTM introduces a cell state vector $c_t \in \mathbb{R}^H$ acting as an additive memory highway.
At step $t$, the LSTM projects the concatenated vector $[h\_{t-1}, x_t]$ to compute four gate vectors $i, f, o, g \in \mathbb{R}^H$:
$$\begin{aligned}
\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} &= \begin{pmatrix} \sigma \\ \sigma \\ \sigma \\ \tanh \end{pmatrix} \left( W \begin{pmatrix} h\_{t-1} \\ x_t \end{pmatrix} + b \right) \
c_t &= f \odot c\_{t-1} + i \odot g \
h_t &= o \odot \tanh(c_t)
\end{aligned}$$
where:
*   $f \in^H$ is the **forget gate**, specifying how much of the old memory $c\_{t-1}$ to retain.
*   $i \in^H$ is the **input gate**, controlling how much new information to write to memory.
*   $g \in [-1, 1]^H$ is the **gate gate** (candidate cell state), representing the new memory update.
*   $o \in^H$ is the **output gate**, determining what parts of the cell state $c_t$ are exposed to the hidden state $h_t$.
*   $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function, and $\odot$ denotes the element-wise (Hadamard) product.

---

#### 3. Architecture / Algorithm Walkthrough

##### Temporal Data Flow and Backpropagation Through Time (BPTT)
Unlike standard feedforward networks, RNNs share weights across all time steps. The forward and backward operations proceed as follows:

```
Forward Pass (Temporal Unrolling):
 h0 (init) ───────> [ RNN Cell ] ───────> [ RNN Cell ] ───────> [ RNN Cell ]
                       ▲                    ▲                    ▲
                       │                    │                    │
                      x_1                  x_2                  x_3
                       │                    │                    │
                       ▼                    ▼                    ▼
                      y_1                  y_2                  y_3
                       │                    │                    │
                       ▼                    ▼                    ▼
                     Loss L1              Loss L2              Loss L3

Backward Pass (BPTT):
* Sum gradients of shared parameters across all time steps:
  dL/dW_hh = dL1/dW_hh + dL2/dW_hh + dL3/dW_hh
```

##### PyTorch Blueprint (Custom RNN & LSTM Implementations)
This blueprint implements a custom vanilla Recurrent cell and a fully functional gated LSTM cell from scratch using only tensor operations to illustrate the mathematical data flow:

```python
import torch
import torch.nn as nn

class CustomRNNCell(nn.Module):
    """
    Vanilla RNN cell executing: h_t = tanh(W_hh * h\_{t-1} + W_xh * x_t + b_h)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CustomRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Combined parameters for hardware efficiency (fusing linear projections)
        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False) # Bias is already in W_xh
        self.tanh = nn.Tanh()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x_t: [Batch, Input_Dim], h_prev: [Batch, Hidden_Dim]
        h_next = self.tanh(self.W_xh(x_t) + self.W_hh(h_prev))
        return h_next


class CustomLSTMCell(nn.Module):
    """
    Gated LSTM Cell utilizing linear projections to compute forget, input,
    output, and candidate gates, with additive cell state updates.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CustomLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Project combined [h_prev, x_t] to 4 * hidden_dim for parallel gating math
        self.W_gate = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        # Concatenate hidden state and input vector along dimension 1
        combined = torch.cat([h_prev, x_t], dim=1) # Shape: [Batch, Hidden_Dim + Input_Dim]
        
        # Project and split into the four gates
        gate_projections = self.W_gate(combined) # Shape: [Batch, 4 * Hidden_Dim]
        i_gate, f_gate, o_gate, g_gate = torch.chunk(gate_projections, 4, dim=1)
        
        # Apply non-linear gate activations
        i = self.sigmoid(i_gate) # Input gate
        f = self.sigmoid(f_gate) # Forget gate
        o = self.sigmoid(o_gate) # Output gate
        g = self.tanh(g_gate)    # Candidate cell state
        
        # Additive cell state update (memory highway)
        c_next = f * c_prev + i * g
        
        # Hidden state calculation
        h_next = o * self.tanh(c_next)
        
        return h_next, c_next
```

---

#### 4. Visual Intuition & Interpretability

##### Character-Level Language Model State Visualizations
The lecture references visualizations of hidden cells in high-capacity character-level language models (such as Andrej Karpathy's 2015 RNN analysis):
*   **Quotes Cell:** Certain individual neurons in the hidden state are optimized to track whether the sequence is currently inside a quotation block. The cell activation switches to $+1$ upon encountering a quote character `"` and stays active until the matching quote terminates, tracking long-term structural dependencies.
*   **Code Indentation Cell:** In models trained on code repositories, specific cells track the depth of code indentation blocks (nesting tab levels), steadily scaling activations up or down at brackets `{` and `}`.
*   **Line-Length Counter Cell:** A cell acts as a step counter, slowly building up its activation level with each successive character and resetting to zero immediately after encountering a newline character `
`.

##### Multi-layer RNN Dependencies
Multi-layer RNNs process sequences in both the temporal (horizontal) and architectural depth (vertical) dimensions:

```
Layer 2:  h2_1 ───────> h2_2 ───────> h2_3 ───────> y_t (output)
            ▲            ▲            ▲
            │            │            │
Layer 1:  h1_1 ───────> h1_2 ───────> h1_3
            ▲            ▲            ▲
            │            │            │
Inputs:    x_1          x_2          x_3
```

##### Failure Modes & Biases
*   **The Co-occurrence/Visual Bias Trap:** Recurrent models trained on image captioning datasets often hallucinate objects based on statistical co-occurrences rather than grounding decisions on raw image pixels. For example, when shown a person holding an object near their face, the model frequently captions it as "a person speaking on a phone", or seeing a hand near a round object outputs "throwing a ball" even if the ball is moving into a glove.
*   **Temporal Context Decoupling:** In vanilla RNNs, early inputs are systematically overwritten by later tokens due to the continuous squashing effect of the $W\_{hh}$ and $\tanh$ updates, making the model completely blind to long context.

---


<div id="plotly-cs231n-7-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 7 visualization blueprint"></div>
<p><em>Figure: Interactive visualization corresponding to the blueprint below.</em></p>

#### 5. Visualization Blueprint (Conceptual Spec)
To represent the inner workings of an LSTM's gates, we propose an interactive **LSTM Gating Dynamics Visualizer**:

*   **Visualization Type:** Dynamic Flow Graph with synchronized Activation Heatmaps.
*   **Data Fields & Encoding:**
    *   **Node Color Map:** Gates ($i, f, o, g$) color-coded individually (e.g., $f$ = red/green denoting forget/retain; $i$ = blue denoting write magnitude).
    *   **Vector Dimensions:** Represented as an $H$-dimensional horizontal grid of pixels inside each gate block.
    *   **Edge Thickness:** Proportional to the magnitude of the activation or gradient moving backward.
*   **Interactive Controls:**
    *   **Character Input Box:** Allows typing a string (e.g., `if (x > y) { print(x); }`).
    *   **Time-step Slider:** Moves step-by-step through characters, showing the hidden state $h_t$ and cell state $c_t$ updating live.
    *   **Gate Filter Toggle:** Toggles the isolation of a single gate (e.g., freezing the forget gate to $f_t=1$ to demonstrate the infinite gradient highway).

---

#### 6. Empirical Design Heuristics & Benchmark Results
*   **Defaults for Hidden State Initialization:** Standard practice is to initialize the initial hidden state $h_0$ (and $c_0$ in LSTMs) to all zeros. However, learning $h_0$ as a trainable parameter often yields marginal convergence boosts on highly structured sequences.
*   **Truncated BPTT Step Size:** When training on long sequences (e.g., book chapters, audio waveforms), the truncation window is typically set between $16$ and $100$ steps depending on GPU hardware memory limits, striking a balance between memory footprint and gradient propagation range.
*   **Activation Bounds:** The hyperbolic tangent ($\tanh$) activation function is systematically preferred in vanilla RNN states and LSTM output paths over ReLU. Because $\tanh$ maps activations strictly to the open interval $(-1, 1)$, it stabilizes activations across hundreds of unrolled iterations, whereas unbounded ReLU activations tend to explode rapidly when repeatedly projected by $W\_{hh}$.

---

#### 7. Pitfalls, Debugging Tips & Reflection Questions

##### Gotchas & Silent Failure Modes
*   **Vanishing Gradients in LSTM Cell Highways:** Although LSTMs alleviate vanishing gradients via additive paths ($c_t = f \odot c\_{t-1} + i \odot g$), if the forget gate bias is initialized too low (e.g., 0), the network will repeatedly clear its memory. **Debugging Tip:** Always initialize the forget gate bias $b_f$ to a high positive value (e.g., $1.0$ or $2.0$) at the start of training to enforce memory retention.
*   **Gradient Explosion Instability:** A model's loss suddenly jumping to `NaN` during sequence training is a clear signature of exploding gradients in backpropagation through time. **Debugging Tip:** Inspect loss logs; if the loss spikes abruptly, immediately enable gradient clipping with a maximum norm threshold of $\tau = 1.0$ or $5.0$.

##### Graduate-Level Reflection Questions
1.  **Gated Additive Highway Derivation:** Mathematically prove how the LSTM cell-state update ($c_t = f_t \odot c\_{t-1} + i_t \odot g_t$) prevents vanishing gradients. Specifically, derive the partial derivative $\frac{\partial c_t}{\partial c\_{t-1}}$ and analyze the gradient flow when the forget gate $f_t \approx 1$. How does this contrast with the vanilla recurrence update?
2.  **The Context Compression Bottleneck:** Given an RNN with a hidden state dimension of $H = 256$, what is the theoretical limit on the amount of information the network can retain over an input sequence of length $T = 1000$? Express this through the lens of lossy information compression.
3.  **Truncated BPTT Bias:** Explain the mathematical and empirical trade-offs of using Truncated Backpropagation Through Time (TBPTT). If we choose a truncation window of $k = 20$ steps for a sequence task that requires a context window of at least $100$ steps to resolve temporal dependencies, how are the learning dynamics affected, and what kind of biases are introduced to the gradient estimates?
