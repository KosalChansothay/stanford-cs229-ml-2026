# CS336 Lecture 1: Overview, Tokenization

## 0. Quick-Recall Summary
- **The "From-Scratch" Philosophy:** CS336 is built around construction and efficiency. High-level abstractions are leaky, and truly understanding frontier systems requires tearing up the full stack — from BPE tokenizers to Triton custom GPU kernels and scaling laws.
- **The Core Metric — Algorithmic Efficiency:** Model performance is governed by $\text{Accuracy} = \text{Efficiency} \times \text{Resources}$ (where $\text{Efficiency} = \text{Output}/\text{Input}$, and $\text{Resources}$ are compute/data). Algorithmic improvements alone provided a 44x efficiency gain on ImageNet from 2012–2019, making efficiency critical at scale.
- **Tokenization's Dual Purpose:** Tokenizers map raw bytes to integers (token IDs). Under an efficiency lens, tokenization (1) compresses raw byte streams to shorter sequences to mitigate the $O(N^2)$ sequence-length bottleneck of transformer attention, and (2) enables adaptive computation where common patterns use fewer resources.
- **Byte-Pair Encoding (BPE) Foundations:** BPE starts with individual bytes (IDs 0–255) as base tokens. It iteratively counts and merges the most frequent adjacent token pairs, assigning them new IDs (starting at 256). If a test sequence contains rare or unseen patterns, they naturally fallback to individual bytes, eliminating the need for a fallback `[UNK]` token.
- **BPE Efficiency Bottlenecks:** Naive BPE encoding/decoding loops over all merges (which scales with $\text{Vocab Size} - 256$), causing massive bottlenecks. Production-ready implementation requires pre-chunking the input text (e.g., via regex splits on whitespace/punctuation) and indexing merges to only search for relevant active merges.

## 1. Core Paradigm & Systems Overview
- **Objective:** The engineering goal of this lecture is to introduce the foundational framework of language modeling from scratch and explore tokenization as the critical interface between raw text (Unicode) and model-digestible integers. The lecture establishes the core efficiency mindset, demonstrating how data-driven token segmentation balances the trade-off between sequence length compression and vocabulary size.
- **Primary Bottleneck:** Both **compute-bound** (due to the quadratic $O(N^2)$ sequence length scaling of transformer attention) and **data-quality-bound** (since model behavior is directly governed by curated pre-training data mixtures). The tokenizer directly controls this trade-off: a larger vocabulary increases compression ratio (fewer tokens per string, reducing quadratic attention cost) but introduces parameter sparsity and increases the output projection softmax layer's compute overhead.
- **Builds on:** Not applicable (this is the first lecture of the course, establishing the foundational concepts for subsequent lectures).

## 2. Theoretical & Mathematical Primitives
- **Model Accuracy & Resource Equation:**
  The baseline mathematical framing for scaling and progress:

  $\text{Accuracy} = \text{Efficiency} \times \text{Resources} \quad$

  where:

  $\text{Efficiency} = \frac{\text{Output}}{\text{Input}} \quad$


  $\text{Resources} = \text{Input} \quad$

- **Tokenizer Compression Ratio:**
  The efficiency metric evaluating the tokenization quality for a sequence:

  $\text{Compression Ratio} = \frac{\text{Number of Bytes}}{\text{Number of Tokens}} \quad$

  For example, GPT-5's tokenizer achieves a compression ratio of $\approx 2.5$ (e.g., a 20-byte string represented by 8 tokens yields $\frac{20}{8} = 2.5$).
- **Vocabulary Size Scaling:**
  Starting with $256$ base byte values ($0 \le x \le 255$ in UTF-8), each merge operation $M$ adds exactly one token to the vocabulary:

  $\text{Vocabulary Size} = 256 + M \quad$

  Modern multilingual tokenizers target a vocab size of $100\text{k}$ to $200\text{k}$ to achieve higher multilingual compression ratios.

## 3. From-Scratch Algorithmic Workflows & Pseudocode
- **Algorithmic Logic (BPE Training & Encoding):**
  1. **Initialization:** Represent the training corpus as a sequence of raw bytes (UTF-8 encoding). Initialize the vocabulary with the 256 base bytes (IDs 0–255).
  2. **Pair Counting:** Scan the sequence of token IDs to count the frequency of all adjacent pairs $(t\_i, t\_{i+1})$.
  3. **Greedy Merge selection:** Select the pair $(t\_A, t\_B)$ that occurs with the maximum frequency in the corpus. Break ties deterministically (e.g., by selecting the first occurrence).
  4. **Vocabulary Expansion:** Register a new token ID $T = 256 + m$ representing the pair $(t\_A, t\_B)$ and store this merge rule.
  5. **Sequence Substitution:** Scan the corpus and replace every occurrence of the pair $[t\_A, t\_B]$ with the single token ID $T$.
  6. **Iterate:** Repeat steps 2-5 until the vocabulary reaches the target size or no more pairs can be merged.
  7. **Encoding New Text:** Convert the new text to a sequence of bytes. Successively apply each learned merge rule in the exact chronological order of training.
  8. **Decoding:** Recursively expand token IDs back into their constituent byte sequences using the learned merge rules, then decode the final byte sequence back to a Unicode string using UTF-8.
<div id="plotly-cs336-1-bpe-merges" class="plotly-chart" aria-label="Interactive Plotly chart: BPE training compression trace"></div>

<p><em>Figure: Real BPE training on a toy corpus — each merge shrinks the token sequence. Vocabulary size = 256 + M.</em></p>

<div id="plotly-cs336-1-attention-savings" class="plotly-chart" aria-label="Interactive Plotly chart: Quadratic attention savings from tokenization compression"></div>

<p><em>Figure: Attention compute is quadratic in sequence length — a 2.5x compression ratio (GPT-5's tokenizer) cuts attention compute by ~6.25x for the same document.</em></p>

- **PyTorch/Pythonic Blueprint:**
  Below is a minimal, clean, from-scratch Pythonic blueprint of a byte-pair tokenizer to illustrate the algorithmic logic:

```python
import regex as re
from typing import List, Dict, Tuple, Set

class BytePairTokenizer:
    """
    Illustrative from-scratch implementation of Byte-Pair Encoding (BPE).
    Note: Highly educational, naive execution is O(V * L). Production requires indexing.
    """
    def __init__(self):
        # Maps (token_a, token_b) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Maps token_id -> byte sequence
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
    def _get_pair_counts(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_ids(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # Ingest text as raw bytes
        ids = list(text.encode("utf-8"))
        
        for m in range(num_merges):
            counts = self._get_pair_counts(ids)
            if not counts:
                break
            # Find the most frequent pair
            best_pair = max(counts, key=counts.get)
            new_id = 256 + m
            
            # Save merge rule and update vocabulary
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Substitute the pair throughout the token sequence
            ids = self._merge_ids(ids, best_pair, new_id)
            
    def encode(self, text: str) -> List[int]:
        # Convert text to base bytes
        ids = list(text.encode("utf-8"))
        # Naive implementation: iterate sequentially through all merges
        for pair, new_id in self.merges.items():
            ids = self._merge_ids(ids, pair, new_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        # Reconstruct byte sequence and decode via UTF-8 (handling invalid/split bytes gracefully)
        byte_segments = [self.vocab.get(i, b'') for i in ids]
        raw_bytes = b"".join(byte_segments)
        return raw_bytes.decode("utf-8", errors="replace")
```

## 4. Hardware Realities & Compute/Memory Accounting
- **Memory Overhead:** Not covered in this lecture. *(Note: Low-level GPU memory details for models are introduced starting in Lecture 2/3).*
- **Hardware Efficiency (MFU):**
  - **Memory-Bound Decoding:** The prefill phase of inference is highly parallelized and compute-bound (similar to training), whereas the token decoding phase (generating tokens one-by-one) is severely **memory-bound**. This is because each single token generation requires loading the entire model parameters and KV cache from High-Bandwidth Memory (HBM) to SRAM.
  - **GPU Specs Reference:** The upcoming NVIDIA H100 successor (B200 GPU) features a massive **2.25 PFLOPS** of compute in BF16 precision, balanced against a memory bandwidth of **8 TB/s**. This wide disparity underscores why memory throughput, and not just peak FLOPs, is the dominant bottleneck in practical model deployment.
  - **Data Movement Minimization:** Standard PyTorch operations launch individual GPU kernels, forcing expensive round-trips of intermediate results back to HBM. Hardware-efficient systems leverage **Operator Fusion** to load data once, perform multiple consecutive operations (e.g., $A \cdot B$ followed by an activation), and write the final result back once, cutting HBM bandwidth demands in half.

## 5. Visualization Blueprint (BPE Iterative Compression Trace)
- **Visualization Type:** Sequential Bar Chart & Sequence Shrinkage Timeline.
- **Data Fields & Encoding:**
  - **X-Axis:** Character index in the raw string.
  - **Y-Axis:** Merging Step index ($m = 0, 1, 2, \dots$).
  - **Color:** Unique color fill representing each distinct token ID (IDs 0–255 share a neutral palette; newly merged token IDs $\ge 256$ use vibrant, high-contrast colors).
  - **Size / Box Width:** The span of raw characters represented by a single token. As merges occur, adjacent boxes merge together horizontally to visually demonstrate the compression of the string representation.
- **Interactive Controls (if built):**
  - **Time-step Slider:** Scrubbing from $0$ to $M$ allows the user to see the exact sequential merges of "the cat in the hat" in real-time.
  - **Text input Box:** Let users input any string and witness how it is segmented step-by-step based on the learned merge rules.
  - **Vocab Size Dropdown:** Toggles between small toy vocabs (e.g., size 300) and standard sizes (e.g., 32k) to see how deep the compression tree goes.

## 6. Empirical Scaling Laws & Hyperparameter Heuristics
- **Chinchilla Scaling Rule of Thumb:**
  To optimize a pre-training compute budget, the optimal parameter-to-token allocation ratio is approximately 1:20.

  $\text{Optimal Token Count (D)} \approx 20 \times \text{Parameter Count (N)}$

  For example, a $70\text{B}$ parameter model should be trained on roughly $1.4\text{T}$ tokens to be compute-optimal.
- **Inference-Driven Deviation:**
  Modern production models (e.g., LLaMA, DeepSeek, Qwen) deliberately violate the Chinchilla optimal ratio by training models far beyond the 1:20 limit (using tens of trillions of tokens on smaller parameter sizes). While this is compute-suboptimal during pre-training, it pays off massive dividends by minimizing downstream inference latency and compute costs.

## 7. Systems Warnings, Pitfalls, & Reflection Questions
- **Gotchas & Common Bugs:**
  - **Lossless Tokenization (Round-Trip Failure):** Ensure your tokenizer satisfies the round-trip invariant: $\text{decode}(\text{encode}(x)) \equiv x$. Failure to properly preserve raw byte chunks, especially with special tokens or malformed UTF-8 inputs, will result in silent corpus corruption.
  - **Naive $O(V \cdot L)$ Encoder Bottleneck:** Implementing the BPE encoder by running a sequential loop over all merges is extremely slow. Since modern multilingual vocabularies scale to $100\text{k}-200\text{k}$ tokens, a naive encoder will stall your data pipeline. To fix this:
    1. Pre-split documents into smaller regex-based chunks (e.g., by space/punctuation) to limit the sequence length $L$ per tokenization thread.
    2. Maintain an index of active pairs and only scan rules that are relevant to the current chunk.
  - **Whitespace Inconsistency:** Space handling is highly sensitive in BPE. "hello" and " hello" (with a preceding space) are processed as two entirely unrelated token IDs, meaning the model's semantic representations of them can be completely disconnected.
- **Conceptual Questions:**
  1. *How does the choice of vocabulary size ($V$) in BPE impact the trade-off between transformer sequence length and softmax projection layer computation?*
     - **Answer:** A larger $V$ increases the compression ratio (higher bytes-per-token), resulting in shorter token sequence lengths $L$, which quadratically reduces attention compute overhead ($O(L^2)$) and context window consumption. However, a larger $V$ increases the parameters in the embedding and output projection layers ($V \times D\_{model}$) and adds substantial compute overhead during the final softmax projection over $V$ options.
  2. *Why is the predictability of a scaling recipe considered as critical as its absolute optimality?*
     - **Answer:** When training at massive scales (e.g., $10^{25}$ FLOPs, costing hundreds of millions of dollars), hyperparameter tuning is impossible. You only get one shot. A scaling recipe must ensure hyperparameter transfer (where hyperparameters are predictable functions of scale) so that the final model's behavior is highly predictable from cheap, small-scale pilot runs. If hyperparameters fluctuate wildly between scales, predicting larger-scale behavior is impossible, creating extreme risk of complete failure.
