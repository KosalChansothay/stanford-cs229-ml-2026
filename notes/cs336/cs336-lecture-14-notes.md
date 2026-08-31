# CS336 Lecture 14: Data (Preprocessing & Mixing)

## 0. Quick-Recall Summary
- **HTML-to-Text Conversion**: HTML text extraction is a lossy process of linearizing structural DOMs into sequences, requiring rule-based filters to prune ads and boilerplate.
- **MinHash Jaccard Estimation**: Near-deduplication Jaccard similarity is estimated by taking the minimum hash of shingled document sets: $P(h(A) = h(B)) = \text{Jaccard}(A, B)$.
- **LSH (Locality Sensitive Hashing)**: Groups $n = b \cdot r$ hashes into $b$ bands of $r$ rows to create an S-curve threshold, isolating near-duplicates in linear time.
- **RegMix (Regression-Based Mixing)**: Employs a swarm of cheap proxy runs to fit a linear regression from mixture weights to target validation loss, mathematically optimizing pre-training data mixtures.
- **Simulated Epoching**: Downsamples sources proportionally during small-scale proxy sweeps to mimic the data-scarcity and overfitting behaviors of full-scale runs.

## 1. Core Paradigm & Systems Overview
- **Objective**: Clean, structure, deduplicate, and mix web-crawled documents to form the ultimate training dataset while managing overfitting bounds under finite high-quality text constraints.
- **Primary Bottleneck**: Structural noise and data scarcity. Raw web crawls are full of low-quality boilerplate and massive duplicate patterns (e.g. mirror sites); without deduplication, massive parameter runs overfit early.
- **Builds on**: Legal and Source Boundaries (Lecture 13) — moves from raw data collection and ethics to programmatic pipeline execution.

## 2. Theoretical & Mathematical Primitives

### Jaccard Similarity
For two document token sets $A, B$, Jaccard Similarity measures overlap as:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

### MinHash Collision Theorem
Given a random permutation hash function $\pi$, the probability of a MinHash collision is exactly the Jaccard similarity:

$$P(\min(\pi(A)) = \min(\pi(B))) = J(A, B)$$

### LSH S-Curve Formulation
For $b$ bands of $r$ rows, the probability of document collision under LSH is derived as:

$$P\_{\text{collision}}(s) = 1 - (1 - s^r)^b$$

Where $s = J(A, B)$. This results in a sharp step-function transition at the threshold:

$$s\_{\text{threshold}} \approx \left(\frac{1}{b}\right)^{1/r}$$

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### MinHash LSH Deduplication Loop
1. Parse document into a set of shingles (e.g., character 5-grams).
2. For $i$ from 1 to $b \cdot r$: compute MinHash signatures.
3. Group the signature vector into $b$ bands, each with $r$ rows.
4. Hash each band separately to bucket indices.
5. Documents that share any bucket are candidate duplicates.
6. Verify candidate Jaccard similarity directly and prune.

### PyTorch/Pythonic Blueprint (Educational MinHash Signature Generator)
```python
import numpy as np

def generate_minhash_signature(shingle_set, num_hashes=128):
    # Map shingles to 32-bit integers
    shingle_hashes = np.array([hash(s) & 0xffffffff for s in shingle_set], dtype=np.uint32)
    if len(shingle_hashes) == 0:
        return np.ones(num_hashes, dtype=np.uint32) * 0xffffffff
        
    signature = []
    for seed in range(num_hashes):
        # Generate hash permutation coefficients pseudo-randomly
        np.random.seed(seed)
        a = np.random.randint(1, 2**31-1, dtype=np.int64)
        b = np.random.randint(0, 2**31-1, dtype=np.int64)
        
        # Linear permutation modulo prime
        permuted = (a * shingle_hashes + b) % 2147483647
        signature.append(np.min(permuted))
        
    return np.array(signature, dtype=np.uint32)
```

## 4. Hardware Realities & Compute/Memory Accounting
- **The $N^2$ Parallelization Trap**: Deduplicating 10 billion web pages is an $O(N^2)$ problem if done naively. MinHash LSH maps the problem into $b$ independent, parallelizable bucketing processes, scaling deduplication to linear $O(N)$ time across large Spark or Ray clusters.

## 5. Hyperparameter Heuristics

<div id="plotly-cs336-14-lsh-scurve" class="plotly-chart" aria-label="Interactive Plotly chart: MinHash LSH S-Curve"></div>

<p><em>Figure: MinHash + LSH S-Curve — Tuning bands $b$ and rows $r$ yields a sharp transition threshold for near-deduplicating web documents.</em></p>

<div id="plotly-cs336-14-epoch-caps" class="plotly-chart" aria-label="Interactive Plotly chart: Epoch Caps Validation Loss Degradation"></div>

<p><em>Figure: Epoch Caps — Repeating web corpora beyond 4 epochs triggers severe validation loss overfitting.</em></p>

- **Epoch Caps**: Under WSD and data constraints, repeating high-quality sources like Wikipedia more than **4 epochs** causes severe validation loss overfitting and performance degradation.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **Span-based Shredding**: Exact deduplication via short sentence-span matches (such as C4's 3-sentence exact matching) strips duplicate strings but tears apart document coherence, leaving ungrammatical, fragmented sentences in the pre-training set.

### Conceptual Reflection Questions
1. *Why does simulated epoching outperform simple proportional downsampling for proxy data sweeps?*
   **Answer**: In proportional downsampling, if a 1-trillion-token run has 10 billion tokens of Wikipedia, the 10-billion-token proxy run gets only 100 million tokens of Wikipedia, which never overfits. However, in the full run, Wikipedia will be repeated multiple times. Simulated epoching explicitly caps or downsamples Wikipedia to its scarcity limit at the proxy scale, forcing the small proxy model to overfit on Wikipedia, thus aligning its loss behavior with the full-scale run.

2. *Explain the trade-off of choosing a high value for row parameter $r$ in LSH.*
   **Answer**: High $r$ values require more hashes to match within a band to trigger a collision, shifting the S-curve threshold to the right. This reduces false positives (unrelated documents colliding) but increases false negatives, allowing near-duplicates with slight formatting changes to slip through undetected.
