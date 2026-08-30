# Stanford CS229: Machine Learning (Spring 2026)
## Lecture 13: LLMs, Next-Word Prediction Loss

### 1. Summary
This lecture marks the transition from visual generative models (such as Diffusion) into the world of **Foundation Models** and large-scale language systems. The instructor, Tengyu, introduces this paradigm shift by looking at the "pre-GPT" technology that underpins modern architectures: **Representation Learning** and **Semantic Search**. He explores how high-dimensional data (such as text, images, or code) can be mapped into a structured semantic embedding space. 

The lecture details the mechanics of **Supervised Pre-training** (and its limitations) and contrasts it with **Contrastive Learning** (specifically the **SimCLR** framework). Finally, the session explores how these learned embeddings are deployed in industry, analyzing the mathematical workflows of **Semantic Vector Search** and **Retrieval-Augmented Generation (RAG)**. RAG is evaluated as a highly modular, privacy-preserving, and cost-effective alternative to fine-tuning for enterprise applications.

---

### 2. Key Concepts & Definitions
- **Foundation Model**: A massive, general-purpose model pre-trained on diverse, unstructured web-scale corpora (often several orders of magnitude larger than traditional datasets) which can be adapted to a wide variety of downstream tasks.
- **Pre-training and Adaptation**: A two-phase paradigm. In *pre-training*, a model learns from massive unlabeled data. In *adaptation*, the model is customized to a downstream task (historically via few-shot/fine-tuning, and modernly via zero-shot prompting).
- **Embedding / Representation**: A mapping function $\phi\_\theta(x)$ that translates raw input data $x \in \mathcal{X}$ into a dense continuous vector in an $m$-dimensional Euclidean space $\mathbb{R}^m$, preserving semantic relationships.
- **Supervised Pre-training**: Learning embeddings by training a model on a large-scale classification task (e.g., ImageNet) and extracting the activations of the penultimate (last-but-one) layer as the features.
- **Contrastive Learning**: A self-supervised paradigm that trains embeddings without labels by encouraging positive pairs (different views of the same image or paired modalities) to be close in vector space, while forcing negative pairs (distinct data points) to be far apart.
- **Degenerate Collapse**: A failure state in self-supervised learning where the encoder maps all inputs to a single constant vector to trivially minimize positive-pair distance. This is prevented by introducing a repulsive force via negative pairs.
- **Data Augmentation**: Algorithmic transformations (such as random cropping, flipping, color jitter, blurring, or adding noise) that change the pixel representation of an image without altering its core semantic meaning.
- **Positive Pair**: Paired representations representing the same semantic entity (e.g., two different random augmentations of the same image, or a document's title and its body text).
- **Negative Pair**: Representations representing different entities, used to prevent collapse (e.g., augmentations of two completely different images).
- **Negative Pair Mining (Hard Negatives)**: The practice of filtering datasets to identify negative examples that look highly similar but are semantically distinct, making the contrastive loss function more demanding and effective.
- **Retrieval-Augmented Generation (RAG)**: A framework that dynamically retrieves relevant documents from an external corpus at test time and appends them to the prompt of an LLM to generate highly accurate, context-grounded answers.
- **Approximate Nearest Neighbors (ANN) & Vector Databases**: Specialized storage and retrieval engines that enable extremely fast k-nearest-neighbor search over high-dimensional vectors, bypassing brute-force $O(N)$ scans.

---

### 3. Mathematical Formulations & Derivations

#### A. Supervised Feature Extraction (The Penultimate Layer)
In a traditional supervised setup, a network is trained to classify images into one of $K$ discrete classes (such as ImageNet's 1,000 classes). The architecture takes an input $x$, processes it through multiple non-linear layers, and yields a penultimate vector representation:
$$
\phi\_\theta(x) \in \mathbb{R}^m
$$
A final classification weight matrix $W \in \mathbb{R}^{K \times m}$ is applied to produce the raw logits, which are mapped to class probabilities via softmax:
$$
\hat{y} = \text{softmax}(W \phi\_\theta(x))
$$
After pre-training on ImageNet, the final classification head $W$ is discarded. The parameter set $\theta$ is saved as the deliverable, and $\phi\_\theta(x)$ is extracted as the data's semantic embedding.

*Limitations of the Supervised Approach*: 
1. **Feature Diversity Bottleneck**: If the labeling task has too few classes (e.g., MNIST digits 0-9) or a simple binary objective (e.g., classifying black-and-white vs. color), the network only learns the narrow set of features needed for that task, ignoring broader semantic structures.
2. **Labeling Costs**: Scaling to larger models requires a proportional scale-up in labels, which becomes prohibitively expensive.

---

#### B. Self-Supervised Contrastive Learning (The SimCLR Framework)
To eliminate the need for human-annotated labels, self-supervised systems use **contrastive objectives**.

##### 1. Data Processing and Augmentation Workflow
For each training batch of size $B$, the system samples $B$ images from the corpus. For each image $x\_i$, two random data augmentations are generated:
$$
\hat{x}\_i \sim \mathcal{A}(x\_i), \quad \tilde{x}\_i \sim \mathcal{A}(x\_i)
$$
This produces a processed batch of $2B$ total augmented images. The images are mapped through the network to generate normalized embeddings:
$$
u\_i = \frac{\phi\_\theta(\hat{x}\_i)}{\|\phi\_\theta(\hat{x}\_i)\|}, \quad v\_i = \frac{\phi\_\theta(\tilde{x}\_i)}{\|\phi\_\theta(\tilde{x}\_i)\|}
$$
Using normalized embeddings ensures that their inner product is equivalent to the **cosine similarity**:
$$
S\_{ij} = u\_i^T v\_j = \text{cosine\_similarity}(\phi\_\theta(\hat{x}\_i), \phi\_\theta(\tilde{x}\_j))
$$
##### 2. The Contrastive Loss Function (NT-Xent / InfoNCE style)
For a given anchor image $i$, we want to encourage the positive pair $(u\_i, v\_i)$ to be close, while forcing all other $2B - 2$ negative pairings in the batch to be far apart. The loss for column $i$ is formulated as a multi-class softmax classification problem:
$$
\mathcal{L}\_i = -\log \frac{\exp(S\_{ii} / \tau)}{\exp(S\_{ii} / \tau) + \sum\_{j \neq i} \exp(S\_{ij} / \tau)}
$$
where $\tau$ is a temperature hyperparameter. The total batch loss is the sum over all columns:
$$
\mathcal{L}\_{\text{batch}} = \sum\_{i=1}^B \mathcal{L}\_i
$$
##### 3. Monotonicity and Force Balance Proof
Let us simplify the loss for a single anchor by denoting the positive pair similarity term as $A = \exp(S\_{ii} / \tau) > 0$ and the sum of the negative pair similarities as $B = \sum\_{j \neq i} \exp(S\_{ij} / \tau) > 0$. The loss is:
$$
\mathcal{L} = -\log \left(\frac{A}{A + B}\right) = \log(A + B) - \log A
$$
To evaluate the mathematical dynamics of the optimizer, we compute the partial derivatives with respect to $A$ and $B$:
- **Derivative with respect to positive similarity ($A$)**:
  

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{1}{A + B} - \frac{1}{A} = \frac{A - (A + B)}{A(A + B)} = \frac{-B}{A(A + B)}
$$
  Since both $A > 0$ and $B > 0$, we have:
  

$$\frac{\partial \mathcal{L}}{\partial A} < 0
$$
  This negative gradient proves that the loss function is monotonically decreasing with respect to positive similarity $A$. Therefore, minimizing $\mathcal{L}$ mathematically forces the optimizer to **maximize positive-pair similarity**.

- **Derivative with respect to negative similarity ($B$)**:
  

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{1}{A + B} > 0
$$
  Since $A + B > 0$, we have:
  

$$\frac{\partial \mathcal{L}}{\partial B} > 0
$$
  This positive gradient proves that the loss function is monotonically increasing with respect to negative similarity $B$. Therefore, minimizing $\mathcal{L}$ mathematically forces the optimizer to **minimize negative-pair similarity (maximizing their distance)**.


<div id="plotly-13-contrastive-loss" class="plotly-chart" aria-label="Interactive Plotly chart: NT-Xent loss versus positive similarity"></div>

<p><em>Figure: The NT-Xent loss decreases monotonically with positive-pair similarity — the attract force — and grows with the number and similarity of negatives — the repel force.</em></p>
<div id="plotly-13-embedding-space" class="plotly-chart" aria-label="Interactive Plotly chart: SimCLR force balance in embedding space"></div>

<p><em>Figure: SimCLR force balance in embedding space.</em></p>

##### 4. The Challenge of Semantically Similar Negatives
Because negative pairs are chosen at random from the batch, some negative examples might belong to the same semantic class (e.g., the anchor is a cat, and a random negative is a different cat image). In this case, the loss function acts counterproductively by pushing similar representations apart.
- *Why this is acceptable*: On diverse, large-scale web datasets, the probability of sampling two highly similar classes at random is extremely small (e.g., you are far more likely to pair a cat with an airplane or a chair). The rare negative matches act as a tiny amount of noise, but the dominant training signals successfully align the semantic boundaries.

---

### 4. Step-by-Step Modern NLP Workflows

#### A. Document Representation Learning
Because text cannot be cropped or flipped like pixels, positive pairs in language are constructed using document structure:
1. **Positive Pair Formulation**: For a given document $D\_i$, define the first view ($\hat{x}\_i$) as the document's **Title and Header sections**, and the second view ($\tilde{x}\_i$) as the **raw body text**. 
2. **Negative Pair Formulation**: Pair the title of document $D\_i$ with the body text of document $D\_j$ ($j \neq i$) to act as negative examples.
3. **Training**: Train the shared text encoder $\phi\_\theta$ on these pairs using the NT-Xent loss to map titles and relevant bodies to nearby points in Euclidean space.

---

#### B. Semantic Vector Search
Once trained, the embeddings are used to perform semantic retrieval:
1. **Precomputation (Offline Phase)**: Take your corpus of documents $\{d\_1, \dots, d\_N\}$ and run them through the frozen encoder to compute and store their embeddings:

$$v\_i = \phi\_\theta(d\_i) \quad \forall i \in \{1, \dots, N\}
$$
2. **Query Processing (Online Phase)**: When a user enters a query $q$ at test time, compute its query embedding:

$$u\_q = \phi\_\theta(q)
$$
3. **Nearest Neighbor Search**: Solve for the document ID $i^*$ that maximizes the inner product:

$$i^* = \text{argmax}\_{i} \left( u\_q^T v\_i \right)
$$
4. **Systems Scale (ANN)**: For massive corpora ($N > 10^9$), brute-force dot products are too slow. We index the precomputed vectors in a **Vector Database** using **Approximate Nearest Neighbors (ANN)** algorithms to retrieve match candidates in sub-millisecond time.

---

#### C. Retrieval-Augmented Generation (RAG)
RAG is the primary mechanism for combining retrieval systems with large language models at test time:

<div id="plotly-13-rag-architecture" class="plotly-chart" aria-label="Interactive Plotly chart: RAG system architecture"></div>

<p><em>Figure: RAG system architecture.</em></p>

1. **User Query**: A user submits a query $q$ containing a specialized question.
2. **Embedding and Match**: The query is mapped to $\phi\_\theta(q)$ and sent to the Vector Database containing pre-embedded enterprise document chunks.
3. **Context Retrieval**: The database returns the top $k$ (e.g., 5 to 10) most semantically similar documents $\{d\_1, \dots, d\_k\}$.
4. **Prompt Augmentation**: The system compiles a prompt containing the retrieved context and the user query:
   ```
   [Context Documents]
   Document 1: [Text of d\_1]
   Document 2: [Text of d\_2]
   ...
   [User Query]
   Based on the context documents above, please answer: q
   ```
5. **Grounded Generation**: The LLM processes the prompt in-context and generates the final factual answer, bypassing the need to modify its internal parameters.

---

### 5. Architectural & Systems Trade-offs: RAG vs. Fine-Tuning

When integrating proprietary data into enterprise applications, developers choose between **RAG** and **Fine-Tuning**.

| Evaluation Vector | Retrieval-Augmented Generation (RAG) | Model Fine-Tuning (FT) |
| :--- | :--- | :--- |
| **Data Governance & Permissions** | **Highly Secure**: Document filtering is handled dynamically during the retrieval phase, allowing different access permissions (e.g., standard employees are prevented from retrieving confidential executive memos). | **Insecure**: Once fine-tuned, model parameters are shared. Standard users can potentially bypass safety filters to extract confidential training data via prompt injection. |
| **Information Updating & Deletion** | **Instantaneous**: To update or delete a fact, simply edit or remove the document chunk from the vector database. It will immediately cease to be retrieved. | **Prohibitively Hard**: Updating facts requires a full retraining run. Deleting learned knowledge from neural network parameters ("machine forgetting") is an unsolved research problem. |
| **Compute & Serving Costs** | **Highly Economical**: Bypasses expensive training runs. Bypasses the need to host custom, user-specific weights on dedicated GPU infrastructure. | **Highly Expensive**: Requires high GPU FLOPs for parameter gradient steps and expensive infrastructure to host and serve separate customized models. |
| **Modularity & Scalability** | **Decoupled**: The retrieval database is independent of the generator LLM. One can swap out the vector database or upgrade the base LLM on the fly without affecting the other. | **Monolithic**: The knowledge is baked directly into the model weights. Upgrading the base LLM requires throwing away the fine-tuned model and retraining from scratch. |

---

### 6. SOTA / Frontier Retrieval Engineering: Hybrid Search
While semantic vector search is the baseline for text retrieval, modern high-performing code-generation systems (such as Anthropic's Claude Code) utilize **Hybrid Retrieval architectures**:
- **Structured Codebases**: Codebases have highly rigid, explicit directories, variables, and function names. Vector embeddings can occasionally struggle with exact keyword lookups or variable tracing.
- **The LLM-Regex Pattern**: Rather than relying purely on vector cosine similarity, frontier systems use the LLM to dynamically generate **Regular Expressions (regex)** and keyword queries at test time.
- **Execution**: The system executes regex pattern matching over the codebase's file tree to find exact variables, imports, and function signatures with 100% precision, combining this with semantic search to provide the most complete context possible to the generator.

---

### 7. Reflection Questions
1. **Explain the Collapse Phenomenon**: Why does a self-supervised representation learning network collapse (i.e., map all inputs to a constant vector) if its loss function only minimizes the distance between positive pairs? How does the SimCLR denominator mathematically prevent this failure mode?
2. **RAG Data Governance**: Suppose an enterprise has strict role-based access control (RBAC) where standard employees must not see financial earning projection memos. Explain why a RAG architecture can easily enforce this policy, whereas a fully fine-tuned model struggles to do so.
3. **The Limits of Supervised Penultimate Embeddings**: Explain why extracting features from a model trained on a simple classification dataset (e.g., MNIST with only 10 classes) fails to produce a robust general-purpose semantic representation compared to a self-supervised contrastive model trained on unlabeled web data.

---

### 8. Further Reading & Resources
- **SimCLR Paper (Chen et al., 2020)**: *A Simple Framework for Contrastive Learning of Visual Representations*. The foundational framework demonstrating how self-supervised contrastive learning with simple data augmentations matches or exceeds supervised pre-training.
- **RAG Paper (Lewis et al., 2020)**: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. This paper introduces the hybrid architecture of combining dense vector retrieval with pre-trained seq2seq models.
- **The Stanford Foundation Models Survey (Bommasani et al., 2021)**: The seminal survey paper that formally defined and studied the emergence of "Foundation Models" across vision, language, and robotics.
