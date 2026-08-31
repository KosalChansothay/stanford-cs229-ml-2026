# CS336 Lecture 13: Data (Sources & Datasets)

## 0. Quick-Recall Summary
- **Data as Secret Sauce**: Web-scale training data is the primary differentiator of frontier LLMs; proprietary filtering and mixtures are guarded closely due to competitive advantages and legal liabilities.
- **Web Crawling Realities**: Standard web text is crawled via Common Crawl, but access is highly constrained by terms of service, robots.txt, Cloudflare anti-bot blocks, and walled gardens (Facebook, LinkedIn).
- **The Shadow Library Legal Cliff**: Early models trained on pirate book shadow libraries (Books3/Bibliotik). Copyright crackdowns have ended this practice, forcing developers to utilize public-domain alternatives like Project Gutenberg.
- **CCNet Wikipedia Filter**: Replaces manual rules with language identification (fastText) and perplexity-based filtering (using an n-gram model trained on Wikipedia) to retain only highly coherent, encyclopedic text.
- **DCLM Quality Filtering**: The DCLM (Data Comp) pipeline filters down massive 240-trillion-token pools by training fast bag-of-words classifiers on positive high-quality sources (OpenHermes, ELI5) vs. raw web pools.

## 1. Core Paradigm & Systems Overview
- **Objective**: Ingest the messy, dynamic public web and extract a high-quality, diverse pre-training corpus while navigating legal copyrights and technical crawler limitations.
- **Primary Bottleneck**: Quality at scale. While raw text is plentiful, high-quality, informative documents are highly finite, requiring aggressive filtering without introducing severe representation biases.
- **Builds on**: Pre-training basics (Lecture 1) — expands the tokenization stage to study what physical source materials are actually compiled to build the vocabulary.

## 2. Theoretical & Mathematical Primitives

### Web Crawling Graph Traversal
Web crawling is modeled as a parallel graph traversal over the web-link structure:

$$G = (V, E)$$

Where:
- $V$: Web page URLs (documents).
- $E$: Hyperlinks.

The crawler maintains a queue $Q$, popping URLs to fetch raw HTML responses (saved to WARC files), extracting nested hyperlinks to populate $Q$, while enforcing strict rate-limits per IP to prevent Denials of Service (DoS).

### Fair Use Evaluation Rubric (Copyright Law)
Under Section 107 of the US Copyright Act of 1976, fair use of copyrighted material is analyzed via four statutory factors:
1. **Purpose and character of use**: Whether it is transformative (e.g., training a model to learn high-level language statistics rather than copying creative expression).
2. **Nature of the copyrighted work**: Creative works receive higher protection than factual documents.
3. **Amount and substantiality of the portion used**: Training uses 100% of the text, but the output representations are highly compressed.
4. **Effect of the use upon the potential market**: Whether the model serves as a market substitute for the original work.

## 3. From-Scratch Algorithmic Workflows & Pseudocode

### BPE Tokenizer Expansion & Data Extraction Pipeline
1. Fetch raw web data as WARC files (containing raw HTTP headers and payload envelopes).
2. Parse HTML to plain text, discarding boilerplate tags, headers, and navigation scripts.
3. Apply CCNet-style language identification: keep only documents matching target languages with confidence $> 0.8$.
4. Run exact and fuzzy document deduplication (e.g., MinHash) to prune mirror sites.
5. Apply classifier-based quality filters to retain high-signal text.

### PyTorch/Pythonic Blueprint (WARC HTML Parser Sketch)
```python
import re

# Educational regex-based HTML text extractor to mimic raw WARC processing
def parse_html_to_clean_text(html_content):
    # 1. Strip script and style tags
    clean_html = re.sub(r'<(script|style).*?>.*?</>', '', html_content, flags=re.DOTALL)
    # 2. Strip standard HTML tags
    clean_text = re.sub(r'<.*?>', '', clean_html)
    # 3. Normalize whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Simple rule-based filter: discard if too few characters or contains boilerplate
    if len(clean_text) < 100 or "Terms of Service" in clean_text:
        return None
    return clean_text
```

## 4. Hardware Realities & Compute/Memory Accounting
- **WARC to WET Footprint**: Common Crawl dumps are massive. A single monthly crawl WARC file (storing raw HTML/headers) can consume hundreds of terabytes. Extracting them down to WET files (plain text) reduces the storage requirements by over **10x**, which is essential to keep memory loading times fast during pre-training data streaming.

## 5. Hyperparameter Heuristics

<div id="plotly-cs336-13-data-waterfall" class="plotly-chart" aria-label="Interactive Plotly chart: Pre-training Data Filtering Waterfall"></div>

<p><em>Figure: Pre-training Data Filtering Funnel — Quality filtering and deduplication extract high-signal tokens from petabytes of Common Crawl.</em></p>

<div id="plotly-cs336-13-domain-mix" class="plotly-chart" aria-label="Interactive Plotly chart: Pre-training Domain Token Mixture"></div>

<p><em>Figure: Frontier Pre-training Token Mixture — Balancing general web text, code, math/STEM, and synthetic reasoning data.</em></p>

- **Karma Filter (webtext)**: OpenAI's WebText filtered out Reddit outgoing links to pages that had fewer than **3 karma**, leveraging social curation as an automated filter for human readability.

## 6. Systems Warnings, Pitfalls, & Reflection Questions

### Gotchas & Common Bugs
- **Wikipedia Poisoning**: Because Wikipedia releases dumps on a strict periodic schedule, malicious attackers can edit critical articles with toxic triggers right before a known dump timestamp. The vandalism is quickly reverted by Wiki bots, but the frozen pre-training dump retains the poisoned triggers.

### Conceptual Reflection Questions
1. *Why did C4 choose to filter out curly braces (`{`), and what were the architectural implications of this choice?*
   **Answer**: Google's C4 dataset aimed to compile high-quality, natural English prose. Because JavaScript and CSS files are saturated with curly braces, filtering out any line containing `{` served as a highly effective rule to remove raw code and web templates from the text corpus. However, this architectural design decision also stripped out nearly all programming language data, rendering subsequent models (like T5) highly suboptimal at code execution tasks.

2. *Why does the legal precedent of "transformative use" (from Google Books) serve as a foundational defense for AI pre-training?*
   **Answer**: In the Authors Guild v. Google lawsuit, the court ruled that Google's scanning of millions of copyrighted books to build a search index and snippet-view tool was "fair use" because it was transformative. It did not seek to re-sell or replicate the books as creative expressions, but rather to analyze search and language statistics. LLM pre-training uses this exact defense: training compresses creative text to learn linguistic statistical structures, making the model a transformative tool rather than a verbatim pirate distributor.
