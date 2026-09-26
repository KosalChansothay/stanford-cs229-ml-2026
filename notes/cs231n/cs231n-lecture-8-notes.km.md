# កំណត់ចំណាំមេរៀន: មេរៀនទី ៨: យន្តការផ្ដោតអារម្មណ៍ និងស្ថាបត្យកម្ម Transformers (Attention & Transformers)

**វគ្គសិក្សា**: Stanford CS231n: Deep Learning for Computer Vision (ការរៀនស៊ីជម្រៅសម្រាប់ការមើលឃើញដោយកុំព្យូទ័រ — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Prof. Fei-Fei Li (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Ranjay Krishna (សាស្ត្រាចារ្យជំនួយផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs231n.stanford.edu](https://cs231n.stanford.edu) (ស្លាយមេរៀន ឯកសារសិក្សា និងកិច្ចការអនុវត្ត)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Quick-Recall Summary)

* **បញ្ហារាំងស្ទះនៃការបង្រួមព័ត៌មាន (The Communication Bottleneck)**: ស្ថាបត្យកម្មបែបចាស់ Seq2Seq RNNs បង្ខំឱ្យព័ត៌មានទាំងអស់នៃលំដាប់ធាតុចូល (Input Sequence) ត្រូវបង្រួមចូលទៅក្នុងវ៉ិចទ័របរិបទថេរតែមួយ $C$ (Hidden State ចុងក្រោយនៃ Encoder) ដែលបង្កើតជាចំណុចស្ទះព័ត៌មានយ៉ាងធ្ងន់ធ្ងរនៅពេលប្រវែងលំដាប់ $N$ កើនឡើងវែង។
* **វ៉ិចទ័របរិបទឌីណាមិក (Dynamic Context Vectors via Attention)**: យន្តការផ្ដោតអារម្មណ៍ (Attention Mechanism) ដោះស្រាយបញ្ហារាំងស្ទះនេះ ដោយអនុញ្ញាតឱ្យ Decoder អាច "ក្រឡេកមើលថយក្រោយ" ទៅកាន់គ្រប់ Hidden States ទាំងអស់របស់ Encoder នៅរាល់ជំហាននៃការបង្កើតទិន្នន័យ ដោយគណនាជាផលបូកទម្ងន់ដេរីវេ (Differentiable Weighted Sum) នៃលក្ខណៈពិសេសដែលពាក់ព័ន្ធបំផុត។
* **Scaled Dot-Product Attention**: ដើម្បីប្រែក្លាយ Attention ឱ្យទៅជាស្រទាប់ប្រមាណវិធីឯករាជ្យដែលគណនាស្របគ្នាលើ GPU កម្រិតស្រដៀងគ្នាត្រូវបានគណនាតាមរយៈផលគុណស្កាលែរនៃម៉ាទ្រីស $Q K^T$។ ដើម្បីទប់ស្កាត់បញ្ហាជម្រាលសាបរលាបក្នុងលំហពហុវិមាត្រ ($d_k$) ផលគុណនេះត្រូវបានបង្រួមដោយការចែកនឹង $\sqrt{d_k}$ មុនពេលអនុវត្ត Softmax។
* **ភាពស៊ីមេទ្រីប្តូរលំដាប់ (Permutation Equivariance)**: យន្តការ Self-Attention ដំណើរការលើ **សំណុំ (Sets)** នៃវ៉ិចទ័រ មិនមែនលើលំដាប់លីនេអ៊ែរឡើយ។ ប្រសិនបើលំដាប់ធាតុចូលត្រូវបានច្របល់ នោះ Output ក៏ត្រូវបានច្របល់តាមទម្រង់ដូចគ្នាបេះបិទ។ ហេតុនេះ ទើបយើងត្រូវបូកបញ្ចូល **Positional Embeddings** ជាដាច់ខាត ដើម្បីបញ្ជាក់ពីទីតាំងភូមិសាស្ត្រ ឬលំដាប់នៃពាក្យ។
* **ស្ថាបត្យកម្ម Transformer**: ការរួមបញ្ចូលគ្នារវាង Multi-Head Self-Attention (ដែលអនុញ្ញាតឱ្យគ្រប់ Token ពិភាក្សាគ្នាទៅវិញទៅមកជាសកល) ជាមួយ Position-wise Feed-Forward Networks (ដែលដំណើរការលើ Token នីមួយៗដោយឯករាជ្យ) ព្រមទាំង Residual Skip Connections និង Layer Normalization បានបង្កើតជាម៉ាស៊ីនគ្រឹះនៃបដិវត្តន៍ AI ទំនើប (LLMs & Vision Transformers)។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

### ក. បញ្ហារាំងស្ទះនៃ Seq2Seq RNN (The Seq2Seq Compression Bottleneck)
នៅក្នុងស្ថាបត្យកម្ម Seq2Seq ផ្អែកលើ RNN បុរាណ Encoder បង្រួមលំដាប់ធាតុចូលប្រវែង $T_x$ ទៅជាវ៉ិចទ័រថេរតែមួយ៖
$$
C = h_{T_x}
$$
Decoder ត្រូវពឹងផ្អែកលើតែវ៉ិចទ័រ $C$ នេះដើម្បីបង្កើតលំដាប់ Output ទាំងមូល។ ប្រសិនបើប្រយោគមាន ១០០ ពាក្យ ព័ត៌មាននៅពាក្យដំបូងៗនឹងត្រូវរលុបបាត់បង់ស្ទើរតែទាំងស្រុង ដែលធ្វើឱ្យគុណភាពនៃការបកប្រែធ្លាក់ចុះយ៉ាងខ្លាំង។

### ខ. យន្តការផ្ដោតអារម្មណ៍បែបឌីណាមិក (Attention Mechanism)
ជំនួសឱ្យការប្រើ $C$ ថេរ Decoder បង្កើតវ៉ិចទ័របរិបទថ្មី $c_t$ នៅរាល់ជំហាន $t$ តាមដំណាក់កាលទាំងបី៖
1. **ពិន្ទុតម្រឹម (Alignment Scores $e_{ti}$)**: វាស់កម្រិតភាពត្រូវគ្នារវាងស្ថានភាពបច្ចុប្បន្នរបស់ Decoder $s_{t-1}$ និងស្ថានភាពនីមួយៗរបស់ Encoder $h_i$៖
   $$
   e_{ti} = f_{\text{att}}(s_{t-1}, h_i)
   $$
2. **ទម្ងន់ផ្ដោតអារម្មណ៍ (Attention Weights $\alpha_{ti}$)**: ធ្វើ Normalization តាម Softmax ដើម្បីទទួលបានបំណែងចែកប្រូបាប៊ីលីតេ ($0 \le \alpha_{ti} \le 1$ និង $\sum_i \alpha_{ti} = 1$)៖
   $$
   \alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{T_x} \exp(e_{tj})}
   $$
3. **វ៉ិចទ័របរិបទឌីណាមិក (Dynamic Context Vector $c_t$)**: ជាផលបូកទម្ងន់នៃ Hidden States ទាំងអស់របស់ Encoder៖
   $$
   c_t = \sum_{i=1}^{T_x} \alpha_{ti} h_i
   $$

---

### គ. ទម្រង់ទូទៅនៃ Attention: Queries, Keys, and Values
នៅក្នុងស្ថាបត្យកម្មទំនើប Attention ត្រូវបានបកស្រាយដូចជាប្រព័ន្ធ Database Querying៖
* **Query ($Q$)**: តំណាងឱ្យអ្វីដែលយើងកំពុងស្វែងរក (What we are looking for)។
* **Key ($K$)**: តំណាងឱ្យស្លាកសម្គាល់នៃទិន្នន័យនីមួយៗ (Labels / Metadata of available information)។
* **Value ($V$)**: តំណាងឱ្យខ្លឹមសារព័ត៌មានពិតប្រាកដដែលត្រូវទាញយក (The actual content)។

---

## ៣. មូលដ្ឋានគ្រឹះគណិតវិទ្យា & រូបមន្ត (Mathematical Foundations & Formulations)

### ក. Scaled Dot-Product Attention
សម្រាប់ម៉ាទ្រីស Queries $Q \in \mathbb{R}^{N \times d_k}$, Keys $K \in \mathbb{R}^{M \times d_k}$, និង Values $V \in \mathbb{R}^{M \times d_v}$៖
$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

#### សម្រាយបញ្ជាក់គណិតវិទ្យានៃកត្តាបង្រួម $\sqrt{d_k}$ (Derivation of Scaling Factor)
ឧបមាថាសមាសភាគនីមួយៗនៃវ៉ិចទ័រ $q \in \mathbb{R}^{d_k}$ និង $k \in \mathbb{R}^{d_k}$ គឺជាអថេរចៃដន្យឯករាជ្យដែលមានមធ្យមភាគសូន្យ និងវ៉ារ្យ៉ង់ស្មើ ១ ($\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$ និង $\operatorname{Var}(q_i) = \operatorname{Var}(k_i) = 1$)។  
ផលគុណស្កាលែរគឺ $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$។  
* មធ្យមភាគនៃផលគុណ: $\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$
* វ៉ារ្យ៉ង់នៃផលគុណ:
  $$
  \operatorname{Var}(q \cdot k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left( \operatorname{Var}(q_i)\operatorname{Var}(k_i) + \mathbb{E}[q_i]^2\operatorname{Var}(k_i) + \mathbb{E}[k_i]^2\operatorname{Var}(q_i) \right) = \sum_{i=1}^{d_k} (1 \times 1 + 0 + 0) = d_k
  $$

**សេចក្តីសន្និដ្ឋាន**: នៅពេលវិមាត្រ $d_k$ កើនឡើងធំ (ឧ. $d_k = 64$ ឬ $128$) វ៉ារ្យ៉ង់នៃផលគុណស្កាលែរកើនឡើងដល់ $d_k$ (គម្លាតគំរូ $\sigma = \sqrt{d_k}$)។ តម្លៃធំៗទាំងនេះរុញច្រានអនុគមន៍ Softmax ឱ្យធ្លាក់ចូលទៅក្នុងតំបន់ឆ្អែតបំផុត (Saturated Tails) ដែលមានដេរីវេជិតសូន្យបេះបិទ បណ្តាលឱ្យជម្រាលសាបរលាប (Vanishing Gradients)។ ការចែកនឹង $\sqrt{d_k}$ បង្រួមវ៉ារ្យ៉ង់ឱ្យត្រឡប់មកស្មើ $1$ វិញ ដែលធានាស្ថេរភាពនៃការហូរជម្រាលក្នុងពេល Backpropagation!

---

### ខ. Masked Self-Attention (Causal Masking)
ក្នុងពេលបណ្តុះបណ្តាល Autoregressive Language Models (ដូចជា GPT) Token នៅជំហាន $i$ មិនត្រូវអនុញ្ញាតឱ្យលួចមើល Token នៅជំហានអនាគត $j > i$ ឡើយ។ យន្តការនេះត្រូវបានអនុវត្តដោយបូកម៉ាទ្រីស Mask $M$ មុនពេលរត់ Softmax៖
$$
M_{ij} = \begin{cases} 0 & \text{បើ } j \le i \\ -\infty & \text{បើ } j > i \end{cases}
$$
$$
\text{MaskedAttention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V
$$
ដោយសារ $e^{-\infty} = 0$ ទម្ងន់ Attention ទៅកាន់អនាគតនឹងស្មើនឹងសូន្យបេះបិទ!

---

### គ. Multi-Head Self-Attention (MHA)
ជំនួសឱ្យការអនុវត្ត Attention តែមួយជុំ MHA បំប្លែង $Q, K, V$ ទៅជាលំហរងចំនួន $H$ (Heads) ផ្សេងៗគ្នា ដើម្បីឱ្យក្បាលនីមួយៗអាចផ្តោតលើទំនាក់ទំនងខុសៗគ្នា (ឧ. ក្បាលទី ១ រកពាក្យកិរិយាស័ព្ទ ក្បាលទី ២ រកគុណនាម)៖
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O
$$
$$
\text{head}_h = \text{Attention}(Q W_Q^{(h)}, K W_K^{(h)}, V W_V^{(h)})
$$
ដែល $W_Q^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_K^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_V^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$, និង $W^O \in \mathbb{R}^{H d_v \times d_{\text{model}}}$។

---

## ៤. ស្ថាបត្យកម្ម និងក្បួនដោះស្រាយ (Architecture & Algorithm Walkthrough)

### ការប្រៀបធៀបគំរូនៃទិន្នន័យលំដាប់ទាំងបី (RNN vs. CNN vs. Transformer)

```text
RNN (Sequential Mixing):
X_1 ──► [Cell] ──► H_1 ──► X_2 ──► [Cell] ──► H_2 ──► ...
          │                          │
          └───(មានការពឹងផ្អែកតាមលំដាប់ពេលវេលា មិនអាចហ្វឹកហាត់ស្របគ្នាលើ GPU បាន)

CNN (Local Sliding Window):
[   X_1   X_2   X_3   X_4   X_5   ]
   \     /     /
    \   /     /
   [  Y_1   Y_2  ] ──► (ហ្វឹកហាត់ស្របគ្នាបាន តែត្រូវជង់ស្រទាប់ច្រើនទើប Receptive Field គ្របដណ្តប់)

Transformer (Global Direct Matching):
X_1 ──┐
X_2 ──┼──► [Self-Attention Multi-Head Operator] ──► Y_1, Y_2, Y_3, Y_4
X_3 ──┼──► (ហ្វឹកហាត់ស្របគ្នា១០០% ហើយគ្រប់ Token ភ្ជាប់ទាក់ទងគ្នាភ្លាមៗក្នុង ១ ស្រទាប់)
X_4 ──┘
```

---

### កូដគំរូស្ថាបត្យកម្មគ្រឹះ (PyTorch Implementation: Multi-Head Attention & Transformer Block)

កូដ PyTorch ខាងក្រោមបង្ហាញពីការសរសេរ Multi-Head Attention ពេញលេញដោយបញ្ចូលប្រមាណវិធី $Q, K, V$ ទៅក្នុង Matrix Multiply តែមួយដើម្បីពន្លឿនល្បឿន GPU ព្រមទាំងបង្កើត Pre-LN Transformer Block៖

```python
import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """
    ស្ថាបត្យកម្ម Multi-Head Self-Attention (MHA)
    បញ្ចូលប្រមាណវិធី Projection នៃ Q, K, V ចូលគ្នាដើម្បីប្រសិទ្ធភាព GPU ខ្ពស់បំផុត
    """
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model ត្រូវតែចែកដាច់នឹង num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Matrix តែមួយសម្រាប់បង្កើត Q, K, V ក្នុងពេលដំណាលគ្នា
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: [Batch_Size (B), Seq_Len (N), d_model (D)]
        B, N, D = x.shape
        
        # ជំហានទី ១: គណនា Q, K, V ក្នុងពេលតែមួយ
        qkv = self.qkv_projection(x) # [B, N, 3 * D]
        q, k, v = torch.chunk(qkv, 3, dim=-1) # ចែកជា 3 Tensors ទំហំ [B, N, D]
        
        # ជំហានទី ២: បំបែកក្បាល (Split Heads) -> [B, num_heads, N, d_k]
        q = q.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # ជំហានទី ៣: គណនា Scaled Dot-Product Attention Scores
        # scores shape: [B, num_heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # បិទទីតាំងហាមឃាត់ដោយជំនួសដោយតម្លៃអវិជ្ជមានធំ
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        
        # ជំហានទី ៤: គុណនឹង Values ដើម្បីទទួលបាន Context Representation
        out = torch.matmul(attention_weights, v) # [B, num_heads, N, d_k]
        
        # ជំហានទី ៥: ផ្គុំក្បាលទាំងអស់ត្រឡប់មកវិញ (Concat Heads)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_projection(out)


class TransformerBlock(nn.Module):
    """
    ស្ថាបត្យកម្ម Pre-LN Transformer Block ស្តង់ដារ:
    x -> LayerNorm -> Self-Attention + x -> LayerNorm -> MLP + x
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN Self-Attention ជាមួយ Residual Connection
        x = x + self.attn(self.ln1(x), mask=mask)
        # Pre-LN MLP ជាមួយ Residual Connection
        x = x + self.mlp(self.ln2(x))
        return x

# ការសាកល្បងដំណើរការ
if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4
    d_ff = 256
    
    sample_tokens = torch.randn(batch_size, seq_len, d_model)
    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(sample_tokens)
    
    print(f"ទំហំ Input: {list(sample_tokens.shape)}")
    print(f"ទំហំ Output Transformer Block: {list(output.shape)}")
```

---

## ៥. ការយល់ដឹងស៊ីជម្រៅតាមបែបវិចារណញាណ (Visual Intuition & Interpretability)

### ផែនទីតម្រឹមក្នុង Machine Translation (Alignment Heatmaps)
នៅពេលយើងបង្ហាញម៉ាទ្រីស Attention Weights នៃម៉ូដែលបកប្រែភាសា៖
* **ខ្សែបន្ទាត់ទ្រូងត្រង់ (Diagonal Path)**: បង្ហាញពីការតម្រឹមពាក្យលំដាប់ ១ ធៀបនឹង ១ ដូចគ្នា (ឧ. អង់គ្លេស "we see" ត្រូវនឹងបារាំង "nous voyons")។
* **បន្ទាត់បញ្ច្រាស (Anti-Diagonal Inversions)**: ឆ្លុះបញ្ចាំងពីវិធានវេយ្យាករណ៍ដែលផ្ទុយគ្នា។ ឧទាហរណ៍ ក្នុងភាសាអង់គ្លេស គុណនាមមកមុននាម ("European economic area") ប៉ុន្តែក្នុងភាសាបារាំង នាមមកមុនគុណនាម ("zone économique européenne") ដែលធ្វើឱ្យផែនទី Attention បង្កើតជាទម្រង់បន្ទាត់បញ្ច្រាសកាត់គ្នាដោយស្វ័យប្រវត្តិ ដោយគ្មានមនុស្សបង្រៀនឡើយ!

---

### ភាពស៊ីមេទ្រីប្តូរលំដាប់ (Permutation Equivariance)
ដោយគ្មាន Positional Encoding ប្រមាណវិធី Self-Attention មានលក្ខណៈ Permutation Equivariant៖
$$
\text{Attention}(PX, PX, PX) = P \cdot \text{Attention}(X, X, X)
$$
ដែល $P$ ជា Permutation Matrix។ ប្រសិនបើយើងមិនបូក Positional Embedding ទេ Transformer នឹងមើលឃើញប្រយោគ "ឆ្មាស៊ីត្រី" និង "ត្រីស៊ីឆ្មា" ដូចគ្នាបេះបិទ!

---

<div id="plotly-cs231n-8-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 8 visualization blueprint"></div>
<p><em>រូបភាព៖ គំនូសតាងអន្តរកម្មបង្ហាញពីទម្ងន់ Self-Attention Heatmap និងការកើនឡើងនៃទំហំគណនាជាអនុគមន៍ដឺក្រេពីរ $O(N^2)$ ធៀបនឹងប្រវែងលំដាប់ Tokens។</em></p>

---

## ៦. ការពិសោធន៍ជាក់ស្តែង និងលទ្ធផលគោលស្តង់ដារ (Empirical Heuristics & Benchmark Results)

* **ស្ថេរភាពនៃការហ្វឹកហាត់ Pre-LN ធៀបនឹង Post-LN**:  
  ស្ថាបត្យកម្មដើមរបស់ Transformer (2017) ប្រើប្រាស់ Post-LN ($x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l))$) ដែលទាមទារការ Warmup Learning Rate យ៉ាងតឹងរ៉ឹង បើពុំនោះទេជម្រាលនឹងផ្ទុះឡើងភ្លាមៗ។ ម៉ូដែលទំនើបទាំងអស់ (GPT, LLaMA, ViT) ប្រើប្រាស់ **Pre-LN** ($x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))$) ដែលបង្កើតផ្លូវល្បឿនលឿន Identity Shortcut ជួយឱ្យការហ្វឹកហាត់រាប់រយស្រទាប់មានស្ថេរភាពខ្ពស់។
* **ការរីកចម្រើននៃ Transformer (Scaling Laws)**:  
  ពីស្ថាបត្យកម្មដំបូងដែលមានត្រឹមតែ 12 ស្រទាប់ និង ~200 លានប៉ារ៉ាម៉ែត្រក្នុងឆ្នាំ ២០១៧ បច្ចុប្បន្នស្ថាបត្យកម្មដូចគ្នានេះត្រូវបានពង្រីកដល់រាប់ពាន់កោដិប៉ារ៉ាម៉ែត្រ (Trillions of parameters) ដោយរក្សាបាននូវការថយចុះនៃកំហុសតាមក្បួន Power Law យ៉ាងច្បាស់លាស់។
* **ពានរង្វាន់កិត្តិយស ICLR 2025 Test of Time Awards**:  
  * អត្ថបទស្រាវជ្រាវរបស់លោក **Dzmitry Bahdanau et al. (2015)** ដែលបានបង្កើតយន្តការ Attention លើកដំបូងសម្រាប់ Neural Machine Translation បានឈ្នះពានរង្វាន់ **ICLR 2025 Test of Time Award (Runner-Up)**។
  * អត្ថបទ **Adam Optimizer (Kingma & Ba, 2015)** បានឈ្នះពានរង្វាន់ **ICLR 2025 Test of Time Award** ធំជាងគេ ដោយសារក្បួននេះជាគ្រឹះចម្បងក្នុងការហ្វឹកហាត់ Transformers ទាំងអស់។

---

## ៧. ឧបសគ្គដែលត្រូវប្រុងប្រយ័ត្ន និងគន្លឹះដោះស្រាយកំហុស (Pitfalls & Debugging Tips)

* **គ្រោះថ្នាក់នៃការភ្លេចចែក $\sqrt{d_k}$**:  
  ប្រសិនបើអ្នកភ្លេចចែកនឹង $\sqrt{d_k}$ តម្លៃ Dot Products នឹងធំសម្បើម ធ្វើឱ្យ Softmax ចេញ One-hot vector ($1$ នៅទីតាំងមួយ និង $0$ នៅកន្លែងផ្សេងទៀត)។ ជម្រាលក្នុងពេល Backpropagation នឹងស្មើនឹងសូន្យស្ទើរតែទាំងអស់ ធ្វើឱ្យម៉ូដែលគាំងមិនព្រមរៀនអ្វីទាំងអស់។
* **របាំងនៃភាពស្មុគស្មាញដឺក្រេពីរ $O(N^2)$ (The Quadratic Barrier)**:  
  ដោយសារ Attention ត្រូវគណនាភាពស្រដៀងគ្នារវាងគ្រប់គូនៃ Token ទាំងអស់ ទំហំគណនា និង Memory របស់វាឡើងដល់ $O(N^2)$ ធៀបនឹងប្រវែងលំដាប់ $N$។ សម្រាប់អត្ថបទវែងៗ (ឧ. លើសពី 32k tokens) ប្រមាណវិធីនេះងាយនឹងធ្វើឱ្យ GPU ជួបបញ្ហា Out-of-Memory (OOM) ខ្លាំងណាស់។
* **ការភ្លេច Positional Embeddings (Order Amnesia)**:  
  កំហុសស្ងាត់ដ៏ធំបំផុតគឺការភ្លេចបូក Positional Encoding ទៅក្នុង Input Embeddings។ ម៉ូដែលនឹងបាត់បង់ការយល់ដឹងអំពីលំដាប់លំហ ឬលំដាប់ពាក្យទាំងស្រុង។

---

## ៨. សំណួរពិចារណាកម្រិតក្រោយឧត្តមសិក្សា (Graduate-Level Reflection Questions)

១. **សម្រាយបញ្ជាក់គណិតវិទ្យានៃ Softmax Gradient Saturation**:  
   ចូរទាញរកម៉ាទ្រីសដេរីវេ Jacobian នៃអនុគមន៍ Softmax $S(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$៖
   $$
   \frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)
   $$
   ចូរពន្យល់តាមរយៈរូបមន្តនេះថា ហេតុអ្វីបានជានៅពេលតម្លៃមួយក្នុងវ៉ិចទ័រ $z$ មានទំហំធំដាច់គេ ($z_{\max} \gg z_k$) ដេរីវេ $\frac{\partial S_i}{\partial z_j}$ ធ្លាក់ចុះជិតដល់សូន្យបេះបិទសម្រាប់គ្រប់ $i, j$ ទាំងអស់? តើការចែកនឹង $\sqrt{d_k}$ ដោះស្រាយបញ្ហានេះយ៉ាងដូចម្តេច?

២. **Inductive Biases ធៀបនឹង Representational Capacity (ViT vs. CNN)**:  
   ស្រទាប់ Convolution ផ្ទុកនូវ Inductive Biases ដ៏រឹងមាំចំនួនពីរគឺ **Local Connectivity** (ភីកសែលជិតគ្នាមានទំនាក់ទំនងគ្នា) និង **Translation Equivariance** (លក្ខណៈពិសេសដដែលៗអាចកើតឡើងនៅគ្រប់ទីតាំង)។ ផ្ទុយទៅវិញ Vision Transformer (ViT) គ្មាន Biases ទាំងនេះឡើយ។ ហេតុអ្វីបានជា ViT ដំណើរការចាញ់ CNN នៅពេលហ្វឹកហាត់លើ Dataset តូច (ដូចជា CIFAR-10) ប៉ុន្តែបែរជាយកឈ្នះ CNN យ៉ាងដាច់ស្រឡះនៅពេលហ្វឹកហាត់លើ Dataset ធំមហិមា (ដូចជា JFT-300M / ImageNet-21k)?

៣. **ការវិភាគអង្គចងចាំ RAM ក្នុងពេល Backpropagation នៃ Attention**:  
   ចំពោះស្រទាប់ Multi-Head Attention ដែលមាន Batch size $B$, ប្រវែងលំដាប់ $N$, វិមាត្រ $d$, និងក្បាល $H$ ចូរគណនាទំហំអង្គចងចាំ GPU ពិតប្រាកដដែលតម្រូវឱ្យផ្ទុក Attention Weights សម្រាប់ប្រើក្នុងពេល Backpropagation។ ហេតុអ្វីបានជាស្ថាបត្យកម្ម **FlashAttention** (Dao et al.) អាចបង្កើនល្បឿន និងសន្សំ Memory បានយ៉ាងសម្បើមដោយប្រើប្រាស់បច្ចេកទេស Tiling និង Recomputation ក្នុង SRAM?
