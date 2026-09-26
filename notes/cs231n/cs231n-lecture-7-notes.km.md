# កំណត់ចំណាំមេរៀន: មេរៀនទី ៧: បណ្តាញញាណវិលជុំ និងស្ថាបត្យកម្ម LSTM (Recurrent Neural Networks & LSTMs)

**វគ្គសិក្សា**: Stanford CS231n: Deep Learning for Computer Vision (ការរៀនស៊ីជម្រៅសម្រាប់ការមើលឃើញដោយកុំព្យូទ័រ — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Prof. Fei-Fei Li (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Ranjay Krishna (សាស្ត្រាចារ្យជំនួយផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs231n.stanford.edu](https://cs231n.stanford.edu) (ស្លាយមេរៀន ឯកសារសិក្សា និងកិច្ចការអនុវត្ត)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Quick-Recall Summary)

* **គំរូនៃទិន្នន័យលំដាប់បត់បែន (Sequential Recurrence Paradigm)**: បណ្តាញញាណវិលជុំ (Recurrent Neural Networks - RNNs) ដោះស្រាយបញ្ហាទិន្នន័យដែលមានប្រវែងមិនថេរ (Variable-length Inputs / Outputs) ដោយរក្សាទុកនូវវ៉ិចទ័រស្ថានភាពលាក់ (Hidden State) $h_t = f_W(h_{t-1}, x_t)$ ដែលដើរតួជាការចងចាំ (Memory) ឆ្លងកាត់ពេលវេលា ដោយប្រើប្រាស់ប៉ារ៉ាម៉ែត្រទម្ងន់រួមគ្នា ($W$) គ្រប់ជំហានទាំងអស់។
* **បញ្ហារាំងស្ទះនៃក្បួន BPTT (The BPTT Bottleneck)**: ក្បួន Backpropagation Through Time (BPTT) ពន្លាតគំនូសតាងគណនាតាមបណ្តោយប្រវែងលំដាប់ ដែលទាមទារការគុណម៉ាទ្រីសទម្ងន់ដដែលៗ ($W_{hh}$) ជាច្រើនដង។ ការធ្វើបែបនេះបណ្តាលឱ្យជម្រាលស្ទុះឡើងដល់អនន្ត (Exploding Gradients) ឬសាបរលាបបាត់បង់ទាំងស្រុង (Vanishing Gradients) យ៉ាងឆាប់រហ័ស។
* **យន្តការដោះស្រាយច្រាំងចោទនៃជម្រាល (Gradient Cliff Mitigation)**: បញ្ហា Exploding Gradients ត្រូវបានទប់ស្កាត់ដោយ **Gradient Clipping** (ការកាត់បន្ថយកម្រិត Norm នៃជម្រាលមិនឱ្យលើសពីកម្រិតកំណត់ $\tau$) រីឯបញ្ហា Vanishing Gradients ត្រូវបានដោះស្រាយដោយស្ថាបត្យកម្ម **LSTM** ដែលបង្កើតផ្លូវល្បឿនលឿននៃការចងចាំតាមរយៈផលបូកលីនេអ៊ែរ (Additive Cell State Highway)។
* **វិធីសាស្ត្រ Truncated BPTT**: ដើម្បីការពារការស៊ីទំហំអង្គចងចាំ GPU ហួសកម្រិត ($O(T)$ Activation Caching) ក្នុងលំដាប់វែងៗ ការហ្វឹកហាត់ត្រូវបានកាត់ជាកង់ៗ (Windows) ដោយផ្តាច់លំហូរជម្រាលនៅត្រឹមព្រំដែន ប៉ុន្តែបញ្ជូនបន្តតម្លៃ Hidden State $h_t$ ទៅមុខរហូត។
* **ការរួមបញ្ចូលគ្នារវាងរូបភាព និងអត្ថបទ (Multimodal Vision-Language Conditioning)**: ក្នុងកិច្ចការដូចជាការបង្កើតចំណងជើងរូបភាព (Image Captioning) លក្ខណៈពិសេសកម្រិតខ្ពស់ពីរូបភាពដែលស្រង់ចេញដោយ CNN ត្រូវបានបញ្ជូនទៅកំណត់តម្លៃដើមនៃ Hidden State ($h_0 = f_W(v)$) របស់ RNN ដោយផ្ទាល់។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

### ក. ទម្រង់ដំណើរការនៃទិន្នន័យលំដាប់ (Sequence Modeling Paradigms)
ផ្ទុយពីបណ្តាញញាណធម្មតាដែលទទួល Input ថេរមួយ ហើយបញ្ចេញ Output ថេរមួយ (One-to-One) បណ្តាញញាណ RNN អាចដំណើរការទិន្នន័យជាច្រើនទម្រង់៖
* **One-to-Many**: ធាតុចូលទោល បញ្ចេញជាលំដាប់ (ឧ. Image Captioning: រូបភាព ១ $\to$ លំដាប់ពាក្យជាច្រើន)។
* **Many-to-One**: ធាតុចូលជាលំដាប់ បញ្ចេញលទ្ធផលទោល (ឧ. Sentiment Classification: ល្បះអត្ថបទ $\to$ ពិន្ទុអារម្មណ៍)។
* **Many-to-Many (Synchronized)**: ធាតុចូលជាលំដាប់ បញ្ចេញ Output ភ្លាមៗគ្រប់ជំហាន (ឧ. Video Frame Classification)។
* **Many-to-Many (Encoder-Decoder / Seq2Seq)**: ទទួលលំដាប់ទាំងមូលសិន រួចទើបបញ្ចេញលំដាប់ថ្មី (ឧ. Machine Translation: ភាសាអង់គ្លេស $\to$ ភាសាខ្មែរ)។

---

### ខ. សមីការគ្រឹះនៃ Vanilla RNN
នៅជំហានពេលវេលានីមួយៗ $t$ វ៉ិចទ័រស្ថានភាពលាក់ $h_t \in \mathbb{R}^H$ ត្រូវបានធ្វើបច្ចុប្បន្នភាពដោយប្រើ Hidden State មុន $h_{t-1} \in \mathbb{R}^H$ និងធាតុចូលបច្ចុប្បន្ន $x_t \in \mathbb{R}^D$៖
$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$
* $W_{hh} \in \mathbb{R}^{H \times H}$: ម៉ាទ្រីសទម្ងន់តភ្ជាប់រវាង Hidden State និង Hidden State។
* $W_{xh} \in \mathbb{R}^{H \times D}$: ម៉ាទ្រីសទម្ងន់តភ្ជាប់រវាង Input និង Hidden State។
* $b_h \in \mathbb{R}^H$: វ៉ិចទ័រ Bias នៃស្រទាប់លាក់។

ការព្យាករណ៍ទិន្នន័យចេញ $y_t \in \mathbb{R}^C$ ត្រូវបានគណនាដោយ៖
$$
y_t = W_{hy} h_t + b_y
$$

---

### គ. ស្ថាបត្យកម្ម Long Short-Term Memory (LSTM)
ដើម្បីលុបបំបាត់បញ្ហា Vanishing Gradient ស្ថាបត្យកម្ម LSTM បានបំបែក Memory ជាពីរ៖ **Cell State ($c_t$)** ដែលដើរតួជាផ្លូវល្បឿនលឿនផ្ទុកការចងចាំរយៈពេលវែង និង **Hidden State ($h_t$)** ដែលតំណាងឱ្យការចងចាំរយៈពេលខ្លី។  
នៅជំហាន $t$ បណ្តាញគណនាទ្វារបញ្ជា (Gates) ចំនួន ៤៖
1. **Forget Gate ($f$)**: សម្រេចថាតើត្រូវលុបការចងចាំចាស់ $c_{t-1}$ ចោលប៉ុន្មានភាគរយ ($0$ = លុបចោលទាំងស្រុង, $1$ = រក្សាទុកទាំងស្រុង)។
2. **Input Gate ($i$)**: សម្រេចថាតើត្រូវសរសេរព័ត៌មានថ្មីចូលទៅក្នុងការចងចាំកម្រិតណា។
3. **Gate Gate / Candidate State ($g$)**: ជាខ្លឹមសារព័ត៌មានថ្មីដែលត្រៀមបញ្ចូលទៅក្នុងការចងចាំ។
4. **Output Gate ($o$)**: សម្រេចថាតើផ្នែកណាខ្លះនៃ Cell State $c_t$ ត្រូវបញ្ចេញទៅជា Hidden State $h_t$។

---

## ៣. មូលដ្ឋានគ្រឹះគណិតវិទ្យា & រូបមន្ត (Mathematical Foundations & Formulations)

### ក. សម្រាយបញ្ជាក់គណិតវិទ្យានៃបញ្ហាជម្រាលក្នុង Vanilla RNN
ឧបមាថាបណ្តាញ RNN ត្រូវបានពន្លាតចំនួន $T$ ជំហាន ហើយ Loss $L$ ត្រូវបានគណនានៅជំហានចុងក្រោយ $T$។  
ដើម្បីគណនាជម្រាលធៀបនឹង Hidden State ដំបូងៗ $h_t$ ($t \ll T$) យើងអនុវត្តក្បួនសង្វាក់ Calculus Chain Rule៖
$$
\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \frac{\partial h_T}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
$$

ម៉ាទ្រីសដេរីវេ Jacobian ក្នុងតំបន់គឺ៖
$$
\frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}\left(1 - \tanh^2(W_{hh} h_{k-1} + W_{xh} x_k + b_h)\right) W_{hh}^T
$$

នៅពេលយើងគុណម៉ាទ្រីស Jacobian នេះផ្ទួនៗគ្នាចំនួន $T - t$ ដង៖
$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \operatorname{diag}\left(1 - \tanh^2(\cdot)\right) W_{hh}^T
$$

* **Exploding Gradients (ជម្រាលផ្ទុះឡើង)**: ប្រសិនបើតម្លៃ Eigenvalue ធំបំផុតនៃម៉ាទ្រីស $W_{hh}$ ធំជាង ១ ($\rho(W_{hh}) > 1$) នោះផលគុណស្វ័យគុណ $\left(W_{hh}^T\right)^{T-t}$ នឹងកើនឡើងជាអនុគមន៍ស្វ័យគុណឆ្ពោះទៅរកអនន្ត បណ្តាលឱ្យទម្ងន់ខូចខាតចេញតម្លៃ `NaN`។
* **Vanishing Gradients (ជម្រាលសាបរលាប)**: ដោយសារដេរីវេនៃអនុគមន៍ $\tanh$ តែងតែស្ថិតក្នុងចន្លោះ $(0, 1]$ ហើយភាគច្រើនតូចជាង ១ បូករួមទាំងករណីដែល $\rho(W_{hh}) < 1$ ផងនោះ ផលគុណផ្ទួនៗគ្នានឹងធ្លាក់ចុះជាលំដាប់ស្វ័យគុណដល់សូន្យបេះបិទ។ ជម្រាលពីជំហានចុងក្រោយមិនអាចហូរមកដល់ជំហានដំបូងៗបានឡើយ ដែលធ្វើឱ្យម៉ូដែល **ភ្លេចបរិបទចាស់ៗទាំងស្រុង**។

---

### ខ. យន្តការកាត់បន្ថយជម្រាលជ្រុល (Gradient Norm Clipping)
ដើម្បីការពារការផ្ទុះឡើងនៃជម្រាល (Gradient Explosion) យើងពិនិត្យមើលប្រវែង $L_2$ Norm នៃវ៉ិចទ័រជម្រាល $g = \nabla_W L$។ ប្រសិនបើវាលើសពីកម្រិតកំណត់ $\tau$ យើងធ្វើការបង្រួមវាឡើងវិញ៖
$$
\hat{g} = \begin{cases} g & \text{បើ } \|g\|_2 \le \tau \\ \frac{\tau}{\|g\|_2} g & \text{បើ } \|g\|_2 > \tau \end{cases}
$$
*វិធីសាស្ត្រនេះរក្សាទិសដៅដើមនៃជម្រាលឱ្យនៅដដែល ១០០% ប៉ុន្តែកាត់បន្ថយទំហំជំហានកុំឱ្យលោតផ្លាតចេញពី Loss Surface*។

---

### គ. សមីការគណិតវិទ្យាពេញលេញនៃស្ថាបត្យកម្ម LSTM
សម្រាប់ជំហាន $t$ ដោយរួមបញ្ចូលគ្នានូវវ៉ិចទ័រ $[h_{t-1}, x_t]$៖

$$
\begin{pmatrix} i_t \\ f_t \\ o_t \\ g_t \end{pmatrix} = \begin{pmatrix} \sigma \\ \sigma \\ \sigma \\ \tanh \end{pmatrix} \left( W \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix} + b \right)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t \quad \text{(Cell State Highway)}
$$
$$
h_t = o_t \odot \tanh(c_t) \quad \text{(Hidden State)}
$$

* **ហេតុអ្វីបានជា LSTM មិនបាត់បង់ជម្រាល?**  
  ពិចារណាដេរីវេដោយផ្នែកនៃ Cell State ថ្មីធៀបនឹង Cell State ចាស់៖
  $$
  \frac{\partial c_t}{\partial c_{t-1}} = f_t
  $$
  នៅក្នុងពេល Backpropagation ជម្រាលហូរថយក្រោយតាមសមីការ៖
  $$
  \frac{\partial L}{\partial c_{t-1}} = \frac{\partial L}{\partial c_t} \odot f_t
  $$
  ប្រសិនបើ Forget Gate $f_t \approx 1$ នោះជម្រាលនឹងហូរថយក្រោយដោយគ្មានការថយចុះតម្លៃឡើយ! គ្មានការគុណនឹងម៉ាទ្រីស $W_{hh}$ គ្មានការកាត់បន្ថយដោយដេរីវេនៃ $\tanh$។ នេះគឺជា **ផ្លូវល្បឿនលឿននៃការចងចាំ (Additive Gradient Superhighway)** ដ៏ល្បីល្បាញរបស់ LSTM។

---

## ៤. ស្ថាបត្យកម្ម និងក្បួនដោះស្រាយ (Architecture & Algorithm Walkthrough)

### ដ្យាក្រាមពន្លាតតាមពេលវេលា (Temporal Unrolling & BPTT)

```text
Forward Pass (ពន្លាតតាមពេលវេលា):
 h0 (init) ───────► [ RNN Cell ] ───────► [ RNN Cell ] ───────► [ RNN Cell ]
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
* ផលបូកជម្រាលនៃប៉ារ៉ាម៉ែត្ររួមឆ្លងកាត់គ្រប់ជំហានទាំងអស់:
  dL/dW_hh = dL1/dW_hh + dL2/dW_hh + dL3/dW_hh
```

---

### កូដគំរូស្ថាបត្យកម្មគ្រឹះ (Custom RNN & LSTM Cells ក្នុង PyTorch)

កូដ PyTorch ខាងក្រោមបង្ហាញពីការកសាង `CustomRNNCell` និង `CustomLSTMCell` ពីកម្រិតគ្រឹះ ដោយបង្ហាញយ៉ាងច្បាស់ពីការផ្គុំប្រមាណវិធីម៉ាទ្រីសដើម្បីបង្កើនល្បឿន GPU (Fusing Linear Projections):

```python
import torch
import torch.nn as nn

class CustomRNNCell(nn.Module):
    """
    ស្រទាប់គ្រឹះ Vanilla RNN: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CustomRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        # រួមបញ្ចូល W_xh និង W_hh សម្រាប់ការគណនាលឿន
        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x_t: [Batch, input_dim], h_prev: [Batch, hidden_dim]
        h_next = self.tanh(self.W_xh(x_t) + self.W_hh(h_prev))
        return h_next


class CustomLSTMCell(nn.Module):
    """
    ស្រទាប់គ្រឹះ LSTM Cell ដែលប្រើប្រាស់ Linear Projection តែមួយ
    ដើម្បីគណនាទ្វារទាំងបួន (i, f, o, g) ក្នុងពេលដំណាលគ្នាយ៉ាងមានប្រសិទ្ធភាព។
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CustomLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # គុណនឹង 4 ដើម្បីគណនា i, f, o, g ក្នុងពេលតែមួយ
        self.W_gate = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # គន្លឹះដោះស្រាយកំហុស: កំណត់ Forget Gate Bias ឱ្យធំជាមុន (bf = 1.0)
        with torch.no_grad():
            # Forget gate ស្ថិតនៅចន្លោះ hidden_dim ដល់ 2 * hidden_dim
            self.W_gate.bias[hidden_dim:2 * hidden_dim].fill_(1.0)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        # ផ្គុំ Hidden State និង Input ចូលគ្នា
        combined = torch.cat([h_prev, x_t], dim=1) # [Batch, hidden_dim + input_dim]
        
        # គណនា Projections ទាំងអស់ក្នុងពេលតែមួយ
        gates = self.W_gate(combined) # [Batch, 4 * hidden_dim]
        i_gate, f_gate, o_gate, g_gate = torch.chunk(gates, 4, dim=1)
        
        # អនុវត្តអនុគមន៍សកម្មកម្មតាមទ្វារនីមួយៗ
        i = self.sigmoid(i_gate) # Input gate
        f = self.sigmoid(f_gate) # Forget gate
        o = self.sigmoid(o_gate) # Output gate
        g = self.tanh(g_gate)    # Candidate cell state
        
        # ផ្លូវល្បឿនលឿននៃការចងចាំ (Additive Cell State Update)
        c_next = f * c_prev + i * g
        
        # គណនា Hidden State ចុងក្រោយ
        h_next = o * self.tanh(c_next)
        
        return h_next, c_next

# ការសាកល្បងដំណើរការលំដាប់ពេលវេលា
if __name__ == "__main__":
    batch_size = 3
    input_dim = 10
    hidden_dim = 20
    seq_len = 5
    
    lstm_cell = CustomLSTMCell(input_dim, hidden_dim)
    
    # កំណត់តម្លៃ Memory ដើមជាសូន្យ
    h_t = torch.zeros(batch_size, hidden_dim)
    c_t = torch.zeros(batch_size, hidden_dim)
    
    # បង្កើតទិន្នន័យគំរូលំដាប់ 5 ជំហាន
    inputs = [torch.randn(batch_size, input_dim) for _ in range(seq_len)]
    
    for t, x_step in enumerate(inputs):
        h_t, c_t = lstm_cell(x_step, h_t, c_t)
        print(f"ជំហាន {t+1}: h_t norm = {h_t.norm().item():.3f}, c_t norm = {c_t.norm().item():.3f}")
```

---

## ៥. ការយល់ដឹងស៊ីជម្រៅតាមបែបវិចារណញាណ (Visual Intuition & Interpretability)

### ការបកស្រាយសមត្ថភាពណឺរ៉ូនក្នុង Character-Level Language Models
ការសិក្សាស្រាវជ្រាវរបស់លោក Andrej Karpathy (2015) តាមរយៈការតាមដានសកម្មកម្មនៃណឺរ៉ូនក្នុង LSTM បានបង្ហាញថាកោសិកាខ្លះរៀនជំនាញស្មុគស្មាញដោយស្វ័យប្រវត្តិ៖
* **កោសិកាចាប់សញ្ញាសម្រង់សំដី (Quotes Cell)**: សកម្មភាពរបស់ណឺរ៉ូនលោតឡើង $+1$ ភ្លាមៗនៅពេលជួបសញ្ញាសម្រង់ `"` ដំបូង ហើយបន្តរក្សាតម្លៃ $+1$ រហូតដល់ជួបសញ្ញាសម្រង់បិទបញ្ចប់ទើបធ្លាក់ចុះមកវិញ។
* **កោសិកាចាប់កម្រិត Tab ក្នុងកូដ (Indentation Cell)**: នៅក្នុងកូដ C/Python ណឺរ៉ូនមួយចំនួនបង្កើនកម្រិតសកម្មកម្មជាលំដាប់រាល់ពេលជួបសញ្ញាបើកវង់ក្រចក `{` ឬការចុះបន្ទាត់ចូលក្នុង ហើយថយចុះវិញពេលជួបសញ្ញា `}`។
* **កោសិការាប់ប្រវែងបន្ទាត់ (Line-Length Counter)**: ណឺរ៉ូនដើរតួជានាឡិកាវាស់ ដោយកើនឡើងជាលំដាប់តាមចំនួនតួអក្សរក្នុងមួយជួរ ហើយ reset មកសូន្យភ្លាមៗពេលជួប `\n`។

---

### បញ្ហាភាន់ច្រឡំក្នុងការបង្កើតចំណងជើងរូបភាព (Image Captioning Biases)

```text
រូបភាពមនុស្សកាន់ផ្លែប៉ោមជិតមុខ ──► [ CNN + RNN ] ──► "មនុស្សម្នាក់កំពុងនិយាយទូរស័ព្ទ!"
```
* ដោយសារនៅក្នុង Dataset ពាក្យ "មនុស្ស", "ដៃ", និង "ជិតមុខ" ភាគច្រើនកើតឡើងជាមួយ "ទូរស័ព្ទ" នោះ RNN ងាយនឹងទន្ទេញចាំនូវស្ថិតិពាក្យរួមគ្នា (Statistical Co-occurrence) ជាជាងការសម្លឹងមើលភីកសែលរូបភាពជាក់ស្តែង។ នេះជាបញ្ហាប្រឈមធំដែលជំរុញឱ្យមានការបង្កើតយន្តការ **Attention Mechanisms** ក្នុងមេរៀនបន្ទាប់។

---

<div id="plotly-cs231n-7-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 7 visualization blueprint"></div>
<p><em>រូបភាព៖ គំនូសតាងអន្តរកម្មបង្ហាញពីទំហំជម្រាលហូរថយក្រោយតាមពេលវេលា (Gradient Flow Through Time) រវាង Vanilla RNN (ធ្លាក់ចុះជាអនុគមន៍ស្វ័យគុណដល់ $10^{-12}$) ធៀបនឹងផ្លូវល្បឿនលឿននៃ LSTM Cell Highway (រក្សាតម្លៃនៅក្បែរ $1.0$)។</em></p>

---

## ៦. ការពិសោធន៍ជាក់ស្តែង និងលទ្ធផលគោលស្តង់ដារ (Empirical Heuristics & Benchmark Results)

* **ការកំណត់តម្លៃដើមនៃ Hidden State ($h_0, c_0$)**: ជាទូទៅ ការកំណត់ $h_0 = 0$ និង $c_0 = 0$ ដំណើរការបានល្អ។ ប៉ុន្តែចំពោះកិច្ចការស្មុគស្មាញ ការកំណត់ឱ្យ $h_0$ ជា Learnable Parameter ឬគណនាចេញពី Feature Vector នៃរូបភាព ($h_0 = W_v v$) ជួយបង្កើនល្បឿននៃការបញ្ចូលគ្នានៃម៉ូដែល។
* **ទំហំបង្អួច Truncated BPTT (Window Size)**: ក្នុងការអនុវត្តជាក់ស្តែង ប្រវែងបង្អួចត្រូវបានកំណត់ចន្លោះពី **១៦ ទៅ ១០០ ជំហាន**។ បង្អួចកាន់តែវែង ម៉ូដែលរៀនទំនាក់ទំនងរយៈពេលវែងបានកាន់តែល្អ ប៉ុន្តែទាមទារអង្គចងចាំ GPU កាន់តែធំ។
* **ហេតុអ្វីត្រូវជ្រើសរើស $\tanh$ ជំនួសឱ្យ ReLU ក្នុង RNN?**:  
  នៅក្នុងស្រទាប់លាក់នៃ RNN ការគុណនឹងម៉ាទ្រីស $W_{hh}$ ដដែលៗរាប់សិបដងធ្វើឱ្យសកម្មកម្មងាយនឹងផ្ទុះឡើងខ្លាំង។ អនុគមន៍ $\tanh$ បង្រួមតម្លៃឱ្យស្ថិតក្នុងចន្លោះ $(-1, 1)$ យ៉ាងតឹងរ៉ឹង ដែលជួយរក្សាស្ថេរភាពនៃប្រព័ន្ធឌីណាមិកឆ្លងកាត់រាប់រយជំហាន។

---

## ៧. ឧបសគ្គដែលត្រូវប្រុងប្រយ័ត្ន និងគន្លឹះដោះស្រាយកំហុស (Pitfalls & Debugging Tips)

* **គន្លឹះ Forget Gate Bias Initialization ($b_f = 1.0$)**:  
  ប្រសិនបើតម្លៃ Bias នៃ Forget Gate ត្រូវបានកំណត់ជា $0$ នោះនៅពេលចាប់ផ្តើមហ្វឹកហាត់ $\sigma(0) = 0.5$ ដែលមានន័យថាការចងចាំពាក់កណ្តាលនឹងត្រូវលុបបំបាត់ចោលនៅរាល់ជំហាននីមួយៗ! **គន្លឹះដោះស្រាយ**: តែងតែកំណត់ $b_f = 1.0$ ឬ $2.0$ តាំងពីដំបូង ដើម្បីបង្ខំឱ្យបណ្តាញចងចាំព័ត៌មានទាំងអស់ជាមុនសិន។
* **ការផ្ទុះឡើងនៃជម្រាល និងសញ្ញា Loss ចេញ `NaN`**:  
  ប្រសិនបើក្រាហ្វ Training Loss ស្រាប់តែលោតឡើងខ្ពស់ខុសប្រក្រតី ឬចេញតម្លៃ `NaN` ភ្លាមៗ នេះជាសញ្ញាច្បាស់លាស់នៃ Exploding Gradients ក្នុង BPTT។ **គន្លឹះដោះស្រាយ**: បន្ថែម `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` ជាបន្ទាន់។
* **ភាពខុសគ្នារវាង Teacher Forcing និង Inference**:  
  ពេល Training យើងបញ្ជូនពាក្យពិតប្រាកដ (Ground Truth token) ទៅកាន់ជំហានបន្ទាប់ (Teacher Forcing)។ ប៉ុន្តែពេល Testing ម៉ូដែលត្រូវប្រើពាក្យដែលខ្លួនឯងទើបនឹងទស្សន៍ទាយចេញ (Autoregressive Sampling)។ ប្រសិនបើម៉ូដែលទស្សន៍ទាយពាក្យដំបូងខុស នោះកំហុសនឹងរីករាលដាលជាសង្វាក់ (Error Accumulation)។

---

## ៨. សំណួរពិចារណាកម្រិតក្រោយឧត្តមសិក្សា (Graduate-Level Reflection Questions)

១. **សម្រាយបញ្ជាក់គណិតវិទ្យានៃផ្លូវកាត់ LSTM Cell State (Additive Highway Derivation)**:  
   ចូរទាញរកដេរីវេដោយផ្នែក $\frac{\partial c_t}{\partial c_{t-1}}$ ចេញពីសមីការ $c_t = f_t \odot c_{t-1} + i_t \odot g_t$។ ចូរពន្យល់ពីមូលហេតុដែលកន្សោមជម្រាលនេះមិនរងការគាបសង្កត់ដោយម៉ាទ្រីសដេរីវេ Jacobian នៃ $W_{hh}$ ដូចនៅក្នុង Vanilla RNN ឡើយ នៅពេលដែល $f_t \approx 1$។

២. **បញ្ហារាំងស្ទះនៃការបង្រួមព័ត៌មាន (The Context Compression Bottleneck)**:  
   ឧបមាថាអ្នកមានម៉ូដែល RNN មួយដែលមានទំហំស្រទាប់លាក់ $H = 256$ វិមាត្រ។ ប្រសិនបើអ្នកបញ្ចូលលំដាប់អត្ថបទដែលមានប្រវែង $T = 1000$ ពាក្យ តើមានដែនកំណត់ទ្រឹស្តីអ្វីខ្លះក្នុងការផ្ទុកព័ត៌មានទាំងអស់នៃអត្ថបទនោះចូលទៅក្នុងវ៉ិចទ័រទំហំ ២៥៦ ចំនួនគត់? ចូរពន្យល់តាមទស្សនៈនៃ Lossy Information Compression និងមូលហេតុដែលជំរុញឱ្យពិភពលោកផ្លាស់ប្តូរទៅរក Transformer Attention។

៣. **ផលប៉ះពាល់ និងលម្អៀងនៃ Truncated BPTT (TBPTT Trade-offs)**:  
   ប្រសិនបើកិច្ចការជាក់ស្តែងមួយទាមទារការចងចាំប្រវែង ១០០ ជំហានទើបអាចយល់អត្ថន័យបាន ប៉ុន្តែដោយសារកម្រិតកំណត់នៃអង្គចងចាំ GPU យើងត្រូវបង្ខំចិត្តកាត់បង្អួច Truncated BPTT ត្រឹម $k = 20$ ជំហាន។ តើការធ្វើបែបនេះបង្កឱ្យមានលម្អៀង (Bias) យ៉ាងដូចម្តេចខ្លះក្នុងការប៉ាន់ស្មានជម្រាលពិតប្រាកដ (True Gradients) ហើយតើម៉ូដែលនឹងរៀនបានអ្វីខ្លះ?
