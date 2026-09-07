# Stanford CS229: ការរៀនម៉ាស៊ីន (Machine Learning)
## មេរៀនទី ១៤: ស្ថាបត្យកម្ម Transformers និងការរៀនក្នុងបរិបទ (Transformers & In-Context Learning)

### ១. សេចក្តីសង្ខេប (Summary)
មេរៀននេះបង្ហាញអំពីស្ថាបត្យកម្មការរៀនជ្រៅ (Deep Learning Architectures) និងគំរូនីតិវិធីប្រូបាប៊ីលីតេដែលជាកម្លាំងចលករស្នូលនៃ **ម៉ូដែលភាសាធំៗ (Large Language Models - LLMs)** ទំនើប។ សាស្ត្រាចារ្យ Tenyu បានកសាងប្រព័ន្ធនៃការបង្កើតគំរូលំដាប់អូតូរ៉េហ្គ្រេស៊ីវ (Autoregressive Sequence Modeling Stack) យ៉ាងមានប្រព័ន្ធ។ ដោយចាប់ផ្តើមចេញពីអត្ថបទដើម មេរៀនបានរៀបរាប់លម្អិតអំពី **ការបំប្លែងទៅជា Token (Tokenization)** ដោយប្រៀបធៀបយ៉ាងច្បាស់លាស់រវាងកម្រិតតួអក្សរ (Character), ពាក្យ (Word) និងផ្នែកនៃពាក្យ (Subword Byte-Pair Encoding - BPE) ព្រមទាំងបង្ហាញពីរបៀបដែល Token ត្រូវបានបញ្ជូនទៅក្នុងលំហវ៉ិចទ័របង្កប់វិមាត្រខ្ពស់ជាប់ (Continuous High-Dimensional Vector Embeddings)។ មេរៀនបានបង្កើតការរៀបចំចំណាត់ថ្នាក់ Token បន្ទាប់ជាផ្លូវការ ដោយបង្ហាញថាការបង្កើតគំរូភាសាតំណាងឱ្យលំដាប់នៃជម្រើស Softmax តាមលក្ខខណ្ឌលើវាក្យសព្ទដ៏ធំសម្បើម ដែលត្រូវបានបណ្តុះបណ្តាលតាមរយៈ **Negative Log-Likelihood (NLL) / Categorical Cross-Entropy Loss**។

ចំណុចស្នូលនៃមេរៀនផ្តោតលើដំណើរការខាងក្នុងនៃ **ស្ថាបត្យកម្ម Transformer (Transformer Architecture)**។ មេរៀនបានគូសបញ្ជាក់ពីមូលហេតុដែលបណ្តាញ Feedforward Multi-Layer Perceptron (MLP) ធម្មតាមិនអាចពង្រីកលើទិន្នន័យលំដាប់បាន ហើយបានពន្យល់ពីសារៈសំខាន់នៃយន្តការ **Self-Attention** ដែលជាយន្តការមូលដ្ឋានក្នុងការរួមបញ្ចូលបរិបទសេម៉ង់ទិកឆ្លងកាត់ទីតាំងនានាក្នុងលំដាប់។ បន្ទាប់ពីទាញរករូបមន្តគណិតវិទ្យានៃ Single-Head Causal Attention និងទម្រង់ម៉ាទ្រីសរបស់វា មេរៀនបានណែនាំអំពី **Causal Masking** (ការបន្ថែមព្រំដែន $-\infty$ ដើម្បីទប់ស្កាត់ការលេចធ្លាយព័ត៌មានអនាគត) និង **Multi-Head Attention (MHA)**។ នៅចុងបញ្ចប់ មេរៀនបានដោះស្រាយបញ្ហាប្រឈមកម្រិតប្រព័ន្ធផ្នែករឹង ដោយវិភាគលើកម្រិតកំណត់នៃការគណនា និងអង្គចងចាំការ៉េ $O(T^2)$ ធៀបនឹងប្រវែងលំដាប់ $T$ និងបង្ហាញពីរបៀបដែលក្បួនដោះស្រាយ Hardware-Aware ទំនើបដូចជា **FlashAttention** ជួយដោះស្រាយបញ្ហាស្ទះទាំងនេះនៅលើ GPU SRAM និង HBM។

---

### ២. គោលគំនិតគន្លឹះ និងនិយមន័យ (Key Concepts & Definitions)
- **ការបង្កើតគំរូលំដាប់អូតូរ៉េហ្គ្រេស៊ីវ (Autoregressive Sequence Modeling)**: គំរូនីតិវិធីប្រូបាប៊ីលីតេដែលប្រូបាប៊ីលីតេរួមនៃលំដាប់ Token ត្រូវបានបង្កើតជាបន្តបន្ទាប់ ដោយបង្កើតម្តងមួយ Token ផ្អែកលើប្រវត្តិនៃ Token ទាំងអស់ដែលបានបង្កើតរួចពីមុនមក។
- **ការបំប្លែងទៅជា Token កម្រិតផ្នែកនៃពាក្យ (Subword Tokenization - BPE)**: បច្ចេកទេសបំប្លែង Token កូនកាត់ (ដូចជា Byte-Pair Encoding) ដែលបំបែកពាក្យទៅជាចំណែកតូចៗដែលកើតឡើងញឹកញាប់បំផុត។ វាដោះស្រាយបញ្ហាផ្ទុះប្រវែងលំដាប់នៃ Character-level និងបញ្ហាពាក្យក្រៅវាក្យសព្ទ (Out-Of-Vocabulary - OOV) ព្រមទាំងការចែករំលែកប៉ារ៉ាម៉ែត្រនៃ Word-level។
- **Beginning of Sentence (BOS) Token ($x_0$)**: Token ពិសេសថេរដែលត្រូវបានដាក់បញ្ចូលនៅដើមដំបូងបង្អស់នៃលំដាប់អត្ថបទនីមួយៗកំឡុងពេលបណ្តុះបណ្តាល និង Inference ដើម្បីផ្តល់នូវស្ថានភាពកំណត់លក្ខខណ្ឌដំបូង (Initial Conditioning State)។
- **Logits**: តម្លៃទិន្នផលដើមមិនទាន់ធ្វើ Normalization ចេញពីស្រទាប់ចុងក្រោយនៃ Transformer ដែលត្រូវនឹងទំហំវាក្យសព្ទ $|V|$។ ការបញ្ជូន Logits ទាំងនេះឆ្លងកាត់អនុគមន៍ Softmax នឹងបម្លែងពួកវាទៅជាការបែងចែកប្រូបាប៊ីលីតេត្រឹមត្រូវ។
- **Causal Masking**: ប្រតិបត្តិការម៉ាទ្រីសត្រីកោណដែលត្រូវបានអនុវត្តលើពិន្ទុ Attention ដើមកំឡុងពេលបណ្តុះបណ្តាល ដើម្បីទប់ស្កាត់លំហូរព័ត៌មានពី Token អនាគត ($j > t$) ត្រឡប់មក Token បច្ចុប្បន្ន ($t$) ដោយធានាថាម៉ូដែលគោរពតាមលក្ខណៈ Causal យ៉ាងតឹងរ៉ឹង។
- **Isotropic Embeddings / Scaling Invariance**: លក្ខណៈសម្បត្តិស្ថិរភាពដែលបង្កើតឡើងតាមរយៈស្រទាប់ LayerNorm និង RMSNorm ដើម្បីរក្សាទំហំវ៉ិចទ័រ និងកត្តាមាត្រដ្ឋានឱ្យមានតុល្យភាពឆ្លងកាត់ស្រទាប់លាក់ជ្រៅៗជាច្រើន។
- **FlashAttention**: ក្បួនដោះស្រាយ Hardware-Aware Tiling ដែលគណនា Attention ជាក់ស្តែងដោយផ្ទាល់នៅលើប្លុក GPU SRAM ល្បឿនលឿន ដើម្បីបញ្ចៀសការសរសេរ និងអានម៉ាទ្រីស Attention ទំហំ $T \times T$ ទៅមកលើ GPU High Bandwidth Memory (HBM) ដែលមាន Latency ខ្ពស់។

---

### ៣. រូបមន្តគណិតវិទ្យា និងការទាញរក (Mathematical Formulations & Derivations)

#### ក. ការបំបែកប្រូបាប៊ីលីតេរួមបែបអូតូរ៉េហ្គ្រេស៊ីវ (Autoregressive Factored Joint Probability)
គេឱ្យវាក្យសព្ទ $V$ ដែលផ្ទុកចំណែកនៃ Subword Tokens ចំនួនបន្សំដែលអាចកើតមានសរុបនៃប្រវែង $T$ គឺ $|V|^T$។ យើងបង្កើតគំរូការបែងចែកប្រូបាប៊ីលីតេរួមនៃលំដាប់ Token $\mathbf{x} = (x_1, x_2, \dots, x_T) \in V^T$ ដោយបំបែកវាទៅជាច្រវាក់នៃការបែងចែកតាមលក្ខខណ្ឌតាមរយៈវិធានច្រវាក់នៃប្រូបាប៊ីលីតេ (Probability Chain Rule)៖
$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

#### ខ. ឧបករណ៍ទស្សន៍ទាយពាក្យបន្ទាប់តាម Softmax (The Softmax Next-Word Predictor)
សម្រាប់ជំហានពេលវេលា $t$ នីមួយៗ ម៉ូដែល Transformer ដែលមានប៉ារ៉ាម៉ែត្រ $\theta$ ផ្គូផ្គងប្រវត្តិនៃ Tokens ពីមុន $(x_0, x_1, \dots, x_{t-1})$ ទៅកាន់វ៉ិចទ័រនៃ Logits $U_t \in \mathbb{R}^{|V|}$៖
$$
U_t = f_\theta(x_0, x_1, \dots, x_{t-1})
$$
ការអនុវត្តអនុគមន៍ Softmax លើធាតុនីមួយៗបង្កើតបានជាវ៉ិចទ័រប្រូបាប៊ីលីតេលើ Simplex នៃវិមាត្រ $|V|$៖
$$
P(x_t = j \mid x_1, \dots, x_{t-1}; \theta) = \text{softmax}(U_t)_j = \frac{e^{U_{t, j}}}{\sum_{l=1}^{|V|} e^{U_{t, l}}}
$$

#### គ. អនុគមន៍បាត់បង់នៃការទស្សន៍ទាយ Token បន្ទាប់ (Next-Token Prediction Loss / Negative Log-Likelihood)
នៅពេលបណ្តុះបណ្តាល ម៉ូដែលត្រូវបានបង្ហាញលំដាប់ពេញលេញនៃ Tokens។ យើងធ្វើសុទិដ្ឋិកម្មកាត់បន្ថយ Negative Log-Likelihood (NLL) នៃការទស្សន៍ទាយ Token បន្ទាប់ពិតប្រាកដ $x_t$ លើគ្រប់ទីតាំងទាំងអស់ $t \in \{1, \dots, T\}$៖
$$
\mathcal{L}(\theta) = -\log P(x_1, \dots, x_T \mid \theta) = -\sum_{t=1}^T \log P(x_t \mid x_1, \dots, x_{t-1}; \theta)
$$
ជំនួសរូបមន្ត Softmax ចូល យើងទទួលបាន៖
$$
\mathcal{L}(\theta) = \sum_{t=1}^T \left[ -U_{t, x_t} + \log \sum_{l=1}^{|V|} e^{U_{t, l}} \right]
$$
*ចំណាំការទាញរក*៖ អនុគមន៍បាត់បង់នេះគឺសមមូលទាំងស្រុងទៅនឹង Categorical Cross-Entropy Loss រវាង Token ពិតប្រាកដបន្ទាប់ (តំណាងដោយ One-Hot Target Vector លើ $V$) និងការបែងចែកប្រូបាប៊ីលីតេដែលទស្សន៍ទាយដោយម៉ូដែល។

---

### ៤. ក្បួនដោះស្រាយសុទិដ្ឋិកម្ម និងការបង្កើតលម្អិតមួយជំហានម្តងៗ (Step-by-Step Optimization & Generation Algorithms)

<div id="plotly-14-causal-attention" class="plotly-chart" aria-label="Interactive Plotly chart: Causal attention probability heatmap"></div>

<p><em>រូបភាព៖ កម្តៅប្រូបាប៊ីលីតេនៃ Causal Attention (Causal Attention Probability Heatmap) ដោយមានទម្រង់ម៉ាទ្រីសត្រីកោណខាងក្រោម (Lower Triangular Matrix) ដែលបិទខ្ទប់ព័ត៌មានអនាគត។</em></p>

#### ក. ការបកស្រាយអូតូរ៉េហ្គ្រេស៊ីវជាមួយមាត្រដ្ឋានសីតុណ្ហភាព (Autoregressive Decoding with Temperature Scaling)
នៅពេលម៉ូដែល $\theta$ ត្រូវបានបណ្តុះបណ្តាលរួចរាល់ យើងបង្កើតអត្ថបទម្តងមួយ Token ជាបន្តបន្ទាប់។ ដើម្បីរក្សាតុល្យភាពរវាងភាពច្នៃប្រឌិត (Creativity) និងភាពស៊ីសង្វាក់គ្នានៃអត្ថបទ (Coherence) យើងអនុវត្តមាត្រដ្ឋានសីតុណ្ហភាព $\tau > 0$ ទៅលើវ៉ិចទ័រ Logits $U_t$ មុនពេលគណនា Softmax៖
1. **ចាប់ផ្តើម (Initialize)**៖ ផ្តល់លំដាប់ Prompt $(x_1, \dots, x_k)$ និងដាក់ BOS Token $x_0$ នៅខាងមុខបង្អស់។
2. **រង្វិលជុំរហូតដល់បញ្ចប់ (Loop until Termination)** (ដូចជា ការបង្កើតបាន End-of-Sentence Token ឬឈានដល់ប្រវែង Context អតិបរមា $T$)៖
   - គណនា Logits $U_t = f_\theta(x_0, \dots, x_{t-1})$។
   - ធ្វើមាត្រដ្ឋាន Logits ជាមួយសីតុណ្ហភាព $\tau$ និងអនុវត្ត Softmax៖
     $$
     P(x_t = j \mid x_1, \dots, x_{t-1}) = \frac{e^{U_{t, j}/\tau}}{\sum_l e^{U_{t, l}/\tau}}
     $$
   - **លក្ខណៈសម្បត្តិនៃសីតុណ្ហភាព $\tau$**៖
     - **$\tau \to 0$ (Greedy / Deterministic Decoding)**៖ ម៉ាស់ប្រូបាប៊ីលីតេផ្តុំទាំងស្រុងលើ Token ដែលមាន Logit ធំបំផុត៖
       $$
       P(x_t = j) \to \begin{cases} 1 & j = \arg\max_l U_{t, l} \\ 0 & \text{ផ្សេងពីនេះ} \end{cases}
       $$
     - **$\tau > 1$ (High Stochasticity)**៖ ការបែងចែកប្រែជារាបស្មើ បង្កើនភាពចម្រុះនៃការបង្កើតដោយទាញយក Token ពីកន្ទុយវែងនៃវាក្យសព្ទ។
   - **Top-$k$ Filtering**៖ រក្សាទុកតែ Token ដែលមានប្រូបាប៊ីលីតេខ្ពស់ជាងគេចំនួន $k$ និងកំណត់តម្លៃផ្សេងទៀតទាំងអស់ជាសូន្យ រួចធ្វើ Renormalize ឡើងវិញដើម្បីបញ្ចៀសការបង្កើត Token គ្មានន័យ។
   - **ទាញយកគំរូ (Sample)** Token បន្ទាប់ $x_t \sim P(x_t \mid x_{1 \dots t-1})$។
   - **ភ្ជាប់បន្ថែម (Append)** $x_t$ ទៅក្នុង Context Window ហើយបន្តទៅជំហានបន្ទាប់។

#### ខ. ស្ថាបត្យកម្ម Causal Self-Attention ទោល (Single-Head Causal Self-Attention)
Self-Attention ផ្គូផ្គងលំដាប់ធាតុចូលនៃវ៉ិចទ័រជួរដេកតំណាងលាក់ $H^{\text{in}} = [h_1^{\text{in}}; \dots; h_T^{\text{in}}] \in \mathbb{R}^{T \times d}$ ទៅកាន់លំដាប់ទិន្នផលនៃ Hidden States $H^{\text{out}} \in \mathbb{R}^{T \times d}$៖
1. **បញ្ជូនធាតុចូលទៅជា Queries, Keys, និង Values**៖
   យើងគុណវ៉ិចទ័រជួរដេកធាតុចូលពីខាងស្តាំជាមួយម៉ាទ្រីសទម្ងន់បណ្តុះបណ្តាល $W^Q \in \mathbb{R}^{d \times d_h}$, $W^K \in \mathbb{R}^{d \times d_h}$, និង $W^V \in \mathbb{R}^{d \times d}$៖
   $$
   q_t = h_t^{\text{in}} W^Q \in \mathbb{R}^{1 \times d_h}
   $$
   $$
   k_t = h_t^{\text{in}} W^K \in \mathbb{R}^{1 \times d_h}
   $$
   $$
   v_t = h_t^{\text{in}} W^V \in \mathbb{R}^{1 \times d}
   $$
2. **គណនាពិន្ទុ Attention ដើមជាមួយ Causal Masking**៖
   សម្រាប់ទីតាំង $t$ ណាមួយ យើងគណនាផលគុណស្កាលែររវាង Query $q_t$ និង Keys $k_1, \dots, k_T$ ដើម្បីវាស់វែងភាពពាក់ព័ន្ធនៃបរិបទ។ ដើម្បីរក្សាលក្ខណៈ Causal Keys នាពេលអនាគត ($j > t$) ត្រូវបានលុបបំបាត់តាមបែបគណិតវិទ្យាដោយបន្ថែមអនន្តអវិជ្ជមាន ($-\infty$) ទៅលើពិន្ទុដើមរបស់វា៖
   $$
   A_{t, j} = \begin{cases} \frac{q_t k_j^T}{c} & j \le t \\ -\infty & j > t \end{cases}
   $$
   ដែល $c = \sqrt{d_h}$ គឺជាកត្តាមាត្រដ្ឋាន (Scaling Factor) ដើម្បីទប់ស្កាត់កុំឱ្យផលគុណស្កាលែររុញច្រានជម្រាល Softmax ទៅក្នុងតំបន់រលាយបាត់ (Vanishing Gradients)។
3. **ធ្វើ Normalization តាមរយៈ Softmax**៖
   $$
   P_t = \text{softmax}(A_t \in \mathbb{R}^{1 \times T})
   $$
   ដោយសារ $e^{-\infty} = 0$ មេគុណសម្រាប់ Token អនាគតទាំងអស់ត្រូវបានលុបចោលទាំងស្រុង ដោយធានាថា $P_{t, j} = 0$ សម្រាប់រាល់ $j > t$។
4. **ការបូកបញ្ចូល Values (Value Aggregation)**៖
   $$
   h_t^{\text{out}} = \sum_{i=1}^t P_{t, i} v_i
   $$
   បន្សំប៉ោង (Convex Combination) នេះប្រមូលផ្តុំ Values នៃវ៉ិចទ័រដែលពាក់ព័ន្ធទាំងអស់ពីអតីតកាលរហូតដល់ជំហានបច្ចុប្បន្ន។

**រូបមន្តម៉ាទ្រីស Causal Attention (Matrix Formulation)**៖
ការប្រមូលផ្តុំជំហានទាំងអស់ជាទម្រង់វ៉ិចទ័រត្រូវបានសរសេរជា៖
$$
H^{\text{out}} = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_h}} + M \right) V
$$
ដែល $M \in \mathbb{R}^{T \times T}$ គឺជាម៉ាទ្រីស Causal Mask៖
$$
M_{i, j} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

---

### ៥. ប្លុកស្ថាបត្យកម្មគ្រឹះ (Architectural Building Blocks)

#### ក. Multi-Head Attention (MHA)
ដើម្បីអនុញ្ញាតឱ្យបណ្តាញអាចផ្ចង់អារម្មណ៍ (attend) ទៅលើរចនាសម្ព័ន្ធសេម៉ង់ទិកខុសៗគ្នាស្របពេលតែមួយ (ដូចជា ការវិភាគវេយ្យាករណ៍ ទល់នឹង អារម្មណ៍របស់អង្គភាព) យើងដំណើរការ $n_h$ ក្បាល Attention ស្របគ្នា៖
1. **ការគណនាស្របគ្នា**៖ ក្បាលនីមួយៗ $i \in \{1, \dots, n_h\}$ ប្រើប្រាស់សំណុំប៉ារ៉ាម៉ែត្រដាច់ដោយឡែក $\{W_i^Q, W_i^K, W_i^V\}$ ដើម្បីបង្កើតម៉ាទ្រីសទិន្នផលផ្ទាល់ខ្លួន $\text{Head}_i \in \mathbb{R}^{T \times d_{\text{head}}}$។
2. **ការតភ្ជាប់ និង Output Projection**៖
   ទិន្នផលនៃក្បាលទាំងអស់ត្រូវបានតភ្ជាប់គ្នាតាមជួរឈរ (Concatenated) ហើយបញ្ជូនត្រឡប់ទៅកាន់លំហលាក់ $d$ វិញតាមរយៈម៉ាទ្រីស $W^O \in \mathbb{R}^{(n_h d_{\text{head}}) \times d}$៖
   $$
   \text{MHA}(H^{\text{in}}) = \text{concat}\left( \text{Head}_1, \text{Head}_2, \dots, \text{Head}_{n_h} \right) W^O
   $$

#### ខ. ការប្រៀបធៀបរវាង Multi-Layer Perceptrons (MLPs) និង Self-Attention
នៅក្នុងប្លុក Transformer ស្រទាប់នីមួយៗត្រូវបានបែងចែកយ៉ាងច្បាស់លាស់៖
- **Self-Attention**៖ រួមបញ្ចូលព័ត៌មានឆ្លងកាត់ទីតាំងពេលវេលាផ្សេងៗគ្នាក្នុងលំដាប់។ វាគឺជាប្រតិបត្តិការ *តែមួយគត់* ដែលទាក់ទងឆ្លងកាត់ជំហានផ្សេងៗគ្នាក្នុង Context Window។
- **ប្លុក MLP**៖ បណ្តាញណឺរ៉ូន Feedforward ពហុស្រទាប់ដែលត្រូវបានអនុវត្តដោយឯករាជ្យ និងស្របគ្នាលើទីតាំង $t$ នីមួយៗ។ វាមិនមានការទាក់ទងគ្នាតាមពេលវេលាឡើយ ហើយដំណើរការ Hidden State របស់ Token នីមួយៗដាច់ដោយឡែកពីគ្នាដោយប្រើប៉ារ៉ាម៉ែត្ររួមគ្នា។
*ហេតុផលទ្រឹស្តី*៖ ប្រសិនបើយើងប្រើប្រាស់បណ្តាញ MLP ដ៏ធំសម្បើមមួយគ្របដណ្តប់លើ Context Window ទាំងមូល វានឹងប្រឈមនឹងការផ្ទុះឡើងនៃប៉ារ៉ាម៉ែត្រ ($O(T^2 d^2)$) និងមិនអាចបត់បែនតាមប្រវែងលំដាប់ $T$ ខុសៗគ្នាបានឡើយ។ ការឆ្លាស់គ្នារវាង Attention (ផ្សារភ្ជាប់បរិបទ) និង MLP (ដំណើរការតាមទីតាំង) ដើរតួជា Inductive Bias ដ៏សំខាន់ក្នុងការកម្រិតចំនួនប៉ារ៉ាម៉ែត្រ ខណៈពេលរក្សាបាននូវសមត្ថភាពតំណាងខ្ពស់បំផុត។

<div id="plotly-14-attention-scaling" class="plotly-chart" aria-label="Interactive Plotly chart: Attention memory scaling versus FlashAttention"></div>

<p><em>រូបភាព៖ ការប្រៀបធៀបមាត្រដ្ឋានអង្គចងចាំនៃស្តង់ដារ Attention ($O(T^2)$) ធៀបនឹង FlashAttention ($O(T)$) នៅលើ GPU SRAM/HBM។</em></p>

---

### ៦. ភាពស្មុគស្មាញនៃការគណនា និងទិដ្ឋភាពផ្នែករឹង (Computational Complexity & Hardware Realities)

#### ក. ឧបសគ្គស្ទះនៃការគណនា Causal Attention (Causal Attention Bottleneck)
ដែនកំណត់នៃការគណនា និងអង្គចងចាំនៃម៉ូដែលភាសាទំនើបត្រូវបានកម្រិតយ៉ាងខ្លាំងដោយប្រវែងលំដាប់ $T$៖
- **ភាពស្មុគស្មាញនៃការគណនា (FLOPs Complexity)**៖ ការគណនា $Q K^T$ តម្រូវឱ្យធ្វើផលគុណស្កាលែររវាង $T$ queries ជាមួយ $T$ keys។ នេះទាមទារការគណនា Dot-product ចំនួន $T^2$ ដងក្នុងវិមាត្រ $d_h$ ដែលនាំឱ្យមានភាពស្មុគស្មាញ៖
  $$
  \text{FLOPs} = O(T^2 d_h)
  $$
- **ភាពស្មុគស្មាញនៃទំហំអង្គចងចាំ (Memory Space Complexity)**៖ ការរក្សាទុកម៉ាទ្រីស Attention ដើម និងម៉ាទ្រីសដែលបាន Normalize ទាមទារការរក្សាទុកតម្លៃទំហំ $T \times T$ ក្នុងមួយស្រទាប់ ដែលនាំឱ្យ៖
  $$
  \text{Space Complexity} = O(T^2)
  $$
នៅពេលដែល $T$ កើនឡើងដល់រាប់លាន Tokens ឧបសគ្គស្ទះ $O(T^2)$ នេះក្លាយជាបញ្ហាធ្ងន់ធ្ងរដែលទាមទារឱ្យមានការកាត់តម្រឹម ឬបង្រួមប្រវត្តិនៃលំដាប់។

#### ខ. ដំណោះស្រាយបង្កើនប្រសិទ្ធភាព FlashAttention (FlashAttention Optimization)
ក្នុងប្រព័ន្ធ Deep Learning ទំនើប ឧបសគ្គផ្នែករឹងពិតប្រាកដនៃការគណនា Attention ជារឿយៗមិនមែនជាល្បឿនគណនា (Compute FLOPS) នោះទេ ប៉ុន្តែគឺកម្រិតបញ្ជូនអង្គចងចាំ (Memory Bandwidth)៖
- **បញ្ហាប្រឈម**៖ ក្នុងវិធីសាស្ត្រធម្មតា ម៉ាទ្រីស Attention ដ៏ធំ $T \times T$ ត្រូវតែសរសេរ និងអានម្តងហើយម្តងទៀតចេញពី GPU High Bandwidth Memory (HBM) ដែលមានល្បឿនយឺត។
- **ដំណោះស្រាយ**៖ FlashAttention គឺជាក្បួនដោះស្រាយ Hardware-Aware Tiling។ វាបែងចែក Queries, Keys, និង Values ទៅជាប្លុកតូចៗ (Tiles) ដែលល្មមនឹងផ្ទុកនៅក្នុង **SRAM (Static Random-Access Memory)** នៅលើយក្ស GPU ផ្ទាល់ដែលមានល្បឿនលឿនបំផុត។ វាគណនា Softmax តាមបែប Online និងធ្វើផលគុណម៉ាទ្រីសជាបន្តបន្ទាប់នៅក្នុង SRAM ដោយសរសេរត្រឡប់ទៅ HBM ដែលយឺតតែម្តងគត់នៅជំហានចុងក្រោយ។ វិធីនេះកាត់បន្ថយភាពស្មុគស្មាញនៃការចូលប្រើអង្គចងចាំ HBM ពីអនុគមន៍ដឺក្រេទីពីរ $O(T^2)$ មកត្រឹមអនុគមន៍លីនេអ៊ែរ $O(T)$ ដែលជួយបង្កើនល្បឿនយ៉ាងសម្បើមដោយមិនផ្លាស់ប្តូរលទ្ធផលគណិតវិទ្យានៃ Attention ឡើយ។

---

### ៧. កម្មវិធីអនុវត្តជាក់ស្តែង (Applications)
- **ឧបករណ៍បង្កើតអត្ថបទបែបអូតូរ៉េហ្គ្រេស៊ីវ (Autoregressive Text Generators)**៖ មូលដ្ឋានគ្រឹះនៃជំនួយការសន្ទនាឆ្លាតវៃ (ដូចជា GPT របស់ OpenAI ឬ Claude របស់ Anthropic) ដែលបង្កើត Tokens ម្តងមួយៗសម្រាប់កិច្ចការវែកញែកទូទៅ និងការសរសេរកូដ។
- **ឧបករណ៍បញ្ជាមនុស្សយន្ត Vision-Language-Action (VLA Controllers)**៖ ប្រព័ន្ធមនុស្សយន្តទំនើបប្រើប្រាស់ឆ្អឹងខ្នង Multimodal Transformer ដែលបម្លែងទិន្នន័យរូបភាពទៅជា Embeddings និងប្រើជំហាន Autoregressive Self-Attention ដើម្បីបង្កើតសកម្មភាពរូបវន្ត និងគន្លងចលនារបស់មនុស្សយន្ត។

---

### ៨. សំណួរពិចារណា និងការឆ្លុះបញ្ចាំង (Reflection Questions)
១. **ការបំប្លែង Token កម្រិតផ្នែកនៃពាក្យ (Subword) ទល់នឹង កម្រិតពាក្យ (Word)**៖ តើការប្រើប្រាស់ Subword Byte-Pair Encoding (BPE) ការពារកំហុស "ពាក្យក្រៅវាក្យសព្ទ" (Out-Of-Vocabulary) យ៉ាងដូចម្តេចនៅពេលជួបប្រទះពាក្យបច្ចេកទេសថ្មីៗដែលមិនធ្លាប់ឃើញ (ឧទាហរណ៍៖ `LLMefication`) ហើយតើវាជួយពង្រឹងសមត្ថភាពតំណាងរបស់ម៉ូដែលយ៉ាងដូចម្តេច?
២. **ឧបសគ្គប៉ារ៉ាម៉ែត្រនៃ MLP**៖ ហេតុអ្វីបានជាការប្រើប្រាស់ Multi-Layer Perceptron (MLP) ដ៏ធំសម្បើមមួយទទួលវ៉ិចទ័រតភ្ជាប់គ្នានៃ Token ទាំងអស់ ($H \in \mathbb{R}^{Td}$) គឺអន់ជាងការឆ្លាស់គ្នារវាង Self-Attention និង Position-wise MLPs ទាំងទិដ្ឋភាពនៃការពង្រីកប៉ារ៉ាម៉ែត្រ និងការដោះស្រាយប្រវែងលំដាប់អថេរ?
៣. **ឥរិយាបថនៃ Causal Masking និង Softmax**៖ ចូរវិភាគតាមបែបគណិតវិទ្យាថាតើនឹងមានអ្វីកើតឡើងចំពោះការបែងចែកប្រូបាប៊ីលីតេ Softmax $P_t$ នៅជំហាន $t$ ប្រសិនបើ Causal Masking ត្រូវបានអនុវត្តដោយកំណត់ $M_{i, j} = 0$ ជំនួសឱ្យ $M_{i, j} = -\infty$ សម្រាប់ទីតាំងអនាគត ($j > i$)? តើជម្រើស $-\infty$ ធានានូវព្រំដែន Causal យ៉ាងដូចម្តេច?

---

### ៩. ឯកសារអានបន្ថែម និងធនធាន (Further Reading & Resources)
- **Vaswani et al. (2017) "Attention Is All You Need"**: ឯកសារស្រាវជ្រាវជាប្រវត្តិសាស្ត្រដែលបានបង្កើតស្ថាបត្យកម្ម Transformer និងផ្លាស់ប្តូរមុខមាត់បញ្ញាសិប្បនិម្មិតសម័យទំនើប។
- **Dao et al. (2022) "FlashAttention"**: ឯកសារស្រាវជ្រាវដ៏ល្បីល្បាញសម្រាប់ការយល់ដឹងអំពី Hardware-Efficient Tiling នៃ Online Softmax លើប្លុក GPU SRAM។
- **Qwen 3.5 & Claude Code Tokenizer Playground**: ឧបករណ៍ធ្វើរោគវិនិច្ឆ័យលើបណ្តាញដើម្បីសាកល្បង និងស្វែងយល់ពីរបៀបដែល Byte-Pair Encoding បំបែកអត្ថបទទៅជា Subwords។
