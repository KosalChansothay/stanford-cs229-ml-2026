# កំណត់ចំណាំមេរៀន: មេរៀនទី ៣: ការកែតម្រូវទម្ងន់ និងការបង្កើនប្រសិទ្ធភាព (Regularization & Optimization)

**វគ្គសិក្សា**: Stanford CS231n: Deep Learning for Computer Vision (ការរៀនស៊ីជម្រៅសម្រាប់ការមើលឃើញដោយកុំព្យូទ័រ — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Prof. Fei-Fei Li (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Ranjay Krishna (សាស្ត្រាចារ្យជំនួយផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs231n.stanford.edu](https://cs231n.stanford.edu) (ស្លាយមេរៀន ឯកសារសិក្សា និងកិច្ចការអនុវត្ត)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Quick-Recall Summary)

* **ភាពចាំបាច់នៃការកែតម្រូវទម្ងន់ (The Regularization Mandate)**: ការកែតម្រូវទម្ងន់ ($R(W)$) មានគោលបំណងបន្ទាបប្រសិទ្ធភាពលើសំណុំទិន្នន័យហ្វឹកហាត់ (Training set) ដោយចេតនា ដើម្បីទប់ស្កាត់បញ្ហា Overfitting និងបង្កើនសមត្ថភាពទូទៅភាវូបនីយកម្ម (Generalization) លើទិន្នន័យតេស្តថ្មីៗ។ ការកែតម្រូវតាម $L2$ (Weight Decay) ដាក់ទោសលើទម្ងន់ណាដែលមានតម្លៃលោតខ្ពស់ដាច់គេ ដោយជំរុញឱ្យម៉ូដែលបែងចែកទម្ងន់ឱ្យរាយស្មើៗគ្នាក្នុងគ្រប់លក្ខណៈពិសេស (Diffuse weights) រីឯការកែតម្រូវតាម $L1$ ជំរុញឱ្យទម្ងន់ជាច្រើនធ្លាក់ចុះដល់សូន្យបេះបិទ (Sparsity)។
* **ភាពខុសគ្នានៃការគណនាជម្រាល (The Gradient Dichotomy)**:  
  * **Numerical Gradient (ជម្រាលលេខគណិត)**: ជាតម្លៃប្រហាក់ប្រហែល ស៊ីពេលគណនាខ្លាំង $O(\#\text{params})$ ប៉ុន្តែងាយស្រួលសរសេរកូដបំផុត ហេតុនេះវាស័ក្តិសមសម្រាប់តែការផ្ទៀងផ្ទាត់កូដ (Gradient Checking) ប៉ុណ្ណោះ។
  * **Analytical Gradient (ជម្រាលវិភាគតាមរូបមន្ត)**: ជាតម្លៃពិតប្រាកដ១០០% គណនាលឿនរហ័សបំផុតតាមរយៈ Backpropagation ប៉ុន្តែងាយនឹងសរសេរខុស ហេតុនេះទើបគេត្រូវប្រើ Numerical Gradient មកផ្ទៀងផ្ទាត់ជានិច្ច។
* **ដែនកំណត់ធំៗទាំងបីនៃ Vanilla SGD (Failures of Vanilla SGD)**: ក្បួនដោះស្រាយ Stochastic Gradient Descent សាមញ្ញជួបការលំបាកធ្ងន់ធ្ងរចំនួនបី៖ (១) ស្ថានភាពជ្រលងភ្នំចោទមិនស្មើគ្នា (Poor Conditioning / High Condition Number) ដែលបណ្តាលឱ្យទិសដៅលំយោលទៅមកយ៉ាងខ្លាំង (Oscillation), (២) ចំណុចកែបសេះ និងអប្បបរមាក្នុងតំបន់ (Saddle Points & Local Minima) ដែលមានជម្រាលស្មើនឹងសូន្យ ធ្វើឱ្យការរៀនគាំងជាប់គាំង, និង (៣) សំឡេងរំខាននៃជម្រាល (Gradient Noise) ដែលកើតចេញពីការទាញយកគំរូទិន្នន័យតូចៗ (Mini-batches)។
* **ក្បួនដោះស្រាយបង្កើនប្រសិទ្ធភាពតាមជំហានបត់បែន (Adaptive Step Optimization)**: ម៉ូដែលទំនើបប្រើប្រាស់រូបវិទ្យាដើម្បីបង្កើនល្បឿន៖ *Momentum* ប្រើប្រាស់សន្ទុះល្បឿន (Velocity) ដើម្បីរុញច្រានម៉ូដែលឱ្យរំលងផុតចំណុចកែបសេះ និងទប់លំយោលសងខាង; *RMSprop* ចែកល្បឿនរៀន (Learning rate) នឹងឫសការ៉េនៃមធ្យមភាគការ៉េជម្រាល ដើម្បីពន្លឿនទិសដៅរាបស្មើ; រីឯ *Adam* រួមបញ្ចូលទាំង Momentum និង RMSprop ព្រមទាំងបន្ថែមយន្តការកែតម្រូវភាពលម្អៀងដំបូង (Bias Correction) ដើម្បីការពារកុំឱ្យបោះជំហានធំជ្រុលនៅវិនាទីដំបូង។
* **ដែនកំណត់នៃការគណនាលំដាប់ទីពីរ (Second-Order Scalability Limit)**: វិធីសាស្ត្រលំដាប់ទីពីរ (Second-order optimization ដូចជា Newton's Method) ផ្តល់នូវការរួមតូចនៃកំហុសយ៉ាងលឿនបំផុត (Quadratic Convergence) ដោយប្រើម៉ាទ្រីសដេរីវេទីពីរ Hessian $H$ ប៉ុន្តែមិនអាចយកមកប្រើក្នុងបណ្តាញញាណជ្រៅបានទេ ពីព្រោះការផ្ទុកម៉ាទ្រីស $N \times N$ ទាមទារទំហំអង្គចងចាំ $O(N^2)$ (បើម៉ូដែលមួយមានប៉ារ៉ាម៉ែត្រ ១ ពាន់លាន ម៉ាទ្រីស Hessian នឹងមានទំហំ $10^{18}$ ធាតុ ដែលគ្មានកុំព្យូទ័រណាផ្ទុកបានឡើយ)។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

### ក. ទម្រង់រួមនៃការខាតបង់ (Generalized Loss Formulation)

ការខាតបង់សរុប $L(W)$ គឺជាផលបូករវាង **ការខាតបង់ទិន្នន័យ (Data Loss)** ដែលវាស់វែងកម្រិតនៃការទស្សន៍ទាយត្រូវ និង **ការខាតបង់កែតម្រូវ (Regularization Loss)** ដែលដាក់ទោសលើភាពស្មុគស្មាញនៃម៉ូដែល៖

$$
L(W) = \underbrace{\frac{1}{N} \sum_{i=1}^{N} L_i\left(f(x_i, W), y_i\right)}_{\text{Data Loss (Model Fit)}} + \underbrace{\lambda R(W)}_{\text{Regularization Loss (Model Simplicity)}}
$$

ដែលក្នុងនោះ $\lambda \ge 0$ គឺជា **កម្រិតកែតម្រូវ (Regularization Strength Hyperparameter)**៖
* ប្រសិនបើ $\lambda = 0$: ម៉ូដែលព្យាយាមទស្សន៍ទាយឱ្យត្រូវ ១០០% លើ Training Set ដែលងាយប្រឈមនឹងបញ្ហា Overfitting។
* ប្រសិនបើ $\lambda$ ធំខ្លាំងពេក: ម៉ូដែលនឹងបង្រួមទម្ងន់ឱ្យនៅតូចបំផុត ដែលអាចនាំឱ្យមានបញ្ហា Underfitting (ម៉ូដែលមិនព្រមរៀនអ្វីទាំងអស់)។

### ខ. ប្រភេទនៃចំណុចលើផ្ទៃ Loss Landscape

* **ចំណុចអប្បបរមាក្នុងតំបន់ (Local Minimum)**: ទីតាំងដែលជម្រាល $\nabla L = 0$ ហើយគ្រប់ទិសដៅជុំវិញសុទ្ធតែមានកម្ពស់ខ្ពស់ជាង (Eigenvalues នៃ Hessian សុទ្ធតែវិជ្ជមាន)។
* **ចំណុចកែបសេះ (Saddle Point)**: ទីតាំងដែលជម្រាល $\nabla L = 0$ ប៉ុន្តែផ្ទៃដីកោងឡើងលើតាមអ័ក្សមួយ និងកោងចុះក្រោមតាមអ័ក្សមួយទៀត (Hessian មានទាំង Eigenvalues វិជ្ជមាន និងអវិជ្ជមាន)។ នៅក្នុងលំហពហុវិមាត្ររាប់លាន ចំណុចកែបសេះកើតឡើងញឹកញាប់ជាង Local Minima រាប់ពាន់ដង!
* **ស្ថានភាពជ្រលងមិនស្មើគ្នា (Ill-conditioned Surface)**: ផ្ទៃដីដែលមាន Condition Number ខ្ពស់ (កោងចោទខ្លាំងតាមទិសដៅមួយ តែរាបស្មើវែងអន្លាយតាមទិសដៅមួយទៀត)។

---

## ៣. មូលដ្ឋានគ្រឹះគណិតវិទ្យា & រូបមន្ត (Mathematical Foundations & Formulations)

### ក. បច្ចេកទេសកែតម្រូវទម្ងន់ (Weight Regularization Techniques)

1. **ការកែតម្រូវតាម $L2$ (L2 Regularization / Weight Decay / Tikhonov)**:  
   ដាក់ទោសលើផលបូកការ៉េនៃទម្ងន់ទាំងអស់៖
   $$
   R_{L2}(W) = \sum_{k} \sum_{l} W_{k,l}^2 = \|W\|_F^2
   $$
   នៅពេលធ្វើ Gradient Descent ជម្រាលរបស់វាគឺ $\nabla_W R_{L2} = 2W$ ដែលធ្វើឱ្យទម្ងន់ត្រូវបានទាញទម្លាក់ចុះជាលំដាប់លីនេអ៊ែរឆ្ពោះទៅកាន់សូន្យ ($W \leftarrow (1 - 2\alpha\lambda)W$)។

2. **ការកែតម្រូវតាម $L1$ (L1 Regularization / Lasso)**:  
   ដាក់ទោសលើផលបូកតម្លៃដាច់ខាតនៃទម្ងន់៖
   $$
   R_{L1}(W) = \sum_{k} \sum_{l} |W_{k,l}|
   $$
   ជម្រាលរបស់វាគឺ $\text{sign}(W)$ ដែលទាញទម្ងន់ចុះក្នុងអត្រាថេរ ដោយបង្ខំឱ្យទម្ងន់ដែលមិនសូវសំខាន់ធ្លាក់ចុះដល់សូន្យបេះបិទ បង្កើតបានជាម៉ូដែលដែលមានទម្ងន់ជ្រើសរើស (Sparse Weights)។

3. **ការកែតម្រូវកូនកាត់ Elastic Net**:  
   រួមបញ្ចូលទាំង $L1$ និង $L2$ ដើម្បីទទួលបានទាំង Sparsity និងស្ថិរភាពក្រុម៖
   $$
   R_{\text{Elastic}}(W) = \sum_{k} \sum_{l} \left( \beta W_{k,l}^2 + (1 - \beta) |W_{k,l}| \right)
   $$

---

### ខ. ការគណនាជម្រាល (Gradient Formulations)

* **Numerical Gradient (ផលធៀបកម្រិតកំណត់ Limit Definition)**:
  $$
  \frac{\partial L}{\partial W_{j}} \approx \frac{L(W + h \cdot e_j) - L(W - h \cdot e_j)}{2h}
  $$
  ដោយប្រើ Centered Difference Formula ជាមួយជំហានតូច $h \approx 10^{-5}$។

* **Analytical Gradient (វ៉ិចទ័រដេរីវេដោយផ្នែក - Gradient Vector)**:
  $$
  \nabla_W L = \left[ \frac{\partial L}{\partial W_{1}}, \frac{\partial L}{\partial W_{2}}, \dots, \frac{\partial L}{\partial W_{D}} \right]^T
  $$

---

### គ. ក្បួនដោះស្រាយបង្កើនប្រសិទ្ធភាពលំដាប់ទីមួយ (First-Order Optimizers)

#### ១. Stochastic Gradient Descent (SGD)
$$
W_{t+1} = W_t - \alpha \nabla_W L(W_t)
$$

#### ២. SGD ជាមួយសន្ទុះ Momentum
រក្សាទុកវ៉ិចទ័រល្បឿន $v$ ជាមួយមេគុណកកិត $\rho \in [0.9, 0.99]$៖
$$
v_{t+1} = \rho v_t + \nabla_W L(W_t)
$$
$$
W_{t+1} = W_t - \alpha v_{t+1}
$$

#### ៣. RMSprop (Root Mean Square Propagation)
រក្សាទុកមធ្យមភាគការ៉េនៃជម្រាល $s$ ជាមួយអត្រារំលាយ $\gamma \approx 0.99$៖
$$
s_{t+1} = \gamma s_t + (1 - \gamma) (\nabla_W L(W_t))^2
$$
$$
W_{t+1} = W_t - \frac{\alpha}{\sqrt{s_{t+1} + \epsilon}} \odot \nabla_W L(W_t)
$$
*(សញ្ញា $\odot$ តំណាងឱ្យផលគុណតាមធាតុ element-wise ហើយ $\epsilon \approx 10^{-8}$ ជួយការពារការចែកនឹងសូន្យ)*។

#### ៤. ក្បួនដោះស្រាយ Adam (Adaptive Moment Estimation)
តាមដានទាំង First Moment ($m_t$ - សន្ទុះ) និង Second Raw Moment ($v_t$ - RMSprop) ព្រមទាំងកែតម្រូវភាពលម្អៀង (Bias Correction)៖

$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla_W L(W_t) \quad \text{(First Moment / Momentum)}
$$
$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla_W L(W_t))^2 \quad \text{(Second Moment / RMSprop)}
$$

**ការកែតម្រូវភាពលម្អៀង (Unbiasing Corrections)**:
$$
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^t}, \quad \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^t}
$$

**ការធ្វើបច្ចុប្បន្នភាពទម្ងន់**:
$$
W_{t+1} = W_t - \frac{\alpha}{\sqrt{\hat{v}_{t+1}} + \epsilon} \odot \hat{m}_{t+1}
$$
*តម្លៃស្តង់ដារដែលប្រើទូទៅបំផុតក្នុងពិភពលោកគឺ៖ $\alpha = 10^{-3}$ ឬ $3 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, និង $\epsilon = 10^{-8}$*។

---

### ឃ. ក្បួនដោះស្រាយលំដាប់ទីពីរ (Newton-Raphson Optimization)

ការគណនាជំហានអប្បបរមាពិតប្រាកដដោយប្រើពង្រីក Taylor លំដាប់ទីពីរ និងម៉ាទ្រីសច្រាស Hessian $H^{-1}$៖

$$
W_{t+1} = W_t - H^{-1} \nabla_W L(W_t)
$$

ដែល $H_{i,j} = \frac{\partial^2 L}{\partial W_i \partial W_j}$។ វិធីសាស្ត្រនេះគ្មាន Hyperparameter Learning Rate $\alpha$ ទេ ពីព្រោះវាលោតទៅដល់បាតនៃទម្រង់ Parabolic quadratic ក្នុង ១ ជំហាន ប៉ុន្តែមិនអាចអនុវត្តលើ Deep Learning ដោយសារការគណនា $H^{-1}$ មានភាពស្មុគស្មាញ $O(N^3)$ និងការផ្ទុកមានទំហំ $O(N^2)$។

---

## ៤. ស្ថាបត្យកម្ម និងក្បួនដោះស្រាយ (Architecture & Algorithm Walkthrough)

### ដ្យាក្រាមរង្វិលជុំនៃការហ្វឹកហាត់ (Training Loop & Optimizer Epoch)

```text
[ចាប់ផ្តើមទម្ងន់ដើម W_0]
         │
         ▼
  ┌───────────────┐
  │ Mini-Batch    │ ◄─────────────────────────┐
  │ Sampling      │                           │
  └───────┬───────┘                           │
          │ ទាញយកឧទាហរណ៍ (Batch Size = 128)  │
          ▼                                   │
  ┌───────────────┐                           │
  │ Forward Pass  │                           │
  │ Compute Loss  │                           │
  └───────┬───────┘                           │
          │ L = Data Loss + λ * R(W)          │
          ▼                                   │ រង្វិលជុំ Epochs
  ┌───────────────┐                           │
  │ Backward Pass │                           │
  │ Compute dW    │                           │
  └───────┬───────┘                           │
          │ Analytic Backpropagation          │
          ▼                                   │
  ┌───────────────┐                           │
  │ Optimizer Step│                           │
  │ Update W      │ ──────────────────────────┘
  └───────────────┘
    W = W - Step_Update(m_hat, v_hat)
```

---

### កូដគំរូស្ថាបត្យកម្មគ្រឹះ (PyTorch Implementation: Adam & AdamW from Scratch)

កូដ PyTorch ខាងក្រោមបង្ហាញពីការសរសេរក្បួនដោះស្រាយ **Adam** និង **AdamW** (Decoupled Weight Decay) ពីកម្រិតសូន្យ ដើម្បីបង្ហាញយ៉ាងច្បាស់ពីយន្តការតាមដាន Moments ការកែតម្រូវ Bias និងរបៀបដែល AdamW ដោះស្រាយបញ្ហាខូចទ្រង់ទ្រាយនៃ Weight Decay៖

```python
import torch
import torch.nn as nn

class CustomAdamWOptimizer:
    """
    ការអនុវត្តក្បួនដោះស្រាយ Adam និង AdamW ពីកម្រិតគ្រឹះ (Scratch)
    ដើម្បីបង្ហាញពីយន្តការតាមដាន Momentum (First Moment), RMSprop (Second Moment),
    ការកែតម្រូវភាពលម្អៀង (Bias Correction) និង Decoupled Weight Decay។
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4, use_adamw=True):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.use_adamw = use_adamw
        self.t = 0
        
        # កំណត់តម្លៃ Moments ដំបូងជាសូន្យសម្រាប់គ្រប់ប៉ារ៉ាម៉ែត្រទាំងអស់
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            
            # ១. ដោះស្រាយ Weight Decay
            if not self.use_adamw:
                # នៅក្នុង Standard Adam: L2 Regularization ត្រូវបានបូកបញ្ចូលផ្ទាល់ទៅក្នុងជម្រាល
                grad = grad + self.wd * p.data
            
            # ២. ធ្វើបច្ចុប្បន្នភាព First Moment (Momentum)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            
            # ៣. ធ្វើបច្ចុប្បន្នភាព Second Raw Moment (RMSprop)
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            
            # ៤. គណនា Bias-Corrected Moments (កែតម្រូវភាពលម្អៀងដំបូង)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            
            # ៥. គណនាជំហានផ្លាស់ប្តូរ
            step_update = m_hat / (torch.sqrt(v_hat) + self.eps)
            
            if self.use_adamw:
                # នៅក្នុង AdamW: Decoupled Weight Decay ត្រូវបានអនុវត្តផ្ទាល់លើទម្ងន់តែម្តង
                p.data = p.data - self.lr * self.wd * p.data - self.lr * step_update
            else:
                p.data = p.data - self.lr * step_update

# ការសាកល្បងដំណើរការលើម៉ូដែលតូចមួយ
if __name__ == "__main__":
    linear_layer = nn.Linear(10, 2)
    optimizer = CustomAdamWOptimizer(linear_layer.parameters(), lr=0.01, use_adamw=True)
    
    # បង្កើតទិន្នន័យគំរូ
    x = torch.randn(8, 10)
    target = torch.randn(8, 2)
    
    # Forward Pass & Backward
    out = linear_layer(x)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    
    # ដំណើរការ Optimizer Step
    optimizer.step()
    print(f"ជំហានទី {optimizer.t} បានបញ្ចប់ដោយជោគជ័យ! តម្លៃ Loss: {loss.item():.4f}")
```

---

## ៥. ការយល់ដឹងស៊ីជម្រៅតាមបែបវិចារណញាណ (Visual Intuition & Interpretability)

### យន្តការជ្រើសរើសទម្ងន់នៃ $L2$ (Diffuse Weight Preference Micro-Example)

ឧបមាថាយើងមានវ៉ិចទ័រធាតុចូលភីកសែល $x = [1, 1, 1, 1]^T$ ហើយមានជម្រើសវ៉ិចទ័រទម្ងន់ពីរដែលត្រូវប្រកួតប្រជែងគ្នា៖
* $W_1 = [1, 0, 0, 0]^T$
* $W_2 = [0.25, 0.25, 0.25, 0.25]^T$

តោះប្រៀបធៀបលទ្ធផលគណិតវិទ្យា៖
1. **ពិន្ទុលីនេអ៊ែរ (Linear Score)**: $W_1 \cdot x = 1$ និង $W_2 \cdot x = 0.25 \times 4 = 1$ (ស្មើគ្នាទាំងស្រុង!)។
2. **ការផាកពិន័យតាម $L1$**: $R(W_1) = |1| = 1$ និង $R(W_2) = 4 \times |0.25| = 1$ (ស្មើគ្នាទាំងស្រុង!)។
3. **ការផាកពិន័យតាម $L2$**:
   * $R(W_1) = 1^2 + 0 + 0 + 0 = \mathbf{1.0}$
   * $R(W_2) = 0.25^2 + 0.25^2 + 0.25^2 + 0.25^2 = 4 \times 0.0625 = \mathbf{0.25}$

**វិចារណញាណរូបភាព**: $L2$ ចូលចិត្តវ៉ិចទ័រ $W_2$ ខ្លាំងជាង $W_1$ ដល់ទៅ ៤ ដង! ពីព្រោះ $L2$ ស្អប់ទម្ងន់ណាដែលប្រមូលផ្តុំលើភីកសែលតែមួយ (Peaky Weights)។ ការរាយទម្ងន់ស្មើៗគ្នាលើគ្រប់ភីកសែលធ្វើឱ្យម៉ូដែល **មានភាពរឹងមាំប្រឆាំងនឹងការប្រែប្រួលភីកសែលចៃដន្យ (Adversarial Robustness)** ហើយបង្ខំឱ្យម៉ូដែលមើលទិដ្ឋភាពជារួម (Holistic scene context) ជំនួសឱ្យការទន្ទេញចំណុចតូចមួយ។

---

### ការប្រៀបធៀបទម្រង់នៃការបរាជ័យលើ Loss Landscape

```text
ជ្រលងភ្នំចោទមិនស្មើគ្នា (High Condition Number)        ចំណុចកែបសេះ (Saddle Point Zero-Gradient Stall)
             /\  ជញ្ជាំងចោទខ្លាំង                                  
            /  \                                                      _  _
   ───►    /    \    ◄─── លំយោលទៅមកខ្លាំង (SGD)                      ( \/ )  ◄── កោងឡើងលើតាមអ័ក្សមួយ
          /   /\ \                                                    \  /
         /   /  \ \                                                   (  )   ◄── ជម្រាលស្មើ ០ នៅចំកណ្តាល (g = 0)
        /___/    _\                                                  /    \  ◄── កោងចុះក្រោមតាមអ័ក្សមួយទៀត
           បាតជ្រលងភ្នំ (រាបស្មើវែងអន្លាយ)
```

1. **ស្ថានភាពជ្រលងមិនស្មើគ្នា (Poor Conditioning)**: ជញ្ជាំងសងខាងចោទខ្លាំង ប៉ុន្តែបាតជ្រលងភ្នំរាបស្មើវែងអន្លាយ។ SGD ធម្មតានឹងលោតបោកបក់រវាងជញ្ជាំងសងខាង (Oscillation) ដោយសារជម្រាលចោទកាត់កែងបោះជំហានធំជ្រុល លុបគ្នាទៅវិញទៅមក ហើយស្ទើរតែមិនរំកិលទៅមុខតាមបណ្តោយបាតជ្រលងភ្នំឡើយ។
2. **អន្ទាក់ចំណុចកែបសេះ (Saddle Point Trap)**: នៅចំកណ្តាល ជម្រាលស្មើនឹងសូន្យបេះបិទ ($\nabla L = 0$)។ SGD ធម្មតាដែលធ្លាក់ចូលចំណុចនេះ នឹងគិតថាវាបានដល់ទីដៅហើយ រួចគាំងជាប់នៅទីនោះរហូត។ ប៉ុន្តែ Momentum ប្រើល្បឿនសន្ទុះដែលនៅសល់រុញម៉ូដែលឱ្យរអិលផុតចំណុចនេះ ហើយបន្តធ្លាក់ចុះទៅតាមជម្រាលចោទនៃអ័ក្សម្ខាងទៀត។

---

<div id="plotly-cs231n-3-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 3 visualization blueprint"></div>
<p><em>រូបភាព៖ គំនូសតាងអន្តរកម្មបង្ហាញពីគន្លងនៃ Optimizers ផ្សេងៗ (SGD, Momentum, Adam) លើផ្ទៃ Loss Landscape មិនស្មើគ្នា។</em></p>

## ៦. ការពិសោធន៍ជាក់ស្តែង និងលទ្ធផលគោលស្តង់ដារ (Empirical Heuristics & Benchmark Results)

ការប្រៀបធៀបសមត្ថភាពនៃយុទ្ធសាស្ត្របង្កើនប្រសិទ្ធភាពលើសំណុំទិន្នន័យ CIFAR-10៖

| យុទ្ធសាស្ត្របង្កើនប្រសិទ្ធភាព | ប្រភេទក្បួនដោះស្រាយ | ភាពត្រឹមត្រូវលើ CIFAR-10 | លក្ខណៈពិសេសនៃការបញ្ចូលគ្នា |
| :--- | :--- | :---: | :--- |
| **ការស្វែងរកដោយចៃដន្យ (Random Search)** | Non-Gradient Heuristic | **15.5%** | បរាជ័យក្នុងលំហពហុវិមាត្រ |
| **Vanilla SGD** | First-Order Basic | **~40.0%** (Linear) / **~72.0%** (CNN) | ងាយនឹងយឺត និងលំយោលនៅជ្រលងចោទ |
| **SGD ជាមួយ Momentum** | First-Order Velocity | **~42.0%** (Linear) / **~84.0%** (CNN) | រំលងផុត Saddle Points យ៉ាងមានប្រសិទ្ធភាព |
| **Adam / AdamW** | Adaptive First & Second Moments | **~44.0%** (Linear) / **~93.0%+** (CNN) | ស្តង់ដារមាសនៃ Deep Learning ពិភពលោក |
| **State-of-the-Art Deep Models (ViT/ConvNeXt)** | Advanced Architectures + AdamW | **99.7%** | ដោះស្រាយបញ្ហា CIFAR-10 ទាំងស្រុង |

* **ច្បាប់ពង្រីកល្បឿនរៀនលីនេអ៊ែរ (Linear Learning Rate Scaling Law)**: នៅពេលអ្នកពង្រីកទំហំ Batch Size ឡើង $N$ ដងក្នុងការហ្វឹកហាត់ស្របគ្នាលើ GPU ច្រើន (Distributed Data Parallel) អ្នកត្រូវតែបង្កើនល្បឿនរៀន $\alpha$ ឡើង $N$ ដងដូចគ្នា ($\alpha_{\text{new}} = N \cdot \alpha_{\text{base}}$) ដើម្បីធានាថាកម្លាំងនៃជម្រាលនៅតែមានទំហំសមាមាត្រត្រឹមត្រូវ។
* **កិត្តិយសរបស់ Adam**: ក្បួនដោះស្រាយ Adam ត្រូវបានបោះពុម្ពផ្សាយដំបូងក្នុងឆ្នាំ ២០១៥ (Kingma & Ba) ហើយបានទទួលពានរង្វាន់កិត្តិយសដ៏ខ្ពង់ខ្ពស់ **ICLR 2025 Test of Time Award** ដោយសារឥទ្ធិពលដ៏ធំធេងដែលជំរុញឱ្យបដិវត្តន៍ Deep Learning ទំនើប និង Generative AI ដំណើរការទៅបាន។

---

## ៧. ឧបសគ្គដែលត្រូវប្រុងប្រយ័ត្ន និងគន្លឹះដោះស្រាយកំហុស (Pitfalls & Debugging Tips)

* **បញ្ហាជំហានដំបូងរបស់ Adam (Zero-Initialization Step Spike)**:  
  នៅជំហានដំបូង ($t = 1$) តម្លៃ Moments $m$ និង $v$ ត្រូវបានកំណត់ជាសូន្យ។ ប្រសិនបើគ្មានការកែតម្រូវ Bias Correction ($/(1 - \beta^t)$) ទេ ភាគបែង $\sqrt{v_1} + \epsilon$ នឹងមានតម្លៃតូចបំផុត ដែលបណ្តាលឱ្យជំហានទីមួយ $\frac{\alpha}{\sqrt{v_1}} m_1$ ផ្ទុះឡើងជាជំហានដ៏ធំសម្បើម (Spike) បំផ្លាញតម្លៃទម្ងន់ដើមទាំងស្រុង!
* **ច្រាំងចោទនៃល្បឿនរៀន (The Learning Rate Cliff)**:
  * **បើ $\alpha$ ធំពេក**: ម៉ូដែលនឹងលោតផ្លាតចេញពីជ្រលងភ្នំ បណ្តាលឱ្យក្រាហ្វ Loss កើនឡើងយ៉ាងគំហុក ឬចេញតម្លៃ `NaN` (Loss Exploding)។
  * **បើ $\alpha$ តូចពេក**: ក្រាហ្វ Loss ធ្លាក់ចុះរាបស្មើស្ទើរតែជាបន្ទាត់ត្រង់ដេក ធ្វើឱ្យការហ្វឹកហាត់ស៊ីពេលរាប់សប្តាហ៍ដោយមិនទៅណាឡើយ។
* **គន្លឹះត្រួតពិនិត្យជម្រាល (Gradient Checking Best Practices)**:
  * ត្រូវប្រើរូបមន្ត Centered Difference ជានិច្ច មិនត្រូវប្រើ One-sided Difference ឡើយ។
  * ត្រូវប្រើកម្រិតលម្អៀងធៀប (Relative Error) ដើម្បីវាយតម្លៃ៖
    $$
    \frac{|g_{\text{numerical}} - g_{\text{analytical}}|}{\max(|g_{\text{numerical}}|, |g_{\text{analytical}}|)}
    $$
    * បើផលធៀប $< 10^{-7}$: ជម្រាលរបស់អ្នកត្រឹមត្រូវ ១០០%។
    * បើផលធៀប $> 10^{-4}$: មានបញ្ហាធ្ងន់ធ្ងរក្នុងរូបមន្ត Backpropagation។

---

## ៨. សំណួរពិចារណាកម្រិតក្រោយឧត្តមសិក្សា (Graduate-Level Reflection Questions)

១. **ភាពខុសគ្នារវាង Adam និង AdamW (Adam vs. AdamW Dynamics)**:  
   នៅក្នុង Standard Adam ការកែតម្រូវ $L2$ ត្រូវបានបូកបញ្ចូលទៅក្នុងជម្រាលដើម ($\nabla L + \lambda W$) មុនពេលគណនា Second Moment $v_t$។ ហេតុអ្វីបានជាការធ្វើបែបនេះបណ្តាលឱ្យទម្ងន់ណាដែលមានជម្រាលប្រវត្តិសាស្ត្រធំ ត្រូវបានទទួលរងការផាកពិន័យ Weight Decay តិចជាងទម្ងន់ដែលមានជម្រាលតូច? តើរូបមន្ត Decoupled Weight Decay របស់ AdamW ($W_{t+1} = W_t(1 - \alpha \lambda) - \text{AdamStep}$) ស្តារឡើងវិញនូវចេតនាពិតប្រាកដនៃ Weight Decay យ៉ាងដូចម្តេច?

២. **ការកើនឡើងនៃចំណុចកែបសេះក្នុងលំហពហុវិមាត្រ (Saddle Point Proliferation)**:  
   នៅក្នុងលំហ 1D/2D ចំណុចអប្បបរមាក្នុងតំបន់ (Local Minima) កើតឡើងញឹកញាប់។ ប៉ុន្តែហេតុអ្វីបានជាសមាមាត្រនៃ Saddle Points កើនឡើងជាអនុគមន៍ស្វ័យគុណធៀបនឹង Local Minima នៅពេលបណ្តាញញាណកើនឡើងពី ១ លាន ដល់រាប់ពាន់លានប៉ារ៉ាម៉ែត្រ? ចូរពន្យល់តាមរយៈបំណែងចែកសញ្ញានៃ Eigenvalues ក្នុងម៉ាទ្រីស Hessian។

៣. **រូបវិទ្យានៃ Momentum ធៀបនឹង RMSprop ក្នុងជ្រលងភ្នំចោទ**:  
   នៅពេលធ្វើដំណើរឆ្លងកាត់ជ្រលងភ្នំដែលមាន Condition Number ខ្ពស់ ទាំង Momentum និង RMSprop សុទ្ធតែទប់ស្កាត់លំយោលបាន ប៉ុន្តែដំណើរការតាមរូបវិទ្យាគណិតវិទ្យាខុសគ្នាស្រឡះ។ ចូរពន្យល់ពីរបៀបដែល Momentum ប្រើប្រាស់ការលុបចោលគ្នានៃវ៉ិចទ័រល្បឿនសងខាង (Vector Cancellation) ខណៈដែល RMSprop ប្រើប្រាស់ការបង្រួមកូអរដោនេតាមទំហំជម្រាល (Coordinate Contraction)។ តើក្បួនមួយណាដែលងាយរងគ្រោះនឹង Gradient Noise ជាង ហើយហេតុអ្វី?
