# កំណត់ចំណាំមេរៀន: មេរៀនទី ៦: ស្ថាបត្យកម្ម CNN និងការហ្វឹកហាត់បណ្តាញញាណ (CNN Architectures & Training Neural Networks)

**វគ្គសិក្សា**: Stanford CS231n: Deep Learning for Computer Vision (ការរៀនស៊ីជម្រៅសម្រាប់ការមើលឃើញដោយកុំព្យូទ័រ — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Prof. Fei-Fei Li (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Ranjay Krishna (សាស្ត្រាចារ្យជំនួយផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs231n.stanford.edu](https://cs231n.stanford.edu) (ស្លាយមេរៀន ឯកសារសិក្សា និងកិច្ចការអនុវត្ត)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Quick-Recall Summary)

* **របកគំហើញដោះស្រាយបញ្ហារាំងស្ទះនៃជម្រៅ (The Depth Barrier Resolved)**: បណ្តាញញាណធម្មតាដែលគ្មានផ្លូវកាត់ (Plain Deep CNNs) ជួបប្រទះបញ្ហាធ្លាក់ចុះគុណភាពនៃការបង្កើនប្រសិទ្ធភាព (Optimization Degradation Problem) ដែលស្រទាប់កាន់តែជ្រៅមិនអាចរៀនសូម្បីតែអនុគមន៍អត្តសញ្ញាណ (Identity Mapping) ធម្មតា។ ស្ថាបត្យកម្ម **ResNet** បានដោះស្រាយបញ្ហានេះតាមរយៈ Residual Skip Connections ($H(x) = F(x) + x$) ដែលជួយឱ្យការហ្វឹកហាត់បណ្តាញញាណជ្រៅជាង ១០០ ស្រទាប់ដំណើរការដោយជោគជ័យ។
* **ប្រសិទ្ធភាពនៃការជង់តម្រងតូចៗ (VGG Stacking Efficiency)**: ស្រទាប់ Convolution $3 \times 3$ ចំនួន ៣ ស្រទាប់ជង់លើគ្នា មាន Receptive Field ស្មើនឹងស្រទាប់ $7 \times 7$ មួយ ប៉ុន្តែសន្សំប៉ារ៉ាម៉ែត្របានប្រមាណ **៤៥%** ព្រមទាំងផ្តល់នូវអនុគមន៍មិនលីនេអ៊ែរ (Non-linearities) ចំនួន ៣ ដង ជំនួសឱ្យម្តង។
* **ភាពខុសគ្នានៃបច្ចេកទេស Normalization (BN vs. LN)**: 
  * **Batch Normalization (BN)**: គណនាស្ថិតិមធ្យមភាគ និងវ៉ារ្យ៉ង់ឆ្លងកាត់ Batch និងវិមាត្រលំហ ($N \times H \times W$) សម្រាប់ Channel នីមួយៗដាច់ដោយឡែក។
  * **Layer Normalization (LN)**: គណនាស្ថិតិឆ្លងកាត់ Channels និងវិមាត្រលំហ ($C \times H \times W$) សម្រាប់គំរូនីមួយៗក្នុង Batch ដោយឯករាជ្យ ដែលស័ក្តិសមបំផុតសម្រាប់ Transformers និងទិន្នន័យដែលមានប្រវែងមិនថេរ។
* **ការកំណត់តម្លៃទម្ងន់ដើម Kaiming (He) Initialization**: ដើម្បីទប់ស្កាត់បញ្ហាជម្រាលសាបរលាប (Vanishing) ឬផ្ទុះឡើង (Exploding) ក្នុងបណ្តាញញាណដែលប្រើ ReLU ទម្ងន់ត្រូវតែកំណត់ចេញពីបំណែងចែក Gaussian ដែលមានមធ្យមភាគសូន្យ និងវ៉ារ្យ៉ង់ $\sigma^2 = \frac{2}{D_{\text{in}}}$។
* **យុទ្ធសាស្ត្រផ្ទេរការរៀនសូត្រ (Transfer Learning Paradigm)**: ប្រសិនបើសំណុំទិន្នន័យថ្មីមានទំហំតូច យើងត្រូវបង្កកឆ្អឹងខ្នង (Freeze Backbone) ហើយហ្វឹកហាត់តែស្រទាប់ចុងក្រោយ (Linear Probing)។ ប្រសិនបើសំណុំទិន្នន័យមានទំហំធំ យើងអាចធ្វើ Fine-tuning លើបណ្តាញញាណទាំងមូលដោយប្រើ Learning Rate តូច។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

### ក. បញ្ហាធ្លាក់ចុះគុណភាពនៃការបង្កើនប្រសិទ្ធភាព (Optimization Degradation Problem)
នៅពេលយើងបន្ថែមស្រទាប់កាន់តែច្រើនទៅក្នុងបណ្តាញញាណធម្មតា (Plain Networks) កំហុសនៃការហ្វឹកហាត់ (Training Error) បែរជាកើនឡើងខ្ពស់ជាងបណ្តាញញាណរាក់ (ឧ. Plain-56 មាន Training Error ខ្ពស់ជាង Plain-20)។ នេះ **មិនមែនបណ្តាលមកពី Overfitting ទេ** (ពីព្រោះ Training Error ខ្លួនឯងក៏ខ្ពស់ដែរ) ប៉ុន្តែវាជាបញ្ហាបរាជ័យនៃក្បួន Optimization ដែលជួបការលំបាកក្នុងការរុញជម្រាលឆ្លងកាត់ស្រទាប់ច្រើនពេក និងមិនអាចរៀន Identity Mapping បាន។

### ខ. យន្តការ Residual Skip Connection
ជំនួសឱ្យការបង្ខំឱ្យស្រទាប់រៀនស្ថាបត្យកម្មគោលដៅ $H(x)$ ដោយផ្ទាល់ ResNet បំប្លែងវាឱ្យរៀនភាពខុសគ្នា (Residual Mapping)៖
$$
F(x) = H(x) - x \implies H(x) = F(x) + x
$$
* ប្រសិនបើ Identity Mapping គឺជាស្ថានភាពល្អបំផុត នោះក្បួនបង្កើនប្រសិទ្ធភាពគ្រាន់តែទាញទម្ងន់នៃ $F(x)$ ឱ្យធ្លាក់ចុះដល់សូន្យ នោះ $H(x)$ នឹងក្លាយជា Identity $x$ ដោយស្វ័យប្រវត្តិ។
* Skip Connection បង្កើតផ្លូវល្បឿនលឿនសម្រាប់ជម្រាល (Gradient Superhighway) ក្នុងពេល Backpropagation ដែលការពារកុំឱ្យជម្រាលសាបរលាបនៅតាមផ្លូវ។

### គ. ស្ថិតិនៃការកែតម្រូវ Normalization (BN vs. LN)
សម្រាប់ Tensor ធាតុចូល $X \in \mathbb{R}^{N \times C \times H \times W}$ ប្រមាណវិធី Normalization បំប្លែង $X$ ទៅជា $\hat{X}$ ហើយអនុវត្តប៉ារ៉ាម៉ែត្ររៀន Scale $\gamma$ និង Shift $\beta$៖
$$
\hat{X}_{n,c,h,w} = \frac{X_{n,c,h,w} - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad Y_{n,c,h,w} = \gamma \hat{X}_{n,c,h,w} + \beta
$$

* **Batch Normalization (BN)**:
  $$
  \mu_{\text{BN}}(c) = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^N \sum_{h=1}^H \sum_{w=1}^W X_{n,c,h,w}
  $$
  $$
  \sigma^2_{\text{BN}}(c) = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^N \sum_{h=1}^H \sum_{w=1}^W (X_{n,c,h,w} - \mu_{\text{BN}}(c))^2
  $$

* **Layer Normalization (LN)**:
  $$
  \mu_{\text{LN}}(n) = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^C \sum_{h=1}^H \sum_{w=1}^W X_{n,c,h,w}
  $$
  $$
  \sigma^2_{\text{LN}}(n) = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^C \sum_{h=1}^H \sum_{w=1}^W (X_{n,c,h,w} - \mu_{\text{LN}}(n))^2
  $$

---

## ៣. មូលដ្ឋានគ្រឹះគណិតវិទ្យា & រូបមន្ត (Mathematical Foundations & Formulations)

### ក. នព្វន្ធ Receptive Field និងការសន្សំប៉ារ៉ាម៉ែត្រ
សម្រាប់ស្រទាប់ Convolution ដែលមាន Kernel $K$ និង Stride $S=1$ ការជង់ $L$ ស្រទាប់ពង្រីក Receptive Field ($RF$) តាមរូបមន្តរំលឹក៖
$$
RF_L = 1 + L(K - 1)
$$
* សម្រាប់ស្រទាប់ $3 \times 3$ ចំនួន ៣ ($L=3, K=3$):
  $$
  RF_3 = 1 + 3(3 - 1) = 1 + 6 = 7
  $$

**ការប្រៀបធៀបចំនួនប៉ារ៉ាម៉ែត្រ (Channels $C$ ថេរ)**:
* **ស្រទាប់ $3 \times 3$ ចំនួន ៣ ស្រទាប់**:
  $$
  \text{Params}_{\text{stack}} = 3 \times (3 \times 3 \times C \times C) = 27 C^2
  $$
* **ស្រទាប់ $7 \times 7$ តែមួយស្រទាប់**:
  $$
  \text{Params}_{\text{single}} = 7 \times 7 \times C \times C = 49 C^2
  $$
* **អត្រាសន្សំប៉ារ៉ាម៉ែត្រ**:
  $$
  \text{Savings} = 1 - \frac{27 C^2}{49 C^2} = 1 - \frac{27}{49} \approx 44.9\%
  $$

---

### ខ. ទ្រឹស្តីកំណត់តម្លៃទម្ងន់ដើម Kaiming (He) Initialization
ពិចារណាស្រទាប់លីនេអ៊ែរ $y = W x$។ ដើម្បីធានាថាកម្រិតវ៉ារ្យ៉ង់នៃសកម្មកម្មមិនរីកធំ ឬរួញតូចឆ្លងកាត់ស្រទាប់ជ្រៅ យើងទាមទារឱ្យ $\operatorname{Var}(y) = \operatorname{Var}(x)$៖
$$
\operatorname{Var}(y_i) = D_{\text{in}} \cdot \operatorname{Var}(w_{ij}) \cdot \mathbb{E}[x_j^2]
$$

ចំពោះអនុគមន៍សកម្មកម្ម **ReLU** ពាក់កណ្តាលនៃតម្លៃធាតុចូលត្រូវបានកាត់ចោលស្មើនឹងសូន្យ ហេតុនេះ $\mathbb{E}[x^2] = \frac{1}{2} \operatorname{Var}(x)$។  
ជំនួសចូលក្នុងសមីការ៖
$$
\operatorname{Var}(y) = D_{\text{in}} \cdot \operatorname{Var}(w) \cdot \frac{1}{2} \operatorname{Var}(x) \implies \operatorname{Var}(w) = \frac{2}{D_{\text{in}}}
$$

ដូចនេះ ទម្ងន់ត្រូវតែកំណត់តាមបំណែងចែកប្រក្រតី៖
$$
W \sim \mathcal{N}\left(0, \frac{2}{D_{\text{in}}}\right) \quad \text{ឬ} \quad \sigma = \sqrt{\frac{2}{D_{\text{in}}}}
$$
* សម្រាប់ស្រទាប់ Fully Connected: $D_{\text{in}} = \text{fan\_in}$
* សម្រាប់ស្រទាប់ Convolutional: $D_{\text{in}} = K_h \times K_w \times C_{\text{in}}$

---

### គ. គណិតវិទ្យានៃ Dropout & Inverted Dropout
Dropout កាត់បន្ថយ Overfitting ដោយទម្លាក់ណឺរ៉ូនចៃដន្យចោលក្នុងអត្រាប្រូបាប $p$៖
* **Standard Dropout**:
  * ពេល Training: $y_{\text{train}} = x \odot M$ (ដែល $M_i \sim \text{Bernoulli}(1-p)$)
  * ពេល Testing: $y_{\text{test}} = (1 - p) \cdot x$ (តម្រូវឱ្យគុណនឹង $1-p$ លើគ្រប់ណឺរ៉ូនដើម្បីរក្សាតម្លៃរំពឹងទុក)
* **Inverted Dropout (ស្តង់ដារក្នុង PyTorch)**:
  * ពេល Training: ធ្វើ Scaling តាំងពីពេលហ្វឹកហាត់ $y_{\text{train}} = \frac{x \odot M}{1 - p}$
  * ពេល Testing: មិនចាំបាច់ធ្វើអ្វីទាំងអស់ $y_{\text{test}} = x$ (សន្សំពេលវេលាគណនាពេល Inference)

---

## ៤. ស្ថាបត្យកម្ម និងក្បួនដោះស្រាយ (Architecture & Algorithm Walkthrough)

### ការប្រៀបធៀបរចនាសម្ព័ន្ធ VGG vs. ResNet Block

```text
VGG Block (Plain Feedforward)               ResNet Block (Residual Identity)
         [Input X]                                     [Input X]
             │                                         │       ╲
             ▼                                         │        ╲ (Skip Connection)
      [3x3 Conv, Pad 1]                                │         │
             │                                         ▼         │
             ▼                                 [3x3 Conv, Pad 1] │
          [ReLU]                                       │         │
             │                                         ▼         │
             ▼                                      [ReLU]       │
      [3x3 Conv, Pad 1]                                │         │
             │                                         ▼         │
             ▼                                 [3x3 Conv, Pad 1] │
          [ReLU]                                       │         │
             │                                         ▼         │
             ▼                                      [Add] ◄──────┘
         [Output Y]                                    │
                                                       ▼
                                                    [ReLU]
                                                       │
                                                       ▼
                                                   [Output Y]
```

---

### កូដគំរូស្ថាបត្យកម្មគ្រឹះ (PyTorch Implementation: VGG & ResNet Blocks)

កូដខាងក្រោមបង្ហាញពីការកសាងស្ថាបត្យកម្ម VGG Block និង ResNet Residual Block ដោយរួមបញ្ចូលទាំង Kaiming Initialization, Batch Normalization, និង Projection Shortcut៖

```python
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    """
    ប្លុកស្ថាបត្យកម្មបែប VGG: ការជង់តម្រង 3x3 Convolutions ជាមួយ BatchNorm និង ReLU
    បញ្ចប់ដោយ Max Pooling ដើម្បីកាត់បន្ថយទំហំលំហពាក់កណ្តាល។
    """
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            cin = in_channels if i == 0 else out_channels
            # K=3, P=1, S=1 រក្សាទំហំលំហ H x W ឱ្យនៅដដែល
            layers.append(nn.Conv2d(cin, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        # អនុវត្ត Kaiming (He) Normal Initialization សម្រាប់ ReLU
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.block(x)


class ResNetBlock(nn.Module):
    """
    ប្លុក Residual Block ស្តង់ដាររបស់ ResNet ជាមួយ Projection Shortcut
    នៅពេលដែលចំនួន Channels កើនឡើង ឬទំហំលំហត្រូវកាត់បន្ថយតាមរយៈ Stride=2។
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut Connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # ប្រើ 1x1 Conv ដើម្បីផ្គូផ្គងទំហំលំហ និងចំនួន Channels
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        identity = self.shortcut(x) # ផ្លូវកាត់ Identity Skip
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # បូកបញ្ចូលគ្នាតាមធាតុ (Residual Connection)
        out += identity
        out = self.relu(out)
        return out

# ការសាកល្បងដំណើរការ
if __name__ == "__main__":
    sample_tensor = torch.randn(2, 64, 32, 32)
    
    # សាកល្បង ResNet Block ជាមួយ Stride=2 (Downsampling)
    res_block = ResNetBlock(in_channels=64, out_channels=128, stride=2)
    out_res = res_block(sample_tensor)
    
    print(f"ទំហំ Input: {list(sample_tensor.shape)}")
    print(f"ទំហំ Output ResNet Block (Downsampled): {list(out_res.shape)}")
```

---

## ៥. ការយល់ដឹងស៊ីជម្រៅតាមបែបវិចារណញាណ (Visual Intuition & Interpretability)

### វិចារណញាណនៃបញ្ហា Optimization Degradation

```text
  Training Error នៃ Plain Networks                    Training Error នៃ ResNet
        |                                                   |
 Error  |      /─── Plain-56 (កំហុសខ្ពស់!)           Error  |      /─── Plain-20
        |     /                                             |     /
        |    /───── Plain-20                                |    /───── ResNet-20
        |   /                                               |   /
        |  /                                                |  /─────── ResNet-56 (កំហុសទាបបំផុត!)
        +───────────────────────── Epochs                   +───────────────────────── Epochs
```

* នៅក្នុង Plain Networks: បណ្តាញជ្រៅ 56 ស្រទាប់មានកំហុសហ្វឹកហាត់ខ្ពស់ជាងបណ្តាញ 20 ស្រទាប់។ នេះបញ្ជាក់ថាក្បួន Optimization បរាជ័យក្នុងការស្វែងរកដំណោះស្រាយ។
* នៅក្នុង ResNet: ការបន្ថែមស្រទាប់កាន់តែជ្រៅ ធានាថាកំហុសហ្វឹកហាត់កាន់តែទាប ពីព្រោះបណ្តាញអាចប្រើផ្លូវកាត់ Identity ដើម្បីរំលងស្រទាប់ដែលមិនចាំបាច់បានយ៉ាងងាយ។

---

### ទិដ្ឋភាពនៃការកំណត់ទម្ងន់ដើម (Weight Initialization Dynamics)

* **Under-initialization ($\sigma = 0.01$)**: សកម្មកម្មធ្លាក់ចុះជាអនុគមន៍ស្វ័យគុណឆ្ពោះទៅកាន់សូន្យ។ នៅស្រទាប់ទី ១០ សកម្មកម្មទាំងអស់ស្មើនឹង $0$ បណ្តាលឱ្យជម្រាលសាបរលាបបាត់បង់ទាំងស្រុង។
* **Over-initialization ($\sigma = 0.05$)**: សកម្មកម្មកើនឡើងជាអនុគមន៍ស្វ័យគុណឆ្ពោះទៅរកអនន្ត ($\infty$) ធ្វើឱ្យតម្លៃក្នុងស្រទាប់ឆ្អែត ឬផ្ទុះចេញតម្លៃ `NaN` (Exploding Gradients)។
* **Kaiming (He) Balance**: រក្សាកម្រិតវ៉ារ្យ៉ង់នៃសកម្មកម្មឱ្យនៅថេរស្មើៗគ្នាគ្រប់ស្រទាប់ទាំងអស់ ទោះបីជាបណ្តាញញាណមានរាប់សិបស្រទាប់ក៏ដោយ។

---

<div id="plotly-cs231n-6-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 6 visualization blueprint"></div>
<p><em>រូបភាព៖ គំនូសតាងអន្តរកម្មបង្ហាញពីស្ថេរភាពនៃគម្លាតគំរូនៃសកម្មកម្ម (Activation Std) តាមជម្រៅស្រទាប់ រវាងម៉ូដែលគ្មាន Normalization ធៀបនឹង Batch Normalization។</em></p>

---

## ៦. ការពិសោធន៍ជាក់ស្តែង និងលទ្ធផលគោលស្តង់ដារ (Empirical Heuristics & Benchmark Results)

### តម្លៃស្ថិតិកំណត់ដើមសម្រាប់ ImageNet Preprocessing
រូបភាពទាំងអស់ត្រូវតែធ្វើ Normalization មុនពេលបញ្ចូលទៅក្នុងម៉ូដែល ដោយប្រើប្រាស់តម្លៃមធ្យម និងគម្លាតគំរូស្តង់ដារនៃ ImageNet៖
$$
\mu_{\text{ImageNet}} = [0.485, 0.456, 0.406], \quad \sigma_{\text{ImageNet}} = [0.229, 0.224, 0.225]
$$
$$
x_{\text{normalized}} = \frac{x_{\text{raw}} - \mu}{\sigma}
$$

---

### យុទ្ធសាស្ត្រ Test-Time Augmentation (TTA)
ដើម្បីបង្កើនភាពត្រឹមត្រូវខ្ពស់បំផុតក្នុងការប្រកួតប្រជែង (Kaggle / Benchmarks) គេប្រើប្រាស់ TTA៖
1. បង្កើតរូបភាពតេស្តជាច្រើនទម្រង់ (Multiple scales, crops, horizontal flips) ពីលើរូបភាពដើមតែមួយ ($M$ គំរូ)។
2. បញ្ជូនរូបភាពទាំង $M$ គំរូកាត់តាមម៉ូដែលដើម្បីទទួលបានវ៉ិចទ័រទស្សន៍ទាយ $y_1, y_2, \dots, y_M$។
3. គណនាមធ្យមភាគពិន្ទុទស្សន៍ទាយចុងក្រោយ: $\bar{y} = \frac{1}{M} \sum_{m=1}^M y_m$។ វិធីសាស្ត្រនេះជួយកាត់បន្ថយអត្រាកំហុសបានពី **១% ទៅ ២%** បន្ថែមទៀត។

---

### តារាងសម្រេចចិត្តលើការផ្ទេរការរៀនសូត្រ (Transfer Learning Decision Matrix)

| ទំហំសំណុំទិន្នន័យថ្មី | កម្រិតស្រដៀងនឹង ImageNet | យុទ្ធសាស្ត្រផ្ទេរការរៀនសូត្រ (Transfer Strategy) | របៀបអនុវត្តជាក់ស្តែង |
| :--- | :--- | :--- | :--- |
| **តូចបំផុត (Very Small)** | ខ្ពស់ (រូបភាពវត្ថុទូទៅ) | **Linear Probe (បង្កក Backbone)** | បង្កកទម្ងន់ទាំងអស់នៃ CNN រួចហ្វឹកហាត់តែស្រទាប់ Linear Head ចុងក្រោយប៉ុណ្ណោះ។ |
| **មធ្យម - ធំ (Medium / Large)** | ខ្ពស់ (រូបភាពវត្ថុទូទៅ) | **Full Fine-Tuning** | ប្រើប្រាស់ទម្ងន់ Pre-trained រួចហ្វឹកហាត់បណ្តាញទាំងមូលឡើងវិញដោយប្រើ Learning Rate តូច ($0.1 \times \text{lr}$ ដំបូង)។ |
| **តូចបំផុត (Very Small)** | ទាប (រូបភាពវេជ្ជសាស្ត្រ / តារាសាស្ត្រ) | **Mid-Layer Linear Probe** | ហ្វឹកហាត់ Linear Classifier លើស្រទាប់ពាក់កណ្តាល (ដែលផ្ទុក Edge/Texture ទូទៅ) ជំនួសឱ្យស្រទាប់ចុងក្រោយ។ |
| **ធំ (Large)** | ទាប (រូបភាពវេជ្ជសាស្ត្រ / តារាសាស្ត្រ) | **Partial Fine-Tuning / From Scratch** | បង្កកស្រទាប់ដំបូងៗ រួច Fine-tune ស្រទាប់ជ្រៅ ឬហ្វឹកហាត់ពីកម្រិតសូន្យតែម្តងបើធនធាន GPU គ្រប់គ្រាន់។ |

---

## ៧. ឧបសគ្គដែលត្រូវប្រុងប្រយ័ត្ន និងគន្លឹះដោះស្រាយកំហុស (Pitfalls & Debugging Tips)

* **កំហុសខុស Scale ក្នុង Inverted Dropout**:  
  ប្រសិនបើអ្នកសរសេរកូដ Dropout ដោយមិនបានចែកនឹង $(1 - p)$ នៅពេល Training ទេ នោះនៅពេល Testing តម្លៃ Magnitude នៃធាតុចូលទៅកាន់ស្រទាប់បន្ទាប់នឹងកើនឡើងទ្វេដង បណ្តាលឱ្យការព្យាករណ៍ខូចខាតទាំងស្រុង។
* **បញ្ហា Batch Size តូចពេកក្នុង Batch Normalization ($N < 8$)**:  
  នៅពេល Batch Size តូចពេក (ឧ. $N = 2$ ឬ $4$) ស្ថិតិមធ្យមភាគ និងវ៉ារ្យ៉ង់ប្រែប្រួលខ្លាំងពេក (High Variance Noise) ដែលធ្វើឱ្យការហ្វឹកហាត់មិនមានស្ថិរភាព។ ក្នុងករណី Batch Size តូច ចូរប្រើ **Group Normalization** ឬ **Layer Normalization** ជំនួសវិញ។
* **កំហុសខុសវិមាត្រក្នុង ResNet Shortcut**:  
  នៅពេលស្រទាប់ Residual Conv ប្រើប្រាស់ `stride=2` ទំហំ $H \times W$ នឹងត្រូវកាត់បន្ថយពាក់កណ្តាល។ ប្រសិនបើ Shortcut មិនបានប្រើ $1 \times 1$ Conv ជាមួយ `stride=2` ដើម្បីកាត់បន្ថយទំហំឱ្យត្រូវគ្នាទេ នោះប្រមាណវិធីបូក `out += identity` នឹង Error រលត់កូដភ្លាមៗ។
* **ការភ្លេចដូរ Mode ទៅ Eval (`model.eval()`) ពេលតេស្ត**:  
  ទាំង Dropout និង Batch Normalization មានឥរិយាបថខុសគ្នាស្រឡះរវាង Training និង Testing។ ប្រសិនបើអ្នកភ្លេចហៅ `model.eval()` មុនពេលធ្វើតេស្ត Batch Norm នឹងបន្តគណនាស្ថិតិតាម batch តេស្ត ឯ Dropout នឹងបន្តទម្លាក់ណឺរ៉ូនចោល ដែលធ្វើឱ្យលទ្ធផលតេស្តធ្លាក់ចុះយ៉ាងខ្លាំង។

---

## ៨. សំណួរពិចារណាកម្រិតក្រោយឧត្តមសិក្សា (Graduate-Level Reflection Questions)

១. **ការបំបែកភាពស៊ីមេទ្រី និងការកំណត់តម្លៃទម្ងន់ដើម (Symmetry Breaking in Initialization)**:  
   ហេតុអ្វីបានជាការកំណត់ទម្ងន់ដើមស្មើគ្នាទាំងអស់ (All-zeros ឬ All-ones) ធ្វើឱ្យបណ្តាញញាណជ្រៅមិនអាចរៀនសូម្បីតែលក្ខណៈពិសេសសាមញ្ញមួយ? ចូរពន្យល់ពីទំនាក់ទំនងគណិតវិទ្យារវាងការកំណត់ទម្ងន់ចៃដន្យ (Random Initialization), ការបំបែកភាពស៊ីមេទ្រី (Symmetry Breaking), និងការបែកខ្ញែកនៃជម្រាលក្នុងក្បួន SGD។

២. **សមត្ថភាពតំណាងនៃស្ថាបត្យកម្ម ResNet (Representational Capacity of ResNet)**:  
   ប្លុក ResNet គណនា $H(x) = F(x) + x$។ ចូរធ្វើសម្រាយបញ្ជាក់តាមគណិតវិទ្យាថា ស្ថាបត្យកម្ម ResNet ដែលមាន ៥៦ ស្រទាប់ តាមទ្រឹស្តីអាចតំណាងឱ្យគ្រប់អនុគមន៍ទាំងអស់ដែល ResNet ២០ ស្រទាប់អាចធ្វើបាន (Subsumption Property)។ ហេតុអ្វីបានជាស្ថាបត្យកម្ម Plain Networks ៥៦ ស្រទាប់គ្មានសមត្ថភាពនេះ ហើយធ្លាក់ចូលទៅក្នុងអន្ទាក់ Optimization Degradation?

៣. **Covariate Shift ក្នុង Layer Norm ធៀបនឹង Batch Norm**:  
   ចំពោះទិន្នន័យដែលមានប្រវែងប្រែប្រួលតាមពេលវេលា (Sequence models / NLP / Transformers) ហេតុអ្វីបានជា Batch Normalization មិនដំណើរការល្អ និងមានអស្ថិរភាពខ្លាំង? តើ Layer Normalization ដោះស្រាយបញ្ហាប្រវែងមិនស្មើគ្នានេះយ៉ាងដូចម្តេច ហើយហេតុអ្វីបានជាវាទទួលបានជោគជ័យយ៉ាងសម្បើមនៅក្នុងស្ថាបត្យកម្ម Modern Transformers (GPT, BERT, ViT)?
