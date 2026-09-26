# កំណត់ចំណាំមេរៀន: មេរៀនទី ៥: បណ្តាញញាណកុងវូលុយស្យុង (Convolutional Neural Networks - CNNs)

**វគ្គសិក្សា**: Stanford CS231n: Deep Learning for Computer Vision (ការរៀនស៊ីជម្រៅសម្រាប់ការមើលឃើញដោយកុំព្យូទ័រ — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Prof. Fei-Fei Li (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Ranjay Krishna (សាស្ត្រាចារ្យជំនួយផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs231n.stanford.edu](https://cs231n.stanford.edu) (ស្លាយមេរៀន ឯកសារសិក្សា និងកិច្ចការអនុវត្ត)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Quick-Recall Summary)

* **ទ្រឹស្តីបទរក្សាទ្រង់ទ្រាយលំហ (The Spatial Preservation Axiom)**: ផ្ទុយពី Fully Connected (FC) Layers ដែលពង្រាយទិន្នន័យឱ្យរាបស្មើ (Flatten) ទៅជាវ៉ិចទ័រ 1D ហើយបំផ្លាញទំនាក់ទំនងភូមិសាស្ត្រ 2D នៃរូបភាពចោល ស្រទាប់ 2D Convolution រក្សាទម្រង់វិមាត្រលំហដើម ($H \times W$) ដោយរំកិលផ្ទាំងតម្រង (Filter / Kernel) រាវកាត់តាមផ្ទៃរូបភាពដើម្បីចាប់យកលក្ខណៈពិសេសក្នុងតំបន់ (Local Patterns)។
* **ការចែករំលែកទម្ងន់ និងការសន្សំទំហំប៉ារ៉ាម៉ែត្រ (Weight Sharing & Parameter Efficiency)**: ដោយសារលក្ខណៈពិសេសមួយ (ដូចជា គែមបន្ទាត់ ឬជ្រុង) មានសារៈសំខាន់នៅគ្រប់ទីតាំងទាំងអស់លើរូបភាព តម្រងកុងវូលុយស្យុងប្រើប្រាស់ទម្ងន់ដដែលៗនៅគ្រប់ទីតាំង (Weight Sharing)។ យន្តការនេះជួយកាត់បន្ថយចំនួនប៉ារ៉ាម៉ែត្ររាប់ម៉ឺនដងធៀបនឹង Fully Connected Layer ខណៈពេលដែលរក្សាបាននូវសមត្ថភាពចាប់សញ្ញាដ៏មានឥទ្ធិពល។
* **នព្វន្ធវិមាត្រលំហ (Spatial Sizing Arithmetic)**: វិមាត្រលទ្ធផលនៃស្រទាប់ Convolution អាស្រ័យលើទំហំរូបភាព $W$, ទំហំ Kernel $K$, កម្រាស់រឹម Padding $P$, និងជំហានរំកិល Stride $S$ តាមរូបមន្ត $W' = \lfloor \frac{W - K + 2P}{S} \rfloor + 1$។
* **ការពង្រីកដែនមើលឃើញ (Receptive Field Dilation)**: ដែនមើលឃើញ (Receptive Field) នៃណឺរ៉ូនមួយក្នុងស្រទាប់ជ្រៅ កើនឡើងជាលំដាប់លីនេអ៊ែរតាមជម្រៅស្រទាប់។ ស្រទាប់តូចៗ $3 \times 3$ ត្រួតគ្នា ៣ ស្រទាប់ ផ្តល់នូវ Receptive Field ស្មើនឹងស្រទាប់ $7 \times 7$ មួយ ប៉ុន្តែសន្សំទម្ងន់បានជាង ៤៥% និងបង្កើនភាពមិនលីនេអ៊ែរ (Non-linearity) ដល់ទៅ ៣ ដង។
* **ភាពស៊ីមេទ្រីប្តូរទីតាំង (Translation Equivariance)**: រចនាសម្ព័ន្ធគណិតវិទ្យានៃ Convolution ធានានូវលក្ខណៈ Equivariance ពោលគឺ ប្រសិនបើរូបភាពធាតុចូលរំកិលទៅខាងស្តាំ $t$ ភីកសែល នោះផែនទីលក្ខណៈពិសេស (Feature Map) លទ្ធផលក៏រំកិលទៅខាងស្តាំ $t$ ភីកសែលដូចគ្នា ($\text{Conv}(g_t(X)) = g_t(\text{Conv}(X))$)។
* **របកគំហើញប្រវត្តិសាស្ត្រនៃ ImageNet & AlexNet (2012)**: ស្ថាបត្យកម្ម CNN (AlexNet) ដែលហ្វឹកហាត់លើ GPU បានបំបែកកំណត់ត្រាប្រកួតប្រជែង ImageNet ក្នុងឆ្នាំ ២០១២ ដោយកាត់បន្ថយកម្រិតកំហុស Top-5 Error ពី ៣០% មកត្រឹម ១៥.៣% ហើយបានបើកទំព័រប្រវត្តិសាស្ត្រថ្មីនៃបដិវត្តន៍ Deep Learning ពិភពលោក។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

### ក. ប្រមាណវិធីកុងវូលុយស្យុង 2D (2D Convolution Primitive)
ស្រទាប់ Convolution ទទួលយក Tensor ធាតុចូល $X$ និងបណ្តុំតម្រង $W$ (Filter Bank) ដើម្បីបង្កើតជា Output Feature Map $Y$៖
* **Input Tensor $X$**: មានទំហំ $(N, C_{\text{in}}, H, W)$ ដែល $N$ ជាទំហំ Batch, $C_{\text{in}}$ ជាចំនួន Channels (ឧ. ៣ សម្រាប់រូបភាព RGB), $H$ ជាកម្ពស់, និង $W$ ជាទទឹង។
* **Filter Weights $W$**: មានទំហំ $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$ ដែល $C_{\text{out}}$ គឺជាចំនួនតម្រងដែលត្រូវរៀន (Output Channels), $K_h \times K_w$ ជាទំហំផ្ទៃតម្រង (ជាទូទៅ $3 \times 3$ ឬ $5 \times 5$)។
* **Bias Vector $b$**: មានទំហំ $(C_{\text{out}},)$ ដោយតម្រងនីមួយៗរៀនតម្លៃ Scalar Bias មួយ។
* **Output Tensor $Y$**: មានទំហំ $(N, C_{\text{out}}, H', W')$។

### ខ. រូបមន្តគណនាទំហំលំហ (Spatial Dimension Formulation)
សម្រាប់រូបភាពទំហំ $W \times H$ តម្រងទំហំ $K_w \times K_h$ រឹម Padding $P$ និងជំហាន Stride $S$៖
$$
W' = \left\lfloor \frac{W - K_w + 2P}{S} \right\rfloor + 1
$$
$$
H' = \left\lfloor \frac{H - K_h + 2P}{S} \right\rfloor + 1
$$

* **Same Padding (ការរក្សាទំហំលំហឱ្យនៅដដែល)**: ដើម្បីធានាថា $W' = W$ នៅពេល $S = 1$ យើងត្រូវជ្រើសរើសទំហំ Kernel ជាចំនួនសេស ($K = 3, 5, 7$) ហើយកំណត់រឹម Padding តាមរូបមន្ត៖
$$
P = \frac{K - 1}{2}
$$
*(ឧទាហរណ៍៖ បើ $K = 3 \implies P = 1$; បើ $K = 5 \implies P = 2$)*។

### គ. ស្រទាប់បង្រួមទំហំលំហ (Pooling Layers)
Pooling ធ្វើការបង្រួមទំហំ $H$ និង $W$ ដោយមិនប៉ះពាល់ដល់ចំនួន Channels $C$ ឡើយ៖
* **Max Pooling**: ជ្រើសរើសតម្លៃអតិបរមាក្នុងផ្ទាំងនីមួយៗ (ជាទូទៅ $2 \times 2$ ជាមួយ Stride $S = 2$) ដែលជួយកាត់បន្ថយទំហំលំហពាក់កណ្តាល ($H/2, W/2$) និងផ្តល់នូវភាពមិនប្រែប្រួលបន្តិចបន្តួចចំពោះការរំកិលរូបភាព (Local Translation Invariance)។
* **Average Pooling**: គណនាមធ្យមភាគនៃតម្លៃទាំងអស់ក្នុងផ្ទាំងនីមួយៗ (ប្រើជាទូទៅជា Global Average Pooling នៅស្រទាប់ចុងក្រោយនៃ ResNet)។

### ឃ. សម្រាយបញ្ជាក់ភាពស៊ីមេទ្រីប្តូរទីតាំង (Translation Equivariance Proof)
ប្រមាណវិធី $f$ ត្រូវបានហៅថា Equivariant ធៀបនឹងការប្តូរទីតាំង $g_t$ ប្រសិនបើ $f(g_t(X)) = g_t(f(X))$។  
ឧបមាថាការប្តូរទីតាំងរូបភាពត្រូវបានកំណត់ដោយ $g_t(X)[n, c, i, j] = X[n, c, i - t_y, j - t_x]$។  
នៅពេលយើងអនុវត្ត Convolution លើរូបភាពដែលបានរំកិល $g_t(X)$៖
$$
\begin{aligned}
f(g_t(X))[n, c, i, j] &= \sum_{ch, ki, kj} g_t(X)[n, ch, i + ki, j + kj] \cdot W[c, ch, ki, kj] \\
&= \sum_{ch, ki, kj} X[n, ch, (i - t_y) + ki, (j - t_x) + kj] \cdot W[c, ch, ki, kj] \\
&= f(X)[n, c, i - t_y, j - t_x] = g_t(f(X))[n, c, i, j]
\end{aligned}
$$
សមភាពនេះបញ្ជាក់ថា ការរំកិលរូបភាពមុនធ្វើ Convolution ផ្តល់លទ្ធផលដូចគ្នាបេះបិទនឹងការធ្វើ Convolution រួចទើបរំកិល Feature Map តាមក្រោយ!

---

## ៣. មូលដ្ឋានគ្រឹះគណិតវិទ្យា & រូបមន្ត (Mathematical Foundations & Formulations)

### ក. សមីការពេញលេញនៃ Batched 4D Tensor Convolution
សម្រាប់គំរូទិន្នន័យ $n \in [1, N]$, Output Channel $c \in [1, C_{\text{out}}]$, និងកូអរដោនេលំហ $(i, j)$៖
$$
Y[n, c, i, j] = b[c] + \sum_{ch=1}^{C_{\text{in}}} \sum_{ki=1}^{K_h} \sum_{kj=1}^{K_w} X[n, ch, i \cdot S + ki, j \cdot S + kj] \cdot W[c, ch, ki, kj]
$$

---

### ខ. ការគណនាជម្រាលនៃកុងវូលុយស្យុងក្នុង Backpropagation
នៅពេលទទួល Upstream Gradient $\frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times C_{\text{out}} \times H' \times W'}$ ពីស្រទាប់បន្ទាប់ យើងគណនាជម្រាលធៀបនឹងទម្ងន់ តម្លៃធាតុចូល និង Bias (ឧបមាថា $S=1, P=0$)៖

1. **ជម្រាលធៀបនឹងទម្ងន់តម្រង $W$ (Filter Gradient)**:
   $$
   \frac{\partial L}{\partial W[c, ch, ki, kj]} = \sum_{n=1}^{N} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial L}{\partial Y[n, c, i, j]} \cdot X[n, ch, i + ki, j + kj]
   $$
   *(នេះជា Cross-correlation រវាង upstream gradient និងផ្ទាំង patch នៃធាតុចូល)*។

2. **ជម្រាលធៀបនឹងធាតុចូល $X$ (Input Activation Gradient)**:
   $$
   \frac{\partial L}{\partial X[n, ch, r, s]} = \sum_{c=1}^{C_{\text{out}}} \sum_{ki=1}^{K_h} \sum_{kj=1}^{K_w} \frac{\partial L}{\partial Y[n, c, r - ki, s - kj]} \cdot W[c, ch, ki, kj]
   $$
   *(តាមលក្ខណៈគណិតវិទ្យា នេះជា Transposed Convolution នៃ upstream gradient ជាមួយទម្ងន់តម្រងដែលបានត្រឡប់ ១៨០ ដឺក្រេ)*។

3. **ជម្រាលធៀបនឹង Bias $b$**:
   $$
   \frac{\partial L}{\partial b[c]} = \sum_{n=1}^{N} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial L}{\partial Y[n, c, i, j]}
   $$

---

### គ. ជម្រាលនៃស្រទាប់ Max Pooling (Argmax Routing Backpropagation)
ដោយសារ Max Pooling គ្រាន់តែជាការជ្រើសរើសតម្លៃអតិបរមា ជម្រាល upstream ត្រូវបានបញ្ជូនបន្តទៅតែទីតាំងណាដែលជាអ្នកឈ្នះ (Argmax index) ក្នុង Forward Pass ប៉ុណ្ណោះ រីឯទីតាំងផ្សេងទៀតទទួលបានជម្រាលសូន្យ៖
$$
\frac{\partial L}{\partial X[n, c, r, s]} = \begin{cases} \frac{\partial L}{\partial Y[n, c, i, j]} & \text{បើ } (r, s) = \operatorname{argmax}_{(u, v) \in \Omega_{i,j}} X[n, c, u, v] \\ 0 & \text{ផ្សេងពីនេះ} \end{cases}
$$

---

## ៤. ស្ថាបត្យកម្ម និងក្បួនដោះស្រាយ (Architecture & Algorithm Walkthrough)

### គំរូរចនាសម្ព័ន្ធលំហូរទិន្នន័យនៃ Classical CNN

```text
រូបភាពដើម (Raw Input): 3 x 32 x 32
   │
   ▼   [Conv2d: 16 filters, 3x3, stride 1, padding 1] ───► រក្សាទំហំលំហ
Activation Map: 16 x 32 x 32
   │
   ▼   [ReLU: Pointwise Non-linearity]
Activated Map: 16 x 32 x 32
   │
   ▼   [MaxPool2d: 2x2, stride 2] ──────────────────────► បង្រួមទំហំលំហពាក់កណ្តាល
Downsampled Map: 16 x 16 x 16
   │
   ▼   [Conv2d: 32 filters, 3x3, stride 1, padding 1]
Activation Map: 32 x 16 x 16
   │
   ▼   [ReLU + MaxPool2d: 2x2, stride 2]
Downsampled Map: 32 x 8 x 8
   │
   ▼   [Flatten: ពង្រាយទិន្នន័យលំហទៅជា 1D Vector]
Vector: 32 * 8 * 8 = 2,048 elements
   │
   ▼   [Linear FC Head: 2,048 -> 10 Classes]
Logits Vector: 10
```

---

### កូដគំរូស្ថាបត្យកម្មគ្រឹះ (Custom 2D ConvNet ក្នុង PyTorch)

កូដ PyTorch ខាងក្រោមបង្ហាញពីការកសាងស្ថាបត្យកម្ម CNN ពេញលេញមួយ រួមទាំងការគណនាវិមាត្រ Spatial Dimensions ដោយផ្ទាល់ និងការផ្ទៀងផ្ទាត់ទំហំ Tensors គ្រប់ដំណាក់កាល៖

```python
import torch
import torch.nn as nn

class CustomConvNet(nn.Module):
    """
    ស្ថាបត្យកម្ម CNN បែបបទស្តង់ដារដែលបង្ហាញពីការផ្លាស់ប្តូរវិមាត្រ:
    (Conv -> ReLU -> MaxPool) x 2 -> Flatten -> FC
    """
    def __init__(self, in_channels=3, num_classes=10):
        super(CustomConvNet, self).__init__()
        
        # ដំណាក់កាលទី ១: Input 3x32x32 -> Conv -> 16x32x32 -> Pool -> 16x16x16
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1, # P = (K-1)/2 = (3-1)/2 = 1 (Same padding)
            bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ដំណាក់កាលទី ២: 16x16x16 -> Conv -> 32x16x16 -> Pool -> 32x8x8
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ដំណាក់កាលទី ៣: Fully Connected Head
        # វិមាត្របន្ទាប់ពី Pooling ពីរដងគឺ: 32 channels * 8 * 8
        self.flatten_dim = 32 * 8 * 8 # = 2048
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):
        # x shape: [N, 3, 32, 32]
        out = self.conv1(x)        # -> [N, 16, 32, 32]
        out = self.relu1(out)      # -> [N, 16, 32, 32]
        out = self.pool1(out)      # -> [N, 16, 16, 16]
        
        out = self.conv2(out)      # -> [N, 32, 16, 16]
        out = self.relu2(out)      # -> [N, 32, 16, 16]
        out = self.pool2(out)      # -> [N, 32, 8, 8]
        
        # ពង្រាយទិន្នន័យលំហទៅជា Vector សម្រាប់ FC Layer
        out = torch.flatten(out, start_dim=1) # -> [N, 2048]
        logits = self.classifier(out)         # -> [N, 10]
        return logits

# ការសាកល្បងដំណើរការ និងផ្ទៀងផ្ទាត់ទំហំ Tensors
if __name__ == "__main__":
    model = CustomConvNet(in_channels=3, num_classes=10)
    sample_images = torch.randn(4, 3, 32, 32) # Batch 4 រូបភាព
    output_scores = model(sample_images)
    
    print(f"ទំហំ Input Tensor: {list(sample_images.shape)}")
    print(f"ទំហំ Output Logits: {list(output_scores.shape)}")
    
    # គណនាចំនួនប៉ារ៉ាម៉ែត្រសរុបដែលត្រូវរៀន
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ចំនួនប៉ារ៉ាម៉ែត្រសរុបរបស់ម៉ូដែល: {total_params:,} ប៉ារ៉ាម៉ែត្រ")
```

---

## ៥. ការយល់ដឹងស៊ីជម្រៅតាមបែបវិចារណញាណ (Visual Intuition & Interpretability)

### ពីរ៉ាមីតលក្ខណៈពិសេសតាមលំដាប់ថ្នាក់ (The Hierarchical Feature Pyramid)

តាមរយៈដំណើរការ Backpropagation តម្រងកុងវូលុយស្យុងរៀនបែងចែកជំនាញដោយស្វ័យប្រវត្តិតាមជម្រៅស្រទាប់៖
1. **ស្រទាប់ដំបូង (Low-Level Primitives)**: រៀនចាប់យកគែមបន្ទាត់តាមទិសដៅផ្សេងៗ (Gabor-like edge filters) ពណ៌ផ្ទុយគ្នា និងចំណុចពន្លឺតូចៗ។
2. **ស្រទាប់កណ្តាល (Mid-Level Motif Detectors)**: ផ្គុំបន្ទាត់និងចំណុចឱ្យទៅជាជ្រុង កោង រង្វង់ ឆ្នូត និងផ្ទៃក្រឡាធរណីមាត្រ (Textures & Corners)។
3. **ស្រទាប់ជ្រៅ (High-Level Semantic Templates)**: ដោយសារ Receptive Field រីកធំគ្របដណ្តប់លើផ្ទៃរូបភាពភាគច្រើន ណឺរ៉ូនរៀនចាប់យកវត្ថុស្មុគស្មាញពិតប្រាកដដូចជា កង់រថយន្ត ភ្នែកសត្វ ផ្ទៃមុខ ឬអក្សរ។

---

### មេកានិចនៃ Receptive Field (Receptive Field Arithmetic)

នៅពេលស្រទាប់កាន់តែជ្រៅ ផ្ទៃនៃរូបភាពដើមដែលណឺរ៉ូនមួយអាច "មើលឃើញ" កើនឡើងជាលំដាប់៖

```text
រូបភាពធាតុចូលដើម (Input Image)
   ███████  ◄── ផ្ទាំងទំហំ 7x7 ក្នុង Input
     \ /
    L1 (3x3) ◄── ផ្ទាំងទំហំ 5x5 ក្នុង Layer 1
     \ /
    L2 (3x3) ◄── ផ្ទាំងទំហំ 3x3 ក្នុង Layer 2
     \ /
    L3 (1x1) ◄── ណឺរ៉ូនតែមួយក្នុង Layer 3 ក្រឡេកឃើញផ្ទៃទំហំ 7x7 ក្នុង Input ដើម!
```

**ការប្រៀបធៀបគណិតវិទ្យា**: ស្រទាប់ $3 \times 3$ ចំនួន ៣ ជង់លើគ្នា មាន Receptive Field ស្មើនឹងស្រទាប់ $7 \times 7$ មួយ៖
* **ចំនួនទម្ងន់ក្នុង $7 \times 7$ មួយស្រទាប់**: $C \times (7 \times 7 \times C) = 49 C^2$
* **ចំនួនទម្ងន់ក្នុង $3 \times 3$ ចំនួន ៣ ស្រទាប់**: $3 \times [C \times (3 \times 3 \times C)] = 27 C^2$
* **ការសន្សំ**: សន្សំប៉ារ៉ាម៉ែត្របាន **$44.9\%$** ព្រមទាំងបន្ថែមអនុគមន៍មិនលីនេអ៊ែរ ReLU បានដល់ទៅ ៣ ដង ដែលជួយបង្កើនសមត្ថភាពតំណាងឱ្យកាន់តែសម្បូរបែប!

---

### ដ្យាក្រាមកូមុយតាត់នៃ Translation Equivariance

```text
រូបភាពដើម (X)  ──────────[ រំកិលទៅស្តាំ t ]──────────►  រូបភាពរំកិល (g_t(X))
      │                                                        │
  [ Conv f ]                                               [ Conv f ]
      │                                                        │
      ▼                                                        ▼
ផែនទី Feature (Y)  ───────[ រំកិលទៅស្តាំ t ]──────────►  ផែនទីរំកិល (g_t(Y))
```
*លទ្ធផលទទួលបានដូចគ្នាបេះបិទ មិនថាអ្នក "រំកិលរូបភាពមុន រួចធ្វើ Conv" ឬ "ធ្វើ Conv មុន រួចរំកិល Feature Map" ឡើយ!*

---

<div id="plotly-cs231n-5-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 5 visualization blueprint"></div>
<p><em>រូបភាព៖ គំនូសតាងអន្តរកម្មបង្ហាញពីទំហំគណនា (MFLOPs) ធៀបនឹងទំហំ Kernel Size និង Stride ($H=32, C_{\text{in}}=8, C_{\text{out}}=16$)។</em></p>

---

## ៦. ការពិសោធន៍ជាក់ស្តែង និងលទ្ធផលគោលស្តង់ដារ (Empirical Heuristics & Benchmark Results)

### ការប្រៀបធៀបប្រសិទ្ធភាពប៉ារ៉ាម៉ែត្រ & ការគណនា (Conv vs Fully Connected)

ឧបមាថាយើងមានរូបភាពទំហំ $3 \times 32 \times 32$ (Input $= 3,072$ ធាតុ) ហើយចង់បំប្លែងទៅជា Output Feature Map ទំហំ $10 \times 32 \times 32$ (Output $= 10,240$ ណឺរ៉ូន)៖

| លក្ខណៈវិនិច្ឆ័យ | ស្រទាប់ Fully Connected ($W x$) | ស្រទាប់ Convolution ($10$ តម្រង, $5 \times 5$, $S=1, P=2$) |
| :--- | :--- | :--- |
| **ចំនួនណឺរ៉ូន Output** | $10 \times 32 \times 32 = \mathbf{10,240}$ | $10 \times 32 \times 32 = \mathbf{10,240}$ |
| **យន្តការតភ្ជាប់** | ភ្ជាប់ទាំងអស់ (Global Unshared Weights) | ភ្ជាប់ក្នុងតំបន់ (Local Shared Kernel Matching) |
| **ប៉ារ៉ាម៉ែត្រដែលត្រូវរៀន** | $10,240 \times 3,072 = \mathbf{31,457,280}$ | $10 \times (3 \times 5 \times 5 + 1 \text{ bias}) = \mathbf{760}$ |
| **ប្រមាណវិធី MAC (FLOPs)** | $10,240 \times 3,072 \approx \mathbf{31.45 \times 10^6}$ | $10,240 \times 75 = \mathbf{768,000}$ |
| **កម្រិតប្រសិទ្ធភាព** | ស៊ីទំហំ និងកម្លាំងគណនាយ៉ាងមហិមា | **សន្សំប៉ារ៉ាម៉ែត្រ ៤១,៣៩១ ដង & សន្សំ FLOPs ៤១ ដង!** |

* **សេចក្តីសន្និដ្ឋាន**: CNN មិនត្រឹមតែគោរពតាមទ្រង់ទ្រាយលំហ 2D នៃរូបភាពប៉ុណ្ណោះទេ ប៉ុន្តែថែមទាំងសន្សំទំហំអង្គចងចាំ GPU និងល្បឿនគណនាយ៉ាងសម្បើមអស្ចារ្យ។

---

## ៧. ឧបសគ្គដែលត្រូវប្រុងប្រយ័ត្ន និងគន្លឹះដោះស្រាយកំហុស (Pitfalls & Debugging Tips)

* **អន្ទាក់ខុសវិមាត្រលំហ (The Fractional Dimension Trap)**:  
  ប្រសិនបើតម្លៃ $W - K + 2P$ ចែកមិនដាច់នឹង $S$ នោះវិមាត្រលទ្ធផលនឹងចេញជាចំនួនទសភាគ៖
  $$
  W' = \frac{10 - 3 + 2(0)}{2} + 1 = 4.5
  $$
  នៅក្នុង PyTorch កូដនឹងកាត់កន្ទុយចោលដោយប្រើកម្រាល Floor ($\lfloor 4.5 \rfloor = 4$) ប៉ុន្តែអាចបណ្តាលឱ្យបាត់បង់ទិន្នន័យនៅជាយគែមរូបភាព ឬបង្កបញ្ហាពេលភ្ជាប់ទៅកាន់ Fully Connected Layer ខាងចុង។ ត្រូវប្រាកដថាគណនាវិមាត្រឱ្យចេញជាចំនួនគត់ជានិច្ច!
* **បញ្ហារឹមសូន្យក្លែងក្លាយ (Zero-Padding Border Artifacts)**:  
  ការបន្ថែមលេខសូន្យនៅគែម ($P > 0$) អាចបណ្តាលឱ្យតម្រងកុងវូលុយស្យុងនៅតាមគែមទទួលបានតម្លៃអសកម្មសិប្បនិម្មិត។ បណ្តាញញាណជ្រៅៗអាចឆ្លៀតឱកាសរៀនទាញយកប្រយោជន៍ពីគែមសូន្យនេះដើម្បីប៉ាន់ស្មានទីតាំងដាច់ខាតនៃវត្ថុ (Spatial Coordinate Leakage)។
* **គ្រោះថ្នាក់នៃការបាត់បង់ភាពស៊ីមេទ្រី (Loss of Symmetry on Weights Initialization)**:  
  ប្រសិនបើអ្នកកំណត់តម្លៃទម្ងន់តម្រងទាំងអស់ក្នុងស្រទាប់ Conv ឱ្យស្មើគ្នា (ឧ. សូន្យទាំងអស់ ឬលេខថេរដូចគ្នា) នោះតម្រងទាំងអស់នឹងគណនា Forward និង Backward ចេញតម្លៃដូចគ្នាបេះបិទ។ ពួកវានឹងមិនអាចបំបែកជំនាញគ្នាបានឡើយ ហើយម៉ូដែលទាំងមូលនឹងស្រុតចុះមកស្មើនឹងតម្រងតែមួយ! ត្រូវប្រើ **Kaiming / He Normal Initialization** ជានិច្ច។
* **ភាពកម្រនៃជម្រាលក្នុង Max Pooling (Gradient Sparsity)**:  
  ដោយសារ Max Pooling បញ្ជូនជម្រាលទៅកាន់តែធាតុមួយគត់ក្នុងចំណោម ៤ ធាតុ (សម្រាប់ផ្ទាំង $2 \times 2$) នោះជម្រាលចំនួន ៧៥% នឹងត្រូវលុបចោលស្មើនឹងសូន្យ។ នៅក្នុងបណ្តាញញាណជ្រៅជ្រុល ការប្រើ Pooling ច្រើនពេកអាចកាត់បន្ថយលំហូរព័ត៌មានជម្រាល។ ម៉ូដែលទំនើបជាច្រើន (ដូចជា All-Convolutional Net ឬ ConvNeXt) ជំនួស Max Pooling ដោយប្រើ Strided Convolutions ($S=2$) វិញ។

---

## ៨. សំណួរពិចារណាកម្រិតក្រោយឧត្តមសិក្សា (Graduate-Level Reflection Questions)

១. **ភាពស៊ីមេទ្រីនៃការជង់តម្រង $3 \times 3$ ចំនួន ៣ ធៀបនឹង $7 \times 7$ មួយ**:  
   ចូរធ្វើសម្រាយបញ្ជាក់តាមគណិតវិទ្យាថា ស្រទាប់ Convolution $3 \times 3$ ចំនួន ៣ ជង់លើគ្នាដោយប្រើ $S=1$ និង $P=1$ រក្សាទំហំលំហ $W \times H$ នៃរូបភាពធាតុចូលឱ្យនៅថេរដដែល។ ប្រសិនបើចំនួន Channels គឺ $C$ ថេរគ្រប់ស្រទាប់ទាំងអស់ ចូរទាញរកផលធៀបនៃចំនួនប៉ារ៉ាម៉ែត្រ (Parameter Ratio) រវាងការជង់តម្រង $3 \times 3$ ទាំងបីនេះ ធៀបនឹងស្រទាប់ $7 \times 7$ តែមួយ។ ហេតុអ្វីបានជាស្ថាបត្យកម្ម VGGNet សម្រេចចិត្តលុបបំបាត់ចោលនូវតម្រងធំៗ $5 \times 5$ និង $7 \times 7$ ទាំងស្រុង?

២. **ភាពខុសគ្នារវាង Equivariance និង Invariance ក្នុងការចាត់ថ្នាក់រូបភាព**:  
   យើងបានបញ្ជាក់តាមបែបវិភាគរួចហើយថា ស្រទាប់ 2D Convolution មានលក្ខណៈ **Translation Equivariant** (បើឆ្មារំកិលលើរូបភាព នោះ Activation Map របស់ឆ្មាក៏រំកិលតាម)។ ប៉ុន្តែនៅក្នុងកិច្ចការ Image Classification យើងចង់បានលទ្ធផលចុងក្រោយដែលមានលក្ខណៈ **Translation Invariant** ពោលគឺ ពិន្ទុទស្សន៍ទាយនៅតែជា Class "ឆ្មា" ដដែល ទោះបីជាឆ្មាដើរទៅដល់ជ្រុងណានៃរូបភាពក៏ដោយ។ តើយន្តការស្ថាបត្យកម្មណាខ្លះនៅចុងបញ្ចប់នៃ CNN (ឧ. Max Pooling, Global Average Pooling, ឬ Fully Connected Layers) ដែលបំបែកភាព Equivariance ហើយបំលែងវាទៅជា Invariance?

៣. **អត្ថប្រយោជន៍ និងគុណវិបត្តិនៃ $1 \times 1$ Convolutions (Network-in-Network)**:  
   តម្រង $1 \times 1$ Convolution គ្មានទំហំលំហ ($K_h = 1, K_w = 1$) ឡើយ។ តើវាមានតួនាទីអ្វីខ្លះក្នុងបណ្តាញញាណជ្រៅ? ចូរពន្យល់ពីរបៀបដែល $1 \times 1$ Convolution អនុវត្តការបំប្លែងលីនេអ៊ែរឆ្លងកាត់ Channels (Cross-channel Linear Projection), របៀបដែលវាជួយកាត់បន្ថយវិមាត្រ (Bottleneck Dimensionality Reduction ក្នុង Inception និង ResNet), និងផលប៉ះពាល់របស់វាទៅលើទំហំគណនា FLOPs ទាំងមូល។
