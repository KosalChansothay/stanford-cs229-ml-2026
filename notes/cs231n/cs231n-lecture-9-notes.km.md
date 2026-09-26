# កំណត់ចំណាំមេរៀន: មេរៀនទី ៩: ការបែងចែករូបភាព ការចាប់វត្ថុ និងការស្វែងយល់ពីបណ្តាញញាណ (Segmentation, Detection & Visualizing)

**វគ្គសិក្សា**: Stanford CS231n: Deep Learning for Computer Vision (ការរៀនស៊ីជម្រៅសម្រាប់ការមើលឃើញដោយកុំព្យូទ័រ — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Prof. Fei-Fei Li (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Ranjay Krishna (សាស្ត្រាចារ្យជំនួយផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs231n.stanford.edu](https://cs231n.stanford.edu) (ស្លាយមេរៀន ឯកសារសិក្សា និងកិច្ចការអនុវត្ត)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Quick-Recall Summary)

* **ការបំប្លែងទម្រង់នៃការបែងចែកអត្ថន័យ (Semantic Segmentation Paradigms)**: ពីមុន ការបែងចែកអត្ថន័យពឹងផ្អែកលើការកាត់រូបភាពជាផ្ទាំងតូចៗ (Patch-based Classification) ដែលយឺតខ្លាំង។ វិធីសាស្ត្រទំនើបប្រើប្រាស់ **Fully Convolutional Networks (FCNs)** ដែលដំណើរការរូបភាពទាំងមូលក្នុងពេលតែមួយ និងស្ថាបត្យកម្ម **U-Net / SegNet** ដែលរក្សាភាពច្បាស់នៃជាយព្រំដែនវត្ថុ (Sharp Boundaries) តាមរយៈ Skip Connections និង Max Unpooling (ប្រើប្រាស់កូអរដោនេដែលរក្សាទុកពី Max Pooling)។
* **មេកានិចនៃការពង្រីកទំហំរូបភាព (Upsampling Mechanics)**: 
  * **Non-parametric / Cached**: Nearest Neighbor (ចម្លងតម្លៃដូចគ្នា), Bed of Nails (ដាក់នៅជ្រុងមួយ ក្រៅពីនោះសូន្យ), និង **Max Unpooling** (ចងចាំទីតាំង Argmax ពីដំណាក់កាល Pooling ដើម ដើម្បីស្ដារកូអរដោនេពិតប្រាកដឡើងវិញដោយគ្មានប៉ារ៉ាម៉ែត្របន្ថែម)។
  * **Parametric**: **Transposed Convolution** (ជួនកាលហៅថា Fractionally Strided Convolution) ដែលរៀនទម្ងន់តម្រងដើម្បីពង្រីកទំហំរូបភាពឡើងវិញ។
* **ការចាប់ទីតាំង និងសម្គាល់វត្ថុ (Object Detection Scaling)**: ពីការទស្សន៍ទាយប្រអប់វត្ថុទោល (Single-object Localization តាម Multitask Loss) ឈានទៅរកការចាប់វត្ថុច្រើនក្នុងពេលតែមួយ (Multiple Objects) តាមរយៈ Region Proposals (R-CNN), Single-stage Dense Regression (YOLO កាត់រូបភាពជាក្រឡាចត្រង្គ $S \times S$), និងស្ថាបត្យកម្ម **DETR (DEtection TRansformer)** ដែលចាត់ទុកការចាប់វត្ថុជា Set Prediction Problem ដោយប្រើក្បួន Bipartite Hungarian Matching ដោយមិនបាច់ពឹងផ្អែកលើ Non-Maximum Suppression (NMS) ឡើយ។
* **មូលដ្ឋានគ្រឹះនៃការបកស្រាយម៉ូដែល (Visual Interpretability)**: **Saliency Maps** គណនាដេរីវេនៃពិន្ទុ Class ធៀបនឹងភីកសែលធាតុចូល ដើម្បីបង្ហាញភីកសែលដែលងាយប្រតិកម្ម រីឯ **Grad-CAM (Gradient-weighted Class Activation Mapping)** ប្រើប្រាស់មធ្យមភាគជម្រាលជាទម្ងន់ $\alpha_c^k$ ទៅលើ Feature Maps នៃស្រទាប់ Convolution ចុងក្រោយ ដើម្បីបង្កើតផែនទីកម្តៅ (Heatmap) ច្បាស់លាស់សម្រាប់ Class នីមួយៗ។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

### ក. ការបែងចែកអត្ថន័យតាមភីកសែល (Semantic Segmentation)
Semantic Segmentation ធ្វើចំណាត់ថ្នាក់គ្រប់ភីកសែលទាំងអស់លើរូបភាព ដោយមិនបែងចែកដាច់ពីគ្នារវាង Instance ផ្សេងគ្នានៃប្រភេទតែមួយឡើយ (ឧ. ឆ្មា ៣ ក្បាលឈរក្បែរគ្នា ត្រូវបានកំណត់ពណ៌ដូចគ្នាទាំងអស់ថាជា Class "ឆ្មា")។ ផ្ទុយពីនេះ **Instance Segmentation** ញែកឆ្មានីមួយៗដាច់ដោយឡែកពីគ្នា។

### ខ. ការចាប់ទីតាំងវត្ថុពហុគោលបំណង (Multitask Localization & Detection)
ម៉ូដែលទទួលរូបភាពមួយ ហើយត្រូវធ្វើកិច្ចការពីរក្នុងពេលដំណាលគ្នា៖
1. **Classification**: ទស្សន៍ទាយប្រូបាប៊ីលីតេនៃប្រភេទវត្ថុ ($p$)។
2. **Bounding Box Regression**: ទស្សន៍ទាយកូអរដោនេនៃប្រអប់ព្រំដែនចំនួន ៤៖ ចំណុចកណ្តាល $(x, y)$, ទទឹង $w$, និងកម្ពស់ $h$។

---

### គ. ស្ថាបត្យកម្ម DETR (DEtection TRansformer)
ស្ថាបត្យកម្ម DETR (Carion et al., 2020) លុបបំបាត់ចោលទាំងស្រុងនូវ Anchor Boxes និង Non-Maximum Suppression (NMS)៖
* ប្រើ CNN Backbone ដើម្បីស្រង់លក្ខណៈពិសេសពីរូបភាព។
* ប្រើ Transformer Encoder-Decoder ដែលទទួលយកសំណុំថេរនៃ **Object Queries** (ឧ. $N = 100$ queries)។
* Queries ទាំង ១០០ ពិភាក្សាគ្នាទៅវិញទៅមកតាមរយៈ Self-Attention ដើម្បីធានាថាមិនមាន Query ពីរណាចាប់វត្ថុត្រួតគ្នាឡើយ។
* ប្រើប្រាស់ក្បួនគណិតវិទ្យា **Hungarian Algorithm** ដើម្បីផ្គូផ្គង ១ ធៀបនឹង ១ (Bipartite Matching) រវាងការទស្សន៍ទាយ និងវត្ថុពិតប្រាកដ។

---

## ៣. មូលដ្ឋានគ្រឹះគណិតវិទ្យា & រូបមន្ត (Mathematical Foundations & Formulations)

### ក. សមីការ Pixel-Wise Cross-Entropy Loss (Semantic Segmentation)
ចំពោះរូបភាពទំហំ $H \times W$ ភីកសែលនីមួយៗត្រូវបានចាត់ទុកជាបញ្ហា Classification ឯករាជ្យមួយ៖
$$
L_{\text{seg}} = -\frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} \log\left( \frac{e^{s_{h, w, y_{h,w}}}}{\sum_{c=1}^{C} e^{s_{h, w, c}}} \right)
$$
* $s_{h,w,c}$: Logit ដែលម៉ូដែលទស្សន៍ទាយសម្រាប់ភីកសែល $(h, w)$ លើ Class $c$។
* $y_{h,w} \in \{1, \dots, C\}$: ស្លាកពិតប្រាកដ (Ground-truth class) នៅភីកសែលនោះ។

---

### ខ. សមីការ Multitask Loss សម្រាប់ Classification និង Box Regression
$$
L_{\text{multitask}} = L_{\text{cls}}(p, y) + \lambda \cdot \mathbb{I}(y \ge 1) \sum_{j \in \{x, y, w, h\}} \|t_j - t_j^*\|_2^2
$$
* $L_{\text{cls}}(p, y)$: ការខាតបង់ Softmax Cross-Entropy។
* $\mathbb{I}(y \ge 1)$: អនុគមន៍ចង្អុលបង្ហាញ (Indicator Function) ដែលដំណើរការតែលើ Foreground Objects ប៉ុណ្ណោះ (មិនគណនា Bounding Box សម្រាប់ផ្ទៃខាងក្រោយ Background $y=0$ ឡើយ)។
* $t = (t_x, t_y, t_w, t_h)$ និង $t^*$: វ៉ិចទ័រកូអរដោនេទស្សន៍ទាយ និងកូអរដោនេពិតប្រាកដ។
* $\lambda$: មេគុណតុល្យភាពរវាងភារកិច្ចទាំងពីរ។

---

### គ. នព្វន្ធនៃ Transposed Convolution
Transposed Convolution គឺជាម៉ាទ្រីសឆ្លាស់នៃប្រមាណវិធី Convolution ធម្មតា។ សម្រាប់ Input ទំហំ $I$, Stride $S$, Kernel $K$, និង Padding $P$ វិមាត្រ Output $O$ ត្រូវបានគណនាដោយ៖
$$
O = S \cdot (I - 1) + K - 2P
$$
*ក្នុងការអនុវត្តជាក់ស្តែង តម្លៃ Input នីមួយៗគុណនឹង Kernel ទំហំ $K \times K$ ហើយបូកស្រោបចូលគ្នានៅត្រង់តំបន់ត្រួតគ្នា (Overlap Regions)*។

---

### ឃ. សមីការ Bipartite Matching Loss (Hungarian Matching ក្នុង DETR)
ដើម្បីផ្គូផ្គងសំណុំការទស្សន៍ទាយ $N$ ប្រអប់ $\hat{y}$ ជាមួយវត្ថុពិត $y$ ក្បួន Hungarian Algorithm ស្វែងរកការឆ្លាស់ $\hat{\sigma} \in \mathfrak{S}_N$ ដែលកាត់បន្ថយតម្លៃខាតបង់ផ្គូផ្គងសរុប៖
$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})
$$
ដែល $\mathcal{L}_{\text{match}}$ រួមបញ្ចូលទាំងកម្រិតស្រដៀងគ្នានៃ Class និងភាពស៊ីគ្នានៃ Bounding Box (L1 loss បូក Generalized IoU loss)។

---

### ង. ផែនទី Saliency Maps តាមរយៈ Backpropagation
ដើម្បីដឹងថាតើភីកសែលណាខ្លះជំរុញឱ្យម៉ូដែលទស្សន៍ទាយពិន្ទុ $S_c(I)$ យើងគណនាដេរីវេនៃពិន្ទុធៀបនឹងភីកសែលធាតុចូល៖
$$
\text{Saliency}(I)_{x,y} = \max_{c \in \{\text{RGB}\}} \left| \frac{\partial S_c(I)}{\partial I_{x, y, c}} \right|
$$
*នេះតំណាងឱ្យការពន្លាត Taylor លំដាប់ទីមួយ ដែលបង្ហាញពីកម្រិតរំញោចនៃភីកសែលនីមួយៗចំពោះការប្រែប្រួលនៃពិន្ទុ Class*។

---

### ច. រូបមន្ត Grad-CAM ពេញលេញ
Grad-CAM បង្កើតផែនទីកម្តៅដោយប្រើប្រាស់ Feature Maps $A^k \in \mathbb{R}^{H \times W}$ នៃស្រទាប់ Convolution ចុងក្រោយ៖

1. **គណនាទម្ងន់សារៈសំខាន់នៃ Channel នីមួយៗ ($\alpha_c^k$)**:
   ធ្វើ Global Average Pooling លើជម្រាលនៃពិន្ទុ Class $S_c$ ធៀបនឹង Feature Map $A^k$៖
   $$
   \alpha_c^k = \frac{1}{Z} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial S_c}{\partial A_{i, j}^k} \quad (\text{ដែល } Z = H \times W)
   $$

2. **បូកបញ្ចូល Feature Maps ដោយប្រើទម្ងន់ រួចអនុវត្ត ReLU**:
   $$
   L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left( \sum_{k} \alpha_c^k A^k \right)
   $$
   *ការប្រើប្រាស់ $\operatorname{ReLU}$ គឺចាំបាច់បំផុត ដើម្បីរក្សាទុកតែលក្ខណៈពិសេសណាដែល **រួមចំណែកជាវិជ្ជមាន** ដល់ Class គោលដៅប៉ុណ្ណោះ និងច្រានចោលលក្ខណៈពិសេសដែលជារបស់ Class ផ្សេង*។

---

## ៤. ស្ថាបត្យកម្ម និងក្បួនដោះស្រាយ (Architecture & Algorithm Walkthrough)

### ការប្រៀបធៀបយុទ្ធសាស្ត្រ Upsampling

```text
[Nearest Neighbor]         [Bed of Nails]              [Max Unpooling]            [Transposed Conv]
   1 2  ──►  1 1 2 2          1 2  ──►  1 0 2 0            1 2  ──►  0 0 2 0          Input x Kernel,
   3 4       1 1 2 2          3 4       0 0 0 0            3 4       0 3 0 0          បូកបញ្ចូលត្រង់
             3 3 4 4                    3 0 4 0                      (ប្រើកូអរដោនេ     កន្លែងត្រួតគ្នា
                                                                      ពី MaxPool)
```

---

### កូដគំរូស្ថាបត្យកម្មគ្រឹះ (PyTorch Implementation: MaxUnpooling & Grad-CAM Hooks)

កូដខាងក្រោមបង្ហាញពីការកសាងស្ថាបត្យកម្ម Segmentation ដែលរក្សាទុក និងស្ដារកូអរដោនេលំហតាម `MaxUnpool2d` ព្រមទាំងការបង្កើត `GradCAMWrapper` ដោយប្រើ PyTorch Forward/Backward Hooks៖

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateTrackingDecoder(nn.Module):
    """
    ស្ថាបត្យកម្ម Encoder-Decoder ដែលប្រើប្រាស់ MaxUnpool2d
    ដើម្បីស្ដារកូអរដោនេដើមនៃជាយព្រំដែន (Sharp Boundaries) ឡើងវិញយ៉ាងច្បាស់។
    """
    def __init__(self, num_classes=10):
        super(CoordinateTrackingDecoder, self).__init__()
        # Encoder: Conv -> MaxPool (រក្សាទុក Indices)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Decoder: MaxUnpool (ប្រើ Indices) -> Conv
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Forward Encoder
        h1 = F.relu(self.conv1(x))
        pooled_h1, pool_indices = self.pool1(h1) # រក្សាទុកកូអរដោនេ
        
        # Forward Decoder
        unpooled = self.unpool1(pooled_h1, pool_indices) # ស្ដារកូអរដោនេ
        logits = self.conv2(unpooled)
        return logits


class GradCAMWrapper:
    """
    ម៉ូឌុលស្រង់យក Feature Maps និង Gradients ដោយស្វ័យប្រវត្តិតាមរយៈ PyTorch Hooks
    ដើម្បីគណនាផែនទីកម្តៅ Grad-CAM សម្គាល់ទីតាំងវត្ថុក្នុងរូបភាព។
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # ចុះឈ្មោះ Hooks
        self.fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, class_idx: int, scores: torch.Tensor) -> torch.Tensor:
        # Backward Pass លើ Class គោលដៅ
        self.model.zero_grad()
        target_score = scores[0, class_idx]
        target_score.backward(retain_graph=True)
        
        # គណនាទម្ងន់ alpha_c^k តាម Global Average Pooling នៃជម្រាល
        # gradients shape: [Batch, Channels, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # [1, C, 1, 1]
        
        # បូកបញ្ចូល Feature Maps ដោយប្រើទម្ងន់
        cam = torch.sum(weights * self.activations, dim=1) # [1, H, W]
        
        # អនុវត្ត ReLU ដើម្បីដកលក្ខណៈអវិជ្ជមានចេញ
        cam = F.relu(cam)
        
        # ធ្វើ Normalization មកចន្លោះ [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        return cam.squeeze()

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

# ការសាកល្បងដំណើរការ
if __name__ == "__main__":
    model = CoordinateTrackingDecoder(num_classes=5)
    sample_img = torch.randn(1, 3, 32, 32)
    
    seg_logits = model(sample_img)
    print(f"ទំហំ Input: {list(sample_img.shape)}")
    print(f"ទំហំ Output Segmentation Map: {list(seg_logits.shape)}")
```

---

## ៥. ការយល់ដឹងស៊ីជម្រៅតាមបែបវិចារណញាណ (Visual Intuition & Interpretability)

### Saliency Maps ធៀបនឹង Grad-CAM
* **Saliency Maps**: ផ្តល់នូវភីកសែលរំញោចកម្រិតល្អិត (Pixel-level) ប៉ុន្តែងាយនឹងរងផលប៉ះពាល់ដោយសំឡេងរំខាន (High-frequency Noise) ហើយជារឿយៗវាគ្រាន់តែចាប់យកជាយបន្ទាត់ (Edges) ទូទៅប៉ុណ្ណោះ ដោយមិនបញ្ជាក់ច្បាស់ពី Class ឡើយ។
* **Grad-CAM**: បង្កើតផែនទីកម្តៅជាដុំៗកម្រិត Semantic (Coarse Heatmaps) ដែលបង្ហាញយ៉ាងច្បាស់ពីទីតាំងភូមិសាស្ត្រនៃវត្ថុ (ឧ. ក្នុងរូបភាពសត្វឆ្កែនិងឆ្មាឈរក្បែរគ្នា បើជ្រើសរើស Class "ឆ្កែ" ផែនទីកម្តៅនឹងភ្លឺឡើងតែលើតួខ្លួនឆ្កែប៉ុណ្ណោះ ឯឆ្មាគ្មានពន្លឺឡើយ)។

---

### ការរក្សាជាយព្រំដែន (Max Unpooling / U-Net vs. FCN)
* **FCN ធម្មតា**: បង្រួមទំហំរូបភាពឱ្យនៅតូចខ្លាំងក្នុង Bottleneck រួចពង្រីកមកវិញដោយប្រើ Bilinear Interpolation ធ្វើឱ្យព្រំដែនវត្ថុប្រែជាព្រាលៗ (Blurred Outlines)។
* **Max Unpooling / U-Net**: តាមរយៈការបញ្ជូន Indices ពី Pooling មកវិញ ឬចម្លង High-resolution Features ពី Encoder មក Decoder ដោយផ្ទាល់ គែមនៃវត្ថុ (ដូចជា ជើងតុ ឬរោមសត្វ) រក្សាបាននូវភាពមុតស្រួច និងត្រឹមត្រូវកម្រិតភីកសែល។

---

<div id="plotly-cs231n-9-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 9 visualization blueprint"></div>
<p><em>រូបភាព៖ គំនូសតាងអន្តរកម្មបង្ហាញពីគុណភាពនៃការបង្កើតជាយព្រំដែនឡើងវិញ (Boundary F-score) ធៀបនឹងកម្រិត Upsampling Factor រវាងវិធីសាស្ត្រ Nearest Neighbor, Max Unpooling, និង Transposed Convolution។</em></p>

---

## ៦. ការពិសោធន៍ជាក់ស្តែង និងលទ្ធផលគោលស្តង់ដារ (Empirical Heuristics & Benchmark Results)

* **ការបង្រួម Loss ក្នុង Semantic Segmentation**:  
  ការបូកសរុប Pixel-wise Cross-Entropy លើរូបភាពទំហំធំ (ឧ. $1024 \times 1024 = 1,048,576$ ភីកសែល) បណ្តាលឱ្យទំហំជម្រាលកើនឡើងខ្លាំងសម្បើម។ ត្រូវប្រាកដថាអ្នកធ្វើមធ្យមភាគដោយចែកនឹង $H \cdot W$ ជានិច្ច (`reduction='mean'`)។
* **ចំនួន Object Queries ក្នុង DETR**:  
  នៅក្នុងស្ថាបត្យកម្មស្តង់ដារ DETR ចំនួន Object Queries ត្រូវបានកំណត់ថេរស្មើនឹង $N = 100$ ដែលមានន័យថាម៉ូដែលស្វែងរកវត្ថុអតិបរមាចំនួន ១០០ ក្នុងមួយរូបភាព។
* **ចំណាត់ថ្នាក់ក្នុង Vision Transformers (ViT)**:  
  ការស្រាវជ្រាវជាក់ស្តែងបង្ហាញថា ការដក `[CLS]` Token ចេញ ហើយជំនួសដោយ **Global Average Pooling (GAP)** លើគ្រប់ Patch Tokens ទាំងអស់ ផ្តល់នូវភាពត្រឹមត្រូវប្រហាក់ប្រហែលគ្នា ប៉ុន្តែជួយឱ្យក្បួន Optimization មានស្ថេរភាពជាងមុន។

---

## ៧. ឧបសគ្គដែលត្រូវប្រុងប្រយ័ត្ន និងគន្លឹះដោះស្រាយកំហុស (Pitfalls & Debugging Tips)

* **អន្ទាក់នៃទំហំជម្រាល Saliency (The Saliency Magnitude Trap)**:  
  កុំយល់ច្រឡំថា Saliency Map ខ្ពស់មានន័យថាម៉ូដែលយល់ដឹងពីវត្ថុនោះឱ្យសោះ។ ជារឿយៗ ដេរីវេ $\frac{\partial S_c}{\partial I}$ គ្រាន់តែឆ្លុះបញ្ចាំងពីការប្រែប្រួលពន្លឺនៃភីកសែលគែមប៉ុណ្ណោះ។ ចូរប្រើ **Integrated Gradients** ឬ **Grad-CAM** ជំនួសវិញ។
* **ការជ្រើសរើស Layer ខុសក្នុង Grad-CAM**:  
  Grad-CAM ដំណើរការបានល្អបំផុតលើស្រទាប់ Convolution ចុងក្រោយបង្អស់ (Penultimate Layer) ពីព្រោះស្រទាប់នេះផ្ទុកព័ត៌មាន Semantic កម្រិតខ្ពស់បំផុត និងនៅមានវិមាត្រលំហ 2D។ ប្រសិនបើអ្នកអនុវត្តលើស្រទាប់ដំបូងៗ ផែនទីកម្តៅនឹងបង្ហាញត្រឹមតែបន្ទាត់ និងចំណុចពន្លឺតូចៗដែលគ្មានប្រយោជន៍ឡើយ។
* **បញ្ហា Checkboard Artifacts ក្នុង Transposed Convolutions**:  
  នៅពេលទំហំ Kernel $K$ ចែកមិនដាច់នឹង Stride $S$ តំបន់ត្រួតគ្នាក្នុង Transposed Convolution នឹងមានការបូកសរុបមិនស្មើគ្នា បង្កើតបានជាស្នាមក្រឡាចត្រង្គអាក្រក់មើលលើរូបភាព (Checkboard Artifacts)។ **ដំណោះស្រាយ**: ប្រើ Nearest Neighbor Upsampling បូកបញ្ចូលជាមួយ 2D Convolution ធម្មតាជំនួសវិញ។

---

## ៨. សំណួរពិចារណាកម្រិតក្រោយឧត្តមសិក្សា (Graduate-Level Reflection Questions)

១. **ការប្រៀបធៀបរវាង FCN និង Coordinate-Cached Decoders (Max Unpooling)**:  
   ចូរធ្វើសម្រាយបញ្ជាក់ថា ហេតុអ្វីបានជាស្ថាបត្យកម្ម FCN ដែលពឹងផ្អែកតែលើ Transposed Convolutions ជួបការលំបាកក្នុងការកសាងឡើងវិញនូវរចនាសម្ព័ន្ធលម្អិតដូចជា ខ្សែបន្ទាត់ស្តើង ឬជ្រុងស្រួចៗ? តើ Max Unpooling អាចជួយដោះស្រាយបញ្ហានេះដោយរបៀបណា ដោយមិនចាំបាច់បន្ថែមប៉ារ៉ាម៉ែត្ររៀនថ្មីសូម្បីតែមួយ?

២. **ភាពកាត់កែងនៃ Object Queries ក្នុង DETR (Query Orthogonality & Disambiguation)**:  
   នៅក្នុង DETR Object Queries ទាំង ១០០ គឺជាវ៉ិចទ័រប៉ារ៉ាម៉ែត្រថេរដែលត្រូវបានរៀនក្នុងពេលហ្វឹកហាត់។ ហេតុអ្វីបានជា Queries ទាំងនេះមិនដួលរលំត្រួតស៊ីគ្នា (Mode Collapse) ទៅលើវត្ថុធំតែមួយក្នុងរូបភាព? តើស្រទាប់ Multi-Head Self-Attention នៅក្នុង Transformer Decoder ដើរតួនាទីយ៉ាងដូចម្តេចខ្លះក្នុងការរុញច្រាន Queries ទាំងនេះឱ្យបំបែកទិសដៅគ្នា និងស្វែងរកទីតាំងផ្សេងៗគ្នាលើរូបភាព?

៣. **តម្រូវការនៃអនុគមន៍ ReLU ក្នុងរូបមន្ត Grad-CAM**:  
   នៅក្នុងរូបមន្តចុងក្រោយនៃ Grad-CAM $L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left( \sum_k \alpha_c^k A^k \right)$ ហេតុអ្វីបានជាអ្នកស្រាវជ្រាវត្រូវអនុវត្តអនុគមន៍ $\operatorname{ReLU}$? ប្រសិនបើយើងដក $\operatorname{ReLU}$ ចេញ ហើយអនុវត្តតម្លៃដាច់ខាត (Absolute Value) ឬទុកតម្លៃអវិជ្ជមាននៅដដែល តើផែនទីកម្តៅដែលទទួលបាននឹងបង្ហាញពីអ្វីខ្លះ ហើយហេតុអ្វីបានជាវាបាត់បង់សមត្ថភាព Class-Specific Localization?
