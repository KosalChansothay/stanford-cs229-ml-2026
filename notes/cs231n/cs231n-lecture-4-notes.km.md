# កំណត់ចំណាំមេរៀន: មេរៀនទី ៤: បណ្តាញញាណសិប្បនិម្មិត និងក្បួន Backpropagation (Neural Networks & Backpropagation)

**វគ្គសិក្សា**: Stanford CS231n: Deep Learning for Computer Vision (ការរៀនស៊ីជម្រៅសម្រាប់ការមើលឃើញដោយកុំព្យូទ័រ — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Prof. Fei-Fei Li (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Ranjay Krishna (សាស្ត្រាចារ្យជំនួយផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs231n.stanford.edu](https://cs231n.stanford.edu) (ស្លាយមេរៀន ឯកសារសិក្សា និងកិច្ចការអនុវត្ត)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Quick-Recall Summary)

* **ថាមពលនៃអនុគមន៍មិនលីនេអ៊ែរ (The Power of Non-Linearity)**: បណ្តាញញាណច្រើនស្រទាប់ (Multi-Layer Perceptrons - MLPs) ដំណើរការដោយការត្រួតគ្នានូវការបំប្លែងលីនេអ៊ែរ (Linear Transformations) ជាមួយនឹងអនុគមន៍សកម្មកម្មមិនលីនេអ៊ែរតាមធាតុ (Element-wise Non-linear Activations)។ ប្រសិនបើយើងដកអនុគមន៍សកម្មកម្មចេញ នោះបណ្តាញញាណជ្រៅទោះបីជាមាន ១០០ ស្រទាប់ក្តី នឹងស្រុតចុះ (Collapse) មកត្រឹមម៉ូដែលលីនេអ៊ែរតែមួយស្រទាប់ធម្មតាដដែលតាមរយៈផលគុណម៉ាទ្រីស។
* **ទម្រង់គំរូផ្នែករងតាមលំដាប់ថ្នាក់ (Hierarchical Part Templates)**: ផ្ទុយពី Linear Classifiers ដែលអាចរៀនបានត្រឹមតែគំរូទូទៅមួយគត់សម្រាប់ថ្នាក់នីមួយៗ ស្រទាប់លាក់ (Hidden Layer) ដែលមានណឺរ៉ូនទំហំ $H$ អាចរៀនគំរូផ្នែករងរាប់សិប (Sub-templates ដូចជា កង់រថយន្ត ភ្នែក ច្រមុះ ជ្រុងដេក) ដែលអាចប្រើប្រាស់រួមគ្នារវាងប្រភេទរូបភាពផ្សេងៗគ្នា។
* **គោលការណ៍គ្រឹះនៃ Backpropagation (Local Computation)**: ក្បួន Backpropagation បំបែកសមីការគណនាការខាតបង់សរុប (Global Loss) ឱ្យទៅជាគំនូសតាងគណនា (Computational Graph) ដែលមានប្រមាណវិធីសាមញ្ញៗ។ ថ្នាំង (Node) នីមួយៗគណនាត្រឹមតែជម្រាលក្នុងតំបន់ (Local Gradient) របស់វា ហើយគុណនឹងជម្រាលខាងលើ (Upstream Gradient) ដែលហូរត្រឡប់មកពីចុងក្រោយតាមក្បួនសង្វាក់ (Chain Rule)។
* **ឥរិយាបថនៃទ្វារគណនាជម្រាលទាំង ៤ (Canonical Gradient Gates)**:
  * **ទ្វារបូក (Add Gate)**: ដើរតួជា *អ្នកចែកចាយជម្រាល (Distributor)* ដោយបញ្ជូនជម្រាល upstream ទៅគ្រប់ input ស្មើៗគ្នា។
  * **ទ្វារគុណ (Multiply Gate)**: ដើរតួជា *អ្នកប្តូរនិងពង្រីកជម្រាល (Scaled Switcher)* ដោយគុណជម្រាល upstream ជាមួយតម្លៃ input ផ្ទុយគ្នា។
  * **ទ្វារចម្លង (Copy Gate / Branching)**: ដើរតួជា *អ្នកបូកសរុបជម្រាល (Summer)* ដោយបូកជម្រាល upstream ទាំងអស់ដែលហូរមកពីសាខាផ្សេងៗ។
  * **ទ្វារអតិបរមា (Max Gate)**: ដើរតួជា *អ្នកជ្រើសរើសផ្លូវ (Router)* ដោយបញ្ជូនជម្រាល upstream ទៅតែធាតុណាដែលមានតម្លៃធំបំផុត រីឯធាតុផ្សេងទៀតទទួលបានជម្រាលសូន្យ។
* **ការគណនាជម្រាលតាមវ៉ិចទ័រដោយមិនឆ្លងកាត់ Jacobian (Bypassing Huge Jacobians)**: ក្នុងការអនុវត្តជាក់ស្តែងលើម៉ាទ្រីស យើងមិនគណនាម៉ាទ្រីសដេរីវេ Jacobian ដែលអាចស៊ីទំហំរាប់រយ Gigabytes នោះឡើយ។ ផ្ទុយទៅវិញ យើងប្រើប្រាស់ការផ្គូផ្គងវិមាត្រនៃផលគុណម៉ាទ្រីសឆ្លាស់ (Dimension-Matching Transposed Matrix Multiplication) ដើម្បីទទួលបានជម្រាលត្រឹមត្រូវ ១០០% ក្នុងល្បឿនលឿនបំផុត។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

### ក. បណ្តាញញាណច្រើនស្រទាប់ (Multi-Layer Perceptron - MLP)
បណ្តាញញាណ ២ ស្រទាប់ (ស្រទាប់លាក់ ១ និងស្រទាប់បញ្ចេញ ១) ត្រូវបានកំណត់ដោយសមីការ៖
$$
f(x, W_1, W_2) = W_2 \max(0, W_1 x)
$$
* $x \in \mathbb{R}^D$: វ៉ិចទ័រធាតុចូល (Input features ឧ. ភីកសែលរូបភាព $3072 \times 1$)។
* $W_1 \in \mathbb{R}^{H \times D}$: ម៉ាទ្រីសទម្ងន់នៃស្រទាប់ទីមួយ ដែលបំប្លែងទិន្នន័យទៅជាលំហលក្ខណៈពិសេសលាក់ (Hidden space) ទំហំ $H$។
* $\max(0, \cdot)$: អនុគមន៍សកម្មកម្ម Rectified Linear Unit (ReLU)។
* $W_2 \in \mathbb{R}^{C \times H}$: ម៉ាទ្រីសទម្ងន់នៃស្រទាប់ទីពីរ ដែលបំប្លែងលក្ខណៈពិសេសលាក់ទៅជាពិន្ទុនៃថ្នាក់ទាំង $C$។

*(ចំណាំ៖ តម្លៃ Bias $b_1 \in \mathbb{R}^H$ និង $b_2 \in \mathbb{R}^C$ ត្រូវបានបញ្ចូលជាទូទៅក្នុងទម្រង់ $W_2 \max(0, W_1 x + b_1) + b_2$)*។

### ខ. ការស្រុតចុះនៃស្រទាប់លីនេអ៊ែរដែលគ្មានសកម្មកម្ម (Collapse of Stacked Linearity)
ប្រសិនបើយើងដកអនុគមន៍សកម្មកម្មចេញពីបណ្តាញញាណ៖
$$
f(x) = W_2 (W_1 x) = (W_2 W_1) x = W_3 x
$$
ដោយសារផលគុណនៃម៉ាទ្រីសពីរ $W_2 W_1$ គឺជាម៉ាទ្រីសថ្មីមួយ $W_3 \in \mathbb{R}^{C \times D}$ នោះបណ្តាញញាណទោះបីជាមាន ១០០ ស្រទាប់ក្តី គឺគ្មានសមត្ថភាពតំណាងលើសពី Linear Classifier មួយស្រទាប់ឡើយ។ ហេតុនេះ **អនុគមន៍មិនលីនេអ៊ែរ (Non-linearity) គឺជាដង្ហើមដ៏សំខាន់បំផុតរបស់ Deep Learning**។

### គ. គំនូសតាងគណនា និងក្បួនសង្វាក់ (Computational Graph & Chain Rule)
ការគណនាស្មុគស្មាញត្រូវបានបំបែកជាបណ្តាញនៃប្រមាណវិធីសាមញ្ញៗ (Nodes)។ សម្រាប់ថ្នាំងមួយដែលមានធាតុចូល $x, y$ និងធាតុចេញ $z = f(x, y)$ ហើយភ្ជាប់ទៅកាន់ការខាតបង់ចុងក្រោយ $L$៖
* **Forward Pass**: ទទួលតម្លៃ $x, y$ គណនា $z = f(x, y)$ និងរក្សាទុក (Cache) តម្លៃចាំបាច់ក្នុងអង្គចងចាំ។
* **Backward Pass**: ទទួល upstream gradient $\frac{\partial L}{\partial z}$ គណនា local gradients $\frac{\partial z}{\partial x}$ និង $\frac{\partial z}{\partial y}$ រួចបញ្ជូន downstream gradients បន្តទៅមុខ៖
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} \quad \text{និង} \quad \frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y}
$$

---

## ៣. មូលដ្ឋានគ្រឹះគណិតវិទ្យា & រូបមន្ត (Mathematical Foundations & Formulations)

### ក. អនុគមន៍សកម្មកម្មចម្បងៗ និងដេរីវេក្នុងតំបន់ (Activation Functions & Local Derivatives)

1. **Sigmoid**:
   $$
   \sigma(x) = \frac{1}{1 + e^{-x}}
   $$
   ដេរីវេក្នុងតំបន់អាចសរសេរតាមតម្លៃ output ផ្ទាល់៖
   $$
   \frac{d\sigma(x)}{dx} = \sigma(x) \cdot (1 - \sigma(x))
   $$
   *លក្ខណៈ*: បង្រួមតម្លៃមកចន្លោះ $(0, 1)$ ប៉ុន្តែងាយប្រឈមនឹងបញ្ហា Vanishing Gradient នៅពេលតម្លៃ $|x|$ ធំជ្រុល។

2. **Hyperbolic Tangent (Tanh)**:
   $$
   \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1
   $$
   ដេរីវេក្នុងតំបន់៖
   $$
   \frac{d\tanh(x)}{dx} = 1 - \tanh^2(x)
   $$
   *លក្ខណៈ*: បង្រួមតម្លៃមកចន្លោះ $(-1, 1)$ ហើយមានចំណុចកណ្តាលស្មើសូន្យ (Zero-centered) ដែលជួយសម្រួលដល់ដំណើរការ Optimization។

3. **Rectified Linear Unit (ReLU)**:
   $$
   \text{ReLU}(x) = \max(0, x)
   $$
   ដេរីវេក្នុងតំបន់៖
   $$
   \frac{d\text{ReLU}(x)}{dx} = \begin{cases} 1 & \text{បើ } x > 0 \\ 0 & \text{បើ } x < 0 \end{cases}
   $$
   *(នៅ $x = 0$ អនុគមន៍គ្មានដេរីវេពិតប្រាកដទេ ប៉ុន្តែក្នុងការសរសេរកូដជាក់ស្តែង គេកំណត់យកតម្លៃ $0$ ឬ $1$ ជា Subgradient)*។  
   *លក្ខណៈ*: គណនាលឿនបំផុត មិនធ្វើឱ្យបាត់បង់ជម្រាលនៅប៉ែកវិជ្ជមាន (No saturation)។

4. **Leaky ReLU**:
   $$
   f(x) = \max(\alpha x, x) \quad (\text{ជាមួយ } \alpha \approx 0.01)
   $$
   ដេរីវេក្នុងតំបន់៖
   $$
   \frac{df(x)}{dx} = \begin{cases} 1 & \text{បើ } x > 0 \\ \alpha & \text{បើ } x \le 0 \end{cases}
   $$
   *លក្ខណៈ*: ដោះស្រាយបញ្ហាណឺរ៉ូនងាប់ (Dead ReLU) ដោយផ្តល់ជម្រាលតូចមួយជានិច្ចនៅប៉ែកអវិជ្ជមាន។

---

### ខ. ការគណនាជម្រាលមួយជំហានម្តងៗលើ Computational Graph (Step-by-Step Scalar Trace)

ពិចារណាណឺរ៉ូនទោលមួយដែលមានអនុគមន៍ Sigmoid៖
$$
f(w, x) = \frac{1}{1 + e^{-(w_0 x_0 + w_1 x_1 + w_2)}}
$$

យើងបំបែកវាជាអថេរមធ្យមនៃគំនូសតាងគណនា៖
1. $u_0 = w_0 x_0$
2. $u_1 = w_1 x_1$
3. $q = u_0 + u_1$
4. $a = q + w_2$
5. $b = -a$
6. $c = e^b$
7. $d = 1 + c$
8. $f = \frac{1}{d}$

#### ដំណាក់កាល Forward Pass
ឧបមាថាតម្លៃធាតុចូលជាក់ស្តែងគឺ៖
$$
w_0 = 2.0, \quad x_0 = -1.0, \quad w_1 = -3.0, \quad x_1 = -2.0, \quad w_2 = -3.0
$$

យើងគណនាតាមលំដាប់លំដោយ៖
* $u_0 = 2.0 \times (-1.0) = -2.0$
* $u_1 = -3.0 \times (-2.0) = 6.0$
* $q = -2.0 + 6.0 = 4.0$
* $a = 4.0 + (-3.0) = 1.0$
* $b = -1.0$
* $c = e^{-1.0} \approx 0.368$
* $d = 1.0 + 0.368 = 1.368$
* $f = \frac{1}{1.368} \approx 0.731$

#### ដំណាក់កាល Backward Pass (ក្បួនសង្វាក់ Chain Rule)
ជម្រាលនៃ output ធៀបនឹងខ្លួនឯងគឺ៖
$$
\frac{\partial f}{\partial f} = 1.0
$$

ត្រឡប់ថយក្រោយមួយជំហានម្តងៗ៖
1. **ទ្វារច្រាស ($f = 1/d$)**: $\frac{\partial f}{\partial d} = -\frac{1}{d^2} = -\frac{1}{1.368^2} \approx -0.535$
2. **ទ្វារបូក ១ ($d = 1 + c$)**: $\frac{\partial f}{\partial c} = \frac{\partial f}{\partial d} \cdot \frac{\partial d}{\partial c} = -0.535 \times 1 = -0.535$
3. **ទ្វារអិចស្បូណង់ស្យែល ($c = e^b$)**: $\frac{\partial f}{\partial b} = \frac{\partial f}{\partial c} \cdot \frac{\partial c}{\partial b} = -0.535 \times e^{-1.0} = -0.535 \times 0.368 \approx -0.197$
4. **ទ្វារដក ($b = -a$)**: $\frac{\partial f}{\partial a} = \frac{\partial f}{\partial b} \cdot \frac{\partial b}{\partial a} = -0.197 \times (-1) = 0.197$
5. **ទ្វារបូក Bias ($a = q + w_2$)**:
   * $\frac{\partial f}{\partial w_2} = \frac{\partial f}{\partial a} \cdot 1 = 0.197$
   * $\frac{\partial f}{\partial q} = \frac{\partial f}{\partial a} \cdot 1 = 0.197$
6. **ទ្វារបូក Sum ($q = u_0 + u_1$)**:
   * $\frac{\partial f}{\partial u_0} = 0.197$
   * $\frac{\partial f}{\partial u_1} = 0.197$
7. **ទ្វារគុណ ($u_0 = w_0 x_0$)**:
   * $\frac{\partial f}{\partial w_0} = \frac{\partial f}{\partial u_0} \cdot x_0 = 0.197 \times (-1.0) = -0.197$
   * $\frac{\partial f}{\partial x_0} = \frac{\partial f}{\partial u_0} \cdot w_0 = 0.197 \times 2.0 = 0.394$
8. **ទ្វារគុណ ($u_1 = w_1 x_1$)**:
   * $\frac{\partial f}{\partial w_1} = \frac{\partial f}{\partial u_1} \cdot x_1 = 0.197 \times (-2.0) = -0.394$
   * $\frac{\partial f}{\partial x_1} = \frac{\partial f}{\partial u_1} \cdot w_1 = 0.197 \times (-3.0) = -0.591$

---

### គ. ការគណនាជម្រាលតាមម៉ាទ្រីស និងវ៉ិចទ័រ (Vectorized Matrix Backpropagation)

គោលការណ៍គ្រឹះដ៏សំខាន់បំផុតគឺ៖ **វិមាត្រនៃជម្រាល $\frac{\partial L}{\partial X}$ ត្រូវតែដូចគ្នាបេះបិទនឹងវិមាត្រនៃអថេរដើម $X$ ជានិច្ច!**
$$
\dim\left(\frac{\partial L}{\partial X}\right) = \dim(X)
$$

ពិចារណាប្រមាណវិធីគុណម៉ាទ្រីសក្នុង Fully Connected Layer៖
$$
Y = X W
$$
* $X \in \mathbb{R}^{N \times D}$ (Batch នៃទិន្នន័យ $N$ គំរូ ក្នុងលំហ $D$ វិមាត្រ)
* $W \in \mathbb{R}^{D \times M}$ (ម៉ាទ្រីសទម្ងន់ភ្ជាប់ទៅកាន់ $M$ ណឺរ៉ូន)
* $Y \in \mathbb{R}^{N \times M}$ (លទ្ធផលនៃស្រទាប់)

នៅពេលទទួលបាន Upstream Gradient $\frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times M}$ យើងទាញរកជម្រាល Downstream ដោយប្រើច្បាប់ផ្គូផ្គងវិមាត្រ (Dimension-Matching Algebra)៖

1. **ជម្រាលធៀបនឹងទម្ងន់ $W$**:  
   ដើម្បីទទួលបានម៉ាទ្រីសទំហំ $(D \times M)$ យើងត្រូវគុណម៉ាទ្រីសឆ្លាស់នៃធាតុចូល $X^T$ ទំហំ $(D \times N)$ ជាមួយ $\frac{\partial L}{\partial Y}$ ទំហំ $(N \times M)$៖
   $$
   \frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
   $$

2. **ជម្រាលធៀបនឹងធាតុចូល $X$**:  
   ដើម្បីទទួលបានម៉ាទ្រីសទំហំ $(N \times D)$ យើងត្រូវគុណ upstream gradient $\frac{\partial L}{\partial Y}$ ទំហំ $(N \times M)$ ជាមួយម៉ាទ្រីសឆ្លាស់នៃទម្ងន់ $W^T$ ទំហំ $(M \times D)$៖
   $$
   \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T
   $$

3. **ជម្រាលធៀបនឹង Bias $b \in \mathbb{R}^{1 \times M}$**:  
   ដោយសារ $b$ ត្រូវបានបូកផ្សាយ (Broadcasted) ទៅកាន់គ្រប់ជួរដេកទាំង $N$ នោះជម្រាលរបស់វាគឺជាផលបូកតាមបណ្តោយអ័ក្ស Batch (Axis 0)៖
   $$
   \frac{\partial L}{\partial b} = \sum_{i=1}^{N} \left(\frac{\partial L}{\partial Y}\right)_{i, :}
   $$

---

## ៤. ស្ថាបត្យកម្ម និងក្បួនដោះស្រាយ (Architecture & Algorithm Walkthrough)

### យន្តការ Forward / Backward Cache Flow

```text
                  FORWARD PASS (ឆ្វេងទៅស្តាំ)
       X ────► ┌──────────────────────────────────┐ ────► Y (Output)
       W ────► │         Computational Node       │
               │  (រក្សាទុក X, W ក្នុង Cache)     │
      dX ◄──── ├──────────────────────────────────┤ ◄──── dY (Upstream Gradient)
      dW ◄──── └──────────────────────────────────┘
                 BACKWARD PASS (ស្តាំមកឆ្វេង)
```

### កូដគំរូស្ថាបត្យកម្មគ្រឹះ (Custom Autograd Modular Pipeline ក្នុង PyTorch)

កូដខាងក្រោមបង្ហាញពីការបង្កើត Custom Autograd Layers ក្នុង PyTorch ដោយបំបែកការគណនាជាម៉ូឌុល `forward` និង `backward` ដាច់ដោយឡែកពីគ្នា ព្រមទាំងបង្ហាញពីការរក្សាទុក Cache និងការប្រើ Transposed Matrix Multiplication៖

```python
import torch

class CustomLinear(torch.autograd.Function):
    """
    ស្រទាប់លីនេអ៊ែរពេញលេញ (Fully Connected Layer: Y = XW + b)
    អនុវត្តដោយប្រើរូបមន្តផ្គូផ្គងវិមាត្រនៃម៉ាទ្រីសឆ្លាស់ (Matrix Transposes)
    """
    @staticmethod
    def forward(ctx, X, W, b):
        # រក្សាទុក Tensors សម្រាប់ប្រើប្រាស់ក្នុង Backward Pass
        ctx.save_for_backward(X, W, b)
        return torch.mm(X, W) + b

    @staticmethod
    def backward(ctx, grad_upstream):
        X, W, b = ctx.saved_tensors
        # dX = dY * W^T  (Shape: [N, D] = [N, M] * [M, D])
        grad_X = torch.mm(grad_upstream, W.t())
        # dW = X^T * dY  (Shape: [D, M] = [D, N] * [N, M])
        grad_W = torch.mm(X.t(), grad_upstream)
        # db = ផលបូកនៃ dY តាមបណ្តោយ batch dimension
        grad_b = grad_upstream.sum(dim=0, keepdim=True)
        return grad_X, grad_W, grad_b


class CustomReLU(torch.autograd.Function):
    """
    ស្រទាប់សកម្មកម្ម ReLU តាមធាតុ (Element-wise ReLU: max(0, X))
    """
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        return torch.clamp(X, min=0.0)

    @staticmethod
    def backward(ctx, grad_upstream):
        X, = ctx.saved_tensors
        # បង្កើត Mask: ផ្តល់តម្លៃ 1 ប្រសិនបើ X > 0, ក្រៅពីនោះស្មើ 0
        mask = (X > 0).float()
        grad_X = grad_upstream * mask
        return grad_X


class ModularMLP:
    """
    បណ្តាញញាណ ២ ស្រទាប់ (2-Layer Neural Network)
    ភ្ជាប់ដោយ Custom Forward និង Manual Backward Chain Rule
    """
    def __init__(self, d_in, h_dim, d_out):
        # កំណត់ទម្ងន់ដំបូងដោយប្រើ Small Gaussian Noise
        self.W1 = torch.randn(d_in, h_dim) * 0.01
        self.b1 = torch.zeros(1, h_dim)
        self.W2 = torch.randn(h_dim, d_out) * 0.01
        self.b2 = torch.zeros(1, d_out)

    def train_step(self, X, Y_target, learning_rate=1e-3):
        # ១. FORWARD PASS
        h_linear = CustomLinear.apply(X, self.W1, self.b1)
        h_activated = CustomReLU.apply(h_linear)
        scores = CustomLinear.apply(h_activated, self.W2, self.b2)
        
        # គណនា Mean Squared Error (MSE Loss)
        loss = 0.5 * torch.sum((scores - Y_target) ** 2)
        
        # ២. BACKWARD PASS (ក្បួនសង្វាក់ពីចុងក្រោយមកដើម)
        grad_scores = scores - Y_target
        
        # ថយក្រោយឆ្លងកាត់ Layer ទីពីរ
        grad_h_activated, grad_W2, grad_b2 = CustomLinear.backward(None, grad_scores)
        # ថយក្រោយឆ្លងកាត់ ReLU
        grad_h_linear = CustomReLU.backward(None, grad_h_activated)
        # ថយក្រោយឆ្លងកាត់ Layer ទីមួយ
        grad_X, grad_W1, grad_b1 = CustomLinear.backward(None, grad_h_linear)
        
        # ៣. OPTIMIZER STEP (Vanilla SGD Update)
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2
        
        return loss.item()

# ការសាកល្បងដំណើរការ
if __name__ == "__main__":
    torch.manual_seed(42)
    mlp = ModularMLP(d_in=4, h_dim=8, d_out=2)
    sample_X = torch.randn(5, 4)
    sample_Y = torch.randn(5, 2)
    
    initial_loss = mlp.train_step(sample_X, sample_Y)
    print(f"ការហ្វឹកហាត់ជំហានដំបូងជោគជ័យ! Initial Loss = {initial_loss:.4f}")
```

---

## ៥. ការយល់ដឹងស៊ីជម្រៅតាមបែបវិចារណញាណ (Visual Intuition & Interpretability)

### ការបំប្លែងលំហមិនលីនេអ៊ែរ (Feature Space Warping)

នៅក្នុងចំណាត់ថ្នាក់លីនេអ៊ែរ បន្ទាត់ត្រង់តែមួយមិនអាចញែកទិន្នន័យដែលមានរាងជារង្វង់មូលមូលប្រមូលផ្តុំ (Concentric circles) ឬទម្រង់ XOR បានឡើយ។ អនុគមន៍សកម្មកម្មមិនលីនេអ៊ែរ (ដូចជា ReLU) ធ្វើការបត់ និងពត់កូអរដោនេដើម (Space Warping) ដែលធ្វើឱ្យទិន្នន័យដែលធ្លាប់តែមិនអាចញែកបាន ក្លាយជាអាចញែកដាច់ពីគ្នាបានយ៉ាងងាយដោយបន្ទាត់ត្រង់ក្នុងស្រទាប់បន្ទាប់៖

```text
  លំហធាតុចូលដើម (RAW INPUT SPACE)          លំហលក្ខណៈពិសេសលាក់ (FEATURE SPACE)
        (មិនអាចញែកដោយបន្ទាត់ត្រង់)                (អាចញែកដាច់ដោយបន្ទាត់ត្រង់យ៉ាងងាយ)
         
            o   o   o                                  o   o   o
          o   x   x   o                              ─────────────────────── (Boundary)
            o   o   o                                  x   x   x
     
  [ចំណុច x នៅចំកណ្តាលរង្វង់ o]                     [ស្រទាប់ ReLU បានទាញចំណុច x ឱ្យឃ្លាតចេញ]
```

---

### ទម្រង់គំរូផ្នែករងដែលអាចប្រើប្រាស់ឡើងវិញបាន (Part-Based Reusable Templates)

* **Linear Classifier**: បង្កើតបានត្រឹមតែរូបភាពគំរូទូទៅតែមួយគត់ក្នុងមួយ Class (ឧ. Class រថយន្ត គឺជាដុំពណ៌ក្រហមព្រាលៗដែលច្របូកច្របល់រវាងរថយន្តបែរមុខ និងរថយន្តបែរចំហៀង)។
* **Two-Layer Neural Network**: ស្រទាប់លាក់ដែលមាន $H$ ណឺរ៉ូន ដំណើរការដូចជា "ឧបករណ៍ចាប់សញ្ញាផ្នែករង (Specialized Part Detectors)"៖
  * ណឺរ៉ូនទី ១: ចាប់កង់រថយន្តរាងមូល
  * ណឺរ៉ូនទី ២: ចាប់កញ្ចក់រថយន្តរាងត្រីកោណ
  * ណឺរ៉ូនទី ៣: ចាប់ភ្នែកឆ្មា
  * ណឺរ៉ូនទី ៤: ចាប់ផ្ទៃភ្លឺចាំងនៃលោហៈ  
  ដោយសារវត្ថុជាច្រើនចែករំលែកលក្ខណៈផ្នែករងជាមួយគ្នា (ឆ្កែ និងឆ្មាសុទ្ធតែមានភ្នែក និងត្រចៀក) បណ្តាញញាណអាចផ្សំផ្គុំ (Compose) ផ្នែករងទាំងនេះតាមលំដាប់ថ្នាក់ ដើម្បីតំណាងឱ្យទម្រង់រូបភាពស្មុគស្មាញរាប់ពាន់ប្រភេទ។

---

### ធរណីមាត្រនៃបន្ទាត់ខណ្ឌចែកតាមទំហំស្រទាប់លាក់ (Capacity & Decision Boundaries)

```text
      H = 3 ណឺរ៉ូន                     H = 20 ណឺរ៉ូន                    H = 100 ណឺរ៉ូន
    (Capacity ទាប)                   (Capacity ល្មមសមរម្យ)               (Capacity ខ្ពស់ជ្រុល)
        \      /                            _..---.._                          /\_/\_/\
         \____/                            /         \                        | _   _  |
     បន្ទាត់ពហុកោណសាមញ្ញ               ខ្សែកោងរលោងស្រស់ស្អាត                 បន្ទាត់បត់បែនរញ៉េរញ៉ៃ
    (ប្រឈមនឹង Underfitting)             (ទូទៅភាវូបនីយកម្មល្អ)              (ប្រឈមនឹង Overfitting)
```

* **អនុសាសន៍**: មិនត្រូវបន្ថយទំហំស្រទាប់លាក់ $H$ ដើម្បីការពារ Overfitting ឡើយ! វិធីសាស្ត្រត្រឹមត្រូវក្នុង Deep Learning គឺត្រូវរក្សាទុកទំហំ $H$ ឱ្យធំសមរម្យ ហើយគ្រប់គ្រង Overfitting តាមរយៈការកែតម្រូវទម្ងន់ ($L2$ Regularization / Dropout) វិញ។

---

<div id="plotly-cs231n-4-blueprint" class="plotly-chart" aria-label="Interactive Plotly chart: Lecture 4 visualization blueprint"></div>
<p><em>រូបភាព៖ គំនូសតាងអន្តរកម្មបង្ហាញពីទំហំជម្រាលធ្លាក់ចុះ (Vanishing Gradients) តាមជម្រៅស្រទាប់នៃ Sigmoid (ដេរីវេអតិបរមា 0.25) ធៀបនឹង ReLU (ដេរីវេថេរ 1.0)។</em></p>

---

## ៦. ការពិសោធន៍ជាក់ស្តែង និងលទ្ធផលគោលស្តង់ដារ (Empirical Heuristics & Benchmark Results)

### តារាងប្រៀបធៀបអនុគមន៍សកម្មកម្ម (Activation Functions Benchmark)

| អនុគមន៍សកម្មកម្ម | រូបមន្តគណិតវិទ្យា | ដែនតម្លៃ (Range) | ចំណុចកណ្តាលសូន្យ? | ហានិភ័យ Vanishing Gradient | ការប្រើប្រាស់ទូទៅ |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Sigmoid** | $\frac{1}{1 + e^{-x}}$ | $(0, 1)$ | ❌ ទេ (តែងវិជ្ជមាន) | ខ្ពស់ខ្លាំង (ដេរីវេ $\le 0.25$) | Binary Classification Output |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | ✅ បាទ/ចាស (Zero-centered) | ខ្ពស់នៅតំបន់ឆ្អែត ($|x| > 2$) | RNNs / LSTMs / NLP Models |
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | ❌ ទេ ($x \ge 0$) | គ្មាននៅប៉ែកវិជ្ជមាន ($1.0$) | ស្តង់ដារមាសទូទៅក្នុង CNNs |
| **Leaky ReLU** | $\max(0.01x, x)$ | $(-\infty, \infty)$ | ស្ទើរតែ (Slightly biased) | គ្មាន (រក្សាជម្រាល $0.01$) | ការពារបញ្ហា Dead Neurons |
| **ELU / GELU** | $x \cdot \Phi(x)$ (GELU) | $(-1, \infty)$ | ✅ ជិតស្និទ្ធនឹងសូន្យ | ទាបបំផុត | Transformers (BERT, GPT, ViT) |

* **ច្បាប់កំណត់ទំហំម៉ូដែល (Capacity Tuning Rule)**: កុំព្យាយាមបង្រួមស្រទាប់លាក់ឱ្យតូចដើម្បីកាត់បន្ថយ Overfitting ពីព្រោះម៉ូដែលតូចៗងាយធ្លាក់ចូលទៅក្នុងចំណុច Local Minima អាក្រក់ៗ។ ផ្ទុយទៅវិញ ចូរប្រើប្រាស់បណ្តាញញាណធំ ហើយបង្កើន Regularization Strength ($\lambda$) ដើម្បីទប់ទល់។
* **ពានរង្វាន់ ICLR 2025 Test of Time Award**: ការបណ្តុះបណ្តាលបណ្តាញញាណជ្រៅទាំងនេះពឹងផ្អែកលើក្បួន Optimization ដែលបានរៀបរាប់ក្នុងមេរៀនមុនៗ។ ក្បួនដោះស្រាយ **Adam Optimizer** (Kingma & Ba, 2015) បានទទួលពានរង្វាន់កិត្តិយសខ្ពស់បំផុតក្នុងសន្និសីទ ICLR 2025 ដោយសារការរួមចំណែកមិនអាចខ្វះបានដល់ការហ្វឹកហាត់ Foundation Models ទំនើប។

---

## ៧. ឧបសគ្គដែលត្រូវប្រុងប្រយ័ត្ន និងគន្លឹះដោះស្រាយកំហុស (Pitfalls & Debugging Tips)

* **បញ្ហាណឺរ៉ូនងាប់ (The Dead ReLU Catastrophe)**:
  * ប្រសិនបើទម្ងន់ត្រូវបានកំណត់តម្លៃដំបូងមិនល្អ ឬជំហាន Learning Rate ធំជ្រុលទាញទម្ងន់ឱ្យធ្លាក់ទៅក្នុងតំបន់ដែល $W \cdot x + b < 0$ សម្រាប់គ្រប់ទិន្នន័យក្នុង Dataset នោះ ReLU នឹងបញ្ចេញតម្លៃ $0$ ជានិច្ច។
  * នៅពេល Backward Pass ជម្រាលក្នុងតំបន់របស់វាគឺ $0$ បណ្តាលឱ្យ Downstream Gradient ស្មើនឹងសូន្យរហូត។ ណឺរ៉ូននោះនឹង "ងាប់" ជារៀងរហូត ហើយទម្ងន់របស់វាមិនអាចធ្វើបច្ចុប្បន្នភាពបានទៀតឡើយ។
  * **ដំណោះស្រាយ**: កំណត់តម្លៃទម្ងន់ដំបូងឱ្យបានត្រឹមត្រូវ (He/Kaiming Initialization) ឬប្រើប្រាស់ Leaky ReLU / GELU។
* **បញ្ហាជម្រាលសាបរលាបនៃ Sigmoid (Vanishing Gradients)**:
  * ដោយសារដេរីវេអតិបរមារបស់ Sigmoid គឺត្រឹមតែ $0.25$ ប៉ុណ្ណោះ នោះនៅពេលយើងគុណជម្រាលកាត់តាម ១០ ស្រទាប់ ជម្រាលដែលហូរទៅដល់ស្រទាប់ដំបូងនឹងត្រូវបង្រួមតូចរហូតដល់ $0.25^{10} \approx 9.5 \times 10^{-7}$ (ស្ទើរតែស្មើ ០ បេះបិទ!)។ ស្រទាប់ដំបូងៗនឹងមិនព្រមរៀនអ្វីទាំងអស់។
* **កំហុសឆ្លាស់ម៉ាទ្រីសក្នុងការសរសេរកូដ (Transposed Matrix Alignment Bugs)**:
  * កំហុសស្ងាត់ដែលពិបាករកឃើញបំផុតគឺការសរសេរច្រឡំលំដាប់គុណម៉ាទ្រីសក្នុង Backward Pass (ឧ. សរសេរ `torch.mm(W.t(), grad_upstream)` ជំនួសឱ្យ `torch.mm(grad_upstream, W.t())`)។
  * **គន្លឹះដោះស្រាយ**: តែងតែពិនិត្យមើល Shape នៃ Tensors មុននិងក្រោយគុណ៖ ត្រូវប្រាកដថា $\dim(\frac{\partial L}{\partial W}) == \dim(W)$ និង $\dim(\frac{\partial L}{\partial X}) == \dim(X)$ ជានិច្ច!
* **ការភ្លេច Cache ឬកែប្រែ Cache ផ្ទាល់ (In-place Mutation Bugs)**:
  * ប្រសិនបើអ្នកកែប្រែ Tensor ក្នុង Forward Pass ដោយប្រើប្រមាណវិធី In-place (ដូចជា `X += 1` ឬ `X[mask] = 0`) នោះតម្លៃចាស់ដែលត្រូវការចាំបាច់សម្រាប់ Backward Pass នឹងត្រូវខូចខាត បណ្តាលឱ្យការគណនាជម្រាលខុសទាំងស្រុង។

---

## ៨. សំណួរពិចារណាកម្រិតក្រោយឧត្តមសិក្សា (Graduate-Level Reflection Questions)

១. **សម្រាយបញ្ជាក់គណិតវិទ្យានៃការស្រុតចុះនៃបណ្តាញលីនេអ៊ែរ (Analytical Proof of Stacked Linearity)**:  
   ចូរធ្វើសម្រាយបញ្ជាក់តាមគណិតវិទ្យាថា បណ្តាញញាណដែលមាន ១០០ ស្រទាប់ដោយគ្មានអនុគមន៍សកម្មកម្ម៖
   $$
   f(x) = W_{100} W_{99} \dots W_2 W_1 x
   $$
   មានសមត្ថភាពតំណាង (Representational Capacity) ស្មើគ្នាបេះបិទនឹងម៉ូដែលលីនេអ៊ែរតែមួយស្រទាប់ $f(x) = W_{\text{equiv}} x$។ តើវត្តមាននៃ Bias vectors $b_i$ អាចជួយឱ្យម៉ូដែលរៀនទម្រង់មិនលីនេអ៊ែរបានដែរឬទេ? ចូរពន្យល់តាមទ្រឹស្តីពីជគណិតលីនេអ៊ែរ។

២. **បញ្ហាស្ទះអង្គចងចាំនៃម៉ាទ្រីស Jacobian (The Jacobian Memory Bottleneck)**:  
   ឧបមាថាអ្នកកំពុងហ្វឹកហាត់ស្រទាប់ Linear Layer មួយដែលមាន Batch size $N = 100$, វិមាត្រធាតុចូល $D = 4096$, និងវិមាត្រធាតុចេញ $M = 4096$៖
   * ចូរគណនាវិមាត្រនៃម៉ាទ្រីសដេរីវេ Jacobian ពេញលេញ $\frac{\partial Y}{\partial X}$។
   * ចូរគណនាទំហំអង្គចងចាំ RAM (គិតជា Gigabytes ក្នុងទម្រង់ FP32 - 4 bytes ក្នុងមួយធាតុ) ដែលចាំបាច់ត្រូវប្រើដើម្បីផ្ទុកម៉ាទ្រីស Jacobian នេះ។
   * ហេតុអ្វីបានជារូបមន្ត Matrix Backpropagation $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T$ អាចជួយយើងចៀសផុតពីការគណនា និងផ្ទុកម៉ាទ្រីស Jacobian ដ៏មហិមានេះទាំងស្រុង?

៣. **អត្ថប្រយោជន៍នៃអនុគមន៍ Zero-Centered (Zero-Centered Activation Dynamics)**:  
   ហេតុអ្វីបានជាអនុគមន៍សកម្មកម្មដែលគ្មានចំណុចកណ្តាលសូន្យដូចជា Sigmoid (ដែលមាន Output តែងតែវិជ្ជមានជានិច្ច $\sigma(x) > 0$) បណ្តាលឱ្យជម្រាលនៃទម្ងន់ $\frac{\partial L}{\partial W}$ ក្នុងស្រទាប់បន្ទាប់ មានសញ្ញាដូចគ្នាទាំងអស់ (ទាំងអស់គ្នាវិជ្ជមាន ឬទាំងអស់គ្នាអវិជ្ជមាន)? តើបាតុភូតនេះបង្កផលប៉ះពាល់អវិជ្ជមានយ៉ាងដូចម្តេចខ្លះដល់ល្បឿននៃការរៀន (Zigzagging Dynamics) ក្នុងពេលធ្វើ Gradient Descent?
