# កំណត់ចំណាំមេរៀន: មេរៀនទី ៤: អម្បូរបំណែងចែកអិចស្ប៉ូណង់ស្យែល, GLMs & ការធ្វើចំណាត់ថ្នាក់ (Exponential Family, GLMs & Classification)

**វគ្គសិក្សា**: Stanford CS229: Foundations and Paradigms of Machine Learning (មូលដ្ឋានគ្រឹះ និងទម្រង់នៃការរៀនរបស់ម៉ាស៊ីន — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Chris Ré (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Tengyu Ma (សាស្ត្រាចារ្យរងផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs229.stanford.edu](https://cs229.stanford.edu/index.html-spr26) (គ្រោងមេរៀន និងឯកសារសិក្សា)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Summary of Main Ideas)

មេរៀននេះណែនាំអំពី **អម្បូរបំណែងចែកអិចស្ប៉ូណង់ស្យែល (Exponential Family of Distributions)** ដែលជាក្របខ័ណ្ឌគណិតវិទ្យាបង្រួបបង្រួមដ៏មានឥទ្ធិពលបំផុតមួយ សម្រាប់តភ្ជាប់ម៉ូដែល Machine Learning បុរាណជាច្រើនដែលបានសិក្សាក្នុងវគ្គនេះ។ សាស្ត្រាចារ្យ Chris Ré បានបង្ហាញថា បំណែងចែកកំហុសទូទៅជាច្រើន (ដូចជា Bernoulli សម្រាប់ទិន្នន័យទ្វេភាគ និង Gaussian សម្រាប់ទិន្នន័យបន្តបន្ទាប់) សុទ្ធសឹងតែជាករណីពិសេសនៃអម្បូរអិចស្ប៉ូណង់ស្យែលនេះទាំងអស់។

ផ្អែកលើគ្រឹះដ៏រឹងមាំនេះ មេរៀនបានណែនាំអំពី **គំរូលីនេអ៊ែរទូទៅ (Generalized Linear Models - GLMs)** ដែលផ្តល់នូវរូបមន្តរៀបចំ ៣ ជំហានដ៏ច្បាស់លាស់ក្នុងការកសាងម៉ូដែលទស្សន៍ទាយសម្រាប់ប្រភេទបញ្ហាស្ទើរតែទាំងអស់ក្នុងពិភពពិត។ មេរៀនបញ្ចប់ដោយការផ្តោតលើក្បួនដោះស្រាយចំណាត់ថ្នាក់ដ៏មានសារៈសំខាន់បំផុតក្នុងវិស័យឧស្សាហកម្មបច្ចុប្បន្ន គឺ **ការវិភាគតម្រែតម្រង់ Softmax (Softmax Regression)** សម្រាប់ការធ្វើចំណាត់ថ្នាក់ពហុថ្នាក់ (Multi-Class Classification)។ សាស្ត្រាចារ្យបានភ្ជាប់គំនិតនេះទៅនឹងស្ថាបត្យកម្ម Deep Learning ទំនើបៗ រួមទាំងយន្តការ Attention Heads ក្នុងម៉ូដែល Transformer (ដូចជា GPT-2) និងបានណែនាំបច្ចេកទេស Regularization ដ៏មានប្រសិទ្ធភាពគឺ **Label Smoothing**។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

* **អម្បូរបំណែងចែកអិចស្ប៉ូណង់ស្យែល (Exponential Family)**: ថ្នាក់នៃបំណែងចែកប្រូបាប៊ីលីតេ ដែលអនុគមន៍ដង់ស៊ីតេប្រូបាប៊ីលីតេ (PDF) ឬអនុគមន៍ម៉ាសប្រូបាប៊ីលីតេ (PMF) អាចសរសេរបានក្នុងទម្រង់កាណូនិច (Canonical Form)៖
  $$p(y; \eta) = b(y) \exp\left(\eta^T T(y) - a(\eta)\right)$$
* **ប៉ារ៉ាម៉ែត្រធម្មជាតិ ($\eta$ - Natural Parameter)**: ប៉ារ៉ាម៉ែត្រកាណូនិច (ឬវ៉ិចទ័រនៃប៉ារ៉ាម៉ែត្រ) ដែលកំណត់លក្ខណៈនៃបំណែងចែក។
* **ស្ថិតិគ្រប់គ្រាន់ ($T(y)$ - Sufficient Statistic)**: អនុគមន៍នៃទិន្នន័យ $y$ ដែលក្តោបក្តាប់រាល់ព័ត៌មានទាំងអស់ដែលទិន្នន័យមានអំពីប៉ារ៉ាម៉ែត្រ។ សម្រាប់ករណីភាគច្រើនក្នុងវគ្គសិក្សានេះ $T(y) = y$ (អនុគមន៍អត្តសញ្ញាណ)។
* **រង្វាស់មូលដ្ឋាន ($b(y)$ - Base Measure)**: កត្តាធ្វើមាត្រដ្ឋាន ឬអនុគមន៍ស្កាលែដែលអាស្រ័យតែលើទិន្នន័យ $y$ ប៉ុណ្ណោះ និងមិនអាស្រ័យលើប៉ារ៉ាម៉ែត្រឡើយ។
* **អនុគមន៍ Log Partition ($a(\eta)$ - Log Partition Function)**: កត្តាកំណត់ស្តង់ដារ (Normalization Constant) ដែលធានាថាផលបូក ឬអាំងតេក្រាលនៃប្រូបាប៊ីលីតេមានតម្លៃស្មើ ១ ជានិច្ច កំណត់ដោយ៖
  $$a(\eta) = \log \int b(y) \exp\left(\eta^T T(y)\right) dy$$
* **អនុគមន៍ភ្ជាប់ (Link Function, $g^{-1}$)**: អនុគមន៍ដែលផ្គូផ្គងតម្លៃរំពឹងទុកនៃបំណែងចែកគោលដៅ ទៅកាន់ប៉ារ៉ាម៉ែត្រធម្មជាតិ $\eta$។
* **អនុគមន៍ឆ្លើយតបកាណូនិច (Canonical Response Function, $g$)**: អនុគមន៍ដែលផ្គូផ្គងប៉ារ៉ាម៉ែត្រធម្មជាតិ $\eta$ ទៅកាន់តម្លៃរំពឹងទុកនៃបំណែងចែកគោលដៅ $\mathbb{E}[T(y)]$។
* **អនុគមន៍ Softmax (Softmax Function)**: ទម្រង់ទូទៅនៃអនុគមន៍ Sigmoid/Logistic ដែលបម្លែងវ៉ិចទ័រពិន្ទុតម្លៃពិតតាមអំពើចិត្ត (Logits) ទៅជាបំណែងចែកប្រូបាប៊ីលីតេលើ $k$ ថ្នាក់ដាច់ដោយឡែកពីគ្នា។
* **ការធ្វើឱ្យស្លាកសម្គាល់រលូន (Label Smoothing)**: បច្ចេកទេស Regularization ដែលជំនួសស្លាកសម្គាល់រឹង One-Hot ដោយបំណែងចែកគោលដៅទន់ និងរលូន ដើម្បីកាត់បន្ថយបញ្ហា Overfitting និងបង្កើនសមត្ថភាពឆ្លើយតបនឹងទិន្នន័យថ្មី។

---

## ៣. រូបមន្តគណិតវិទ្យា និងការទាញសមីការ (Mathematical Formulations & Derivations)

### ក. លក្ខណៈសម្បត្តិបង្រួបបង្រួមនៃ $a(\eta)$

អនុគមន៍ Log Partition $a(\eta)$ ដើរតួជា Cumulant Generating Function ដែលធានាថាដេរីវេរបស់វាផ្តល់នូវម៉ូម៉ង់ស្ថិតិ (Statistical Moments) នៃស្ថិតិគ្រប់គ្រាន់ $T(y)$ ដោយស្វ័យប្រវត្តិ៖

1. **ដេរីវេទីមួយ (តម្លៃរំពឹងទុក - Expectation)**:
   $$\frac{\partial}{\partial \eta} a(\eta) = \mathbb{E}[T(y)]$$
2. **ដេរីវេទីពីរ (វ៉ារ្យ៉ង់ - Variance)**:
   $$\frac{\partial^2}{\partial \eta^2} a(\eta) = \text{Var}(T(y))$$

ដោយសារដេរីវេទីពីរតំណាងឱ្យវ៉ារ្យ៉ង់ (ដែលជាតម្លៃមិនអវិជ្ជមានជានិច្ច $\ge 0$) នោះនាំឱ្យ $a(\eta)$ ត្រូវបានធានាថាជា **អនុគមន៍ប៉ោង (Convex Function)** ដែលធ្វើឱ្យការបង្កើនប្រសិទ្ធភាពតាម Gradient មានស្ថេរភាពខ្ពស់ និងរួមគ្នាមកកាន់ចំណុចប្រសើរបំផុតសកលជានិច្ច។

### ខ. ភស្តុតាង៖ បំណែងចែក Bernoulli ជាសមាជិកនៃ Exponential Family

ពិចារណាលើការបោះកាក់ដែលតាងដោយបំណែងចែក Bernoulli ជាមួយប៉ារ៉ាម៉ែត្រ $\phi \in (0, 1)$ តំណាងឱ្យប្រូបាប៊ីលីតេចំពោះក្បាល ($y=1$)៖
$$p(y; \phi) = \phi^y (1-\phi)^{1-y}$$

ដើម្បីបម្លែងសមីការនេះទៅជាទម្រង់កាណូនិចនៃ Exponential Family យើងយកអិចស្ប៉ូណង់ស្យែលលើលោការីតធម្មជាតិនៃបំណែងចែក៖
$$p(y; \phi) = \exp\left(\log\left(\phi^y (1-\phi)^{1-y}\right)\right)$$
$$p(y; \phi) = \exp\left(y \log \phi + (1-y) \log(1-\phi)\right)$$
$$p(y; \phi) = \exp\left(y \log \phi - y\log(1-\phi) + \log(1-\phi)\right)$$
$$p(y; \phi) = \exp\left(y \log\left(\frac{\phi}{1-\phi}\right) + \log(1-\phi)\right)$$

ប្រៀបធៀបតួនេះទៅនឹងទម្រង់កាណូនិច $b(y) \exp\left(\eta^T T(y) - a(\eta)\right)$ យើងកំណត់បាន៖
* $b(y) = 1$
* $T(y) = y$
* $\eta = \log\left(\frac{\phi}{1-\phi}\right) \quad \text{(នេះជាអនុគមន៍ Log-Odds ឬ Logit)}$
* $a(\eta) = -\log(1-\phi)$

ដើម្បីសរសេរ $a(\eta)$ ឱ្យអាស្រ័យសុទ្ធសាធលើ $\eta$ យើងដោះស្រាយរក $\phi$ ជាអនុគមន៍នៃ $\eta$៖
$$\mathrm{e}^\eta = \frac{\phi}{1-\phi} \implies \phi(1-\phi)\mathrm{e}^\eta = \phi \implies \phi = \frac{\mathrm{e}^\eta}{1 + \mathrm{e}^\eta} = \frac{1}{1 + \mathrm{e}^{-\eta}}$$
នេះបង្ហាញឡើងវិញនូវ **អនុគមន៍ Sigmoid/Logistic** ដ៏ល្បីល្បាញ! ជំនួស $\phi$ ចូលក្នុង $a(\eta)$ យើងទទួលបាន៖
$$a(\eta) = -\log\left(1 - \frac{1}{1+\mathrm{e}^{-\eta}}\right) = -\log\left(\frac{\mathrm{e}^{-\eta}}{1+\mathrm{e}^{-\eta}}\right) = \log\left(1 + \mathrm{e}^\eta\right)$$

### គ. ភស្តុតាង៖ បំណែងចែក Gaussian ជាសមាជិកនៃ Exponential Family

ពិចារណាលើបំណែងចែក Gaussian ដែលមានមធ្យម $\mu$ និងវ៉ារ្យ៉ង់ថេរ $\sigma^2 = 1$៖
$$p(y; \mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(y-\mu)^2}{2}\right)$$

ពន្លាតកន្សោមដឺក្រេទីពីរក្នុងស្វ័យគុណ៖
$$p(y; \mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2} + \mu y - \frac{\mu^2}{2}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right) \exp\left(\mu y - \frac{\mu^2}{2}\right)$$

ប្រៀបធៀបតួទៅនឹង $b(y) \exp\left(\eta^T T(y) - a(\eta)\right)$៖
* $b(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right)$
* $T(y) = y$
* $\eta = \mu$
* $a(\eta) = \frac{\mu^2}{2} = \frac{\eta^2}{2}$

---

## ៤. គំរូលីនេអ៊ែរទូទៅ (Generalized Linear Models - GLMs)

ដើម្បីបង្កើត GLM សម្រាប់ទស្សន៍ទាយអថេរគោលដៅ $y$ ដោយផ្អែកលើលក្ខណៈពិសេស $x \in \mathbb{R}^{d+1}$ និងប៉ារ៉ាម៉ែត្រ $\theta \in \mathbb{R}^{d+1}$ យើងផ្អែកលើការសន្មតរៀបចំចំនួន ៣៖

1. **កំហុសរំខានជា Exponential Family**: បំណែងចែកមានលក្ខខណ្ឌនៃ $y$ ធៀបនឹង $x$ ស្ថិតនៅក្នុង Exponential Family កំណត់ដោយប៉ារ៉ាម៉ែត្រ $\eta$៖
   $$y \mid x; \theta \sim \text{ExponentialFamily}(\eta)$$
2. **ការទស្សន៍ទាយតម្លៃរំពឹងទុក**: គោលដៅរបស់យើងគឺទស្សន៍ទាយតម្លៃរំពឹងទុកនៃ $T(y)$ ដោយផ្តល់ទិន្នន័យ $x$។ នោះមានន័យថា អនុគមន៍សម្មតិកម្ម $h_\theta(x)$ ត្រូវបានកំណត់ដោយ៖
   $$h_\theta(x) = \mathbb{E}[T(y) \mid x]$$
3. **ភាពលីនេអ៊ែរ**: ប៉ារ៉ាម៉ែត្រធម្មជាតិ $\eta$ និងលក្ខណៈពិសេស $x$ មានទំនាក់ទំនងលីនេអ៊ែរតាមរយៈទម្ងន់ម៉ូដែល $\theta$៖
   $$\eta = \theta^T x$$

### ទម្រង់ជម្រាលកែកំហុសរួម (Common Error-Correcting Gradient Update)

ក្រោមការប៉ាន់ស្មានលទ្ធភាពអតិបរមា (MLE) ការគណនា Log-Likelihood នៃ GLM ណាមួយ និងការរកជម្រាលធៀបនឹង $\theta$ ផ្តល់នូវវិធានធ្វើបច្ចុប្បន្នភាពកែកំហុសដូចគ្នាបេះបិទសម្រាប់ Stochastic Gradient Descent (SGD) កាត់តាម *គ្រប់* បំណែងចែកក្នុងអម្បូរនេះទាំងអស់៖
$$\theta_j := \theta_j - \alpha \left( h_{\theta^{(t)}}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

នេះមិនមែនជារឿងចៃដន្យតាមពីជគណិតទេ ប៉ុន្តែជារចនាសម្ព័ន្ធផ្ទាល់ដែលកើតចេញពីលក្ខណៈសម្បត្តិកាណូនិចនៃ Exponential Family!

<div id="plotly-glm-crank" class="plotly-chart" aria-label="Interactive Plotly diagram: the GLM parameter crank mapping features through weights, the natural parameter, and the canonical response function to the prediction"></div>

---

## ៥. ការធ្វើចំណាត់ថ្នាក់ពហុថ្នាក់៖ ការវិភាគតម្រែតម្រង់ Softmax (Softmax Regression)

នៅពេលទស្សន៍ទាយលទ្ធផលដាច់ដោយឡែករវាង $k > 2$ ថ្នាក់ (ឧ. ការចាត់ថ្នាក់រូបភាពជា ឆ្មា ឆ្កែ ឡាន ឬឡានក្រុង) យើងប្រើប្រាស់ **Softmax Regression**។

### តំណាងវ៉ិចទ័រ One-Hot (One-Hot Vector Representation)

ជំនួសឱ្យការដាក់ស្លាកថ្នាក់ជាលេខស្កាលែ $\{1, 2, 3, 4\}$ យើងតំណាងឱ្យពួកវាជាវ៉ិចទ័រ One-Hot ក្នុង $\mathbb{R}^k$៖
* $\text{ឆ្មា} = [1, 0, 0, 0]^T$
* $\text{ឆ្កែ} = [0, 1, 0, 0]^T$
* $\text{ឡាន} = [0, 0, 1, 0]^T$
* $\text{ឡានក្រុង} = [0, 0, 0, 1]^T$

ប៉ារ៉ាម៉ែត្ររបស់យើងរួមមានវ៉ិចទ័រប៉ារ៉ាម៉ែត្រចំនួន $k$ ផ្សេងគ្នា សម្រាប់ថ្នាក់នីមួយៗ៖ $\theta_1, \theta_2, \dots, \theta_k \in \mathbb{R}^{d+1}$។ ពិន្ទុ (Logit) សម្រាប់ថ្នាក់ $j$ ចំពោះធាតុចូល $x$ គឺ $\theta_j^T x$។

### រូបមន្ត Softmax

ប្រូបាប៊ីលីតេមានលក្ខខណ្ឌដែលធាតុចូល $x$ ស្ថិតក្នុងថ្នាក់ $j$ ត្រូវបានគណនាដោយការលើកជាអិចស្ប៉ូណង់ស្យែល និងធ្វើស្តង់ដារលើផលបូកពិន្ទុទាំងអស់៖
$$p(y = j \mid x; \theta) = \frac{\exp(\theta_j^T x)}{\sum_{l=1}^k \exp(\theta_l^T x)}$$

### អនុគមន៍បាត់បង់ Cross-Entropy (Cross-Entropy Loss)

ក្រោម MLE ការធ្វើអតិបរមាកម្មលើ Multinomial Log-Likelihood គឺស្មើគ្នានឹងការកាត់បន្ថយ Cross-Entropy Loss លើទិន្នន័យហ្វឹកហាត់ $n$ ឧទាហរណ៍៖
$$\mathcal{L}(\theta) = -\sum_{i=1}^n \sum_{j=1}^k y_j^{(i)} \log p(y^{(i)} = j \mid x^{(i)}; \theta)$$

<div id="plotly-softmax-geometry" class="plotly-chart" aria-label="Interactive Plotly chart: softmax decision regions for four classes in a 2D feature plane with probability heatmap"></div>

---

## ៦. ការធ្វើឱ្យស្លាកសម្គាល់រលូន (Label Smoothing)

ស្លាកសម្គាល់ One-Hot "រឹង" ធម្មតា $y^{(i)} = [1, 0, 0, 0]^T$ បង្ខំឱ្យ Cross-Entropy រុញ Logits របស់ម៉ូដែល $\theta_1^T x \to \infty$ ធៀបនឹងថ្នាក់ផ្សេងៗ ដើម្បីសម្រេចបានប្រូបាប៊ីលីតេ ១.០ ឥតខ្ចោះ។ ចំណុចនេះនាំឱ្យម៉ូដែលកើតបញ្ហា Overfitting ធ្ងន់ធ្ងរ និងមានទំនុកចិត្តខ្ពស់ជ្រុលហួសហេតុ (Overconfident)។

**Label Smoothing** កែប្រែវ៉ិចទ័រគោលដៅដោយបែងចែកទម្ងន់ប្រូបាប៊ីលីតេតូចមួយ $\epsilon$ ស្មើៗគ្នាទៅគ្រប់ថ្នាក់ទាំងអស់៖
$$y_{\text{smooth}, j} = y_j (1 - \epsilon) + \frac{\epsilon}{k}$$

ឧទាហរណ៍ ក្នុងបញ្ហាចំណាត់ថ្នាក់ ៤ ថ្នាក់ ជាមួយ $\epsilon = 0.1$ ស្លាកសម្គាល់ផ្លាស់ប្តូរពី៖
$$y = [1, 0, 0, 0]^T \implies y_{\text{smooth}} = [0.925, 0.025, 0.025, 0.025]^T$$

### អត្ថប្រយោជន៍៖
* **ទប់ស្កាត់ Overfitting**: ការពារមិនឱ្យទម្ងន់ $\theta$ ផ្ទុះកើនឡើងដល់អនន្ត។
* **ភាពធន់ (Robustness)**: ជួយឱ្យម៉ូដែលធន់នឹងកំហុសទិន្នន័យដែលមានស្លាកសម្គាល់ខុស (Label Noise)។
* **ស្ថេរភាពនៃជម្រាល (Gradient Stability)**: ធានាថាលំហូរ Gradient ដំណើរការល្អ និងមានសុខភាពល្អអំឡុងពេល Backpropagation។

---

## ៧. កម្មវិធីអនុវត្តជាក់ស្តែង (Practical Applications)

* **ការទស្សន៍ទាយ Token បន្ទាប់ក្នុង LLMs (Next-Token Prediction)**: ម៉ូដែលភាសាធំៗទំនើប (ដូចជា ChatGPT និង Claude) ប្រើប្រាស់ស្រទាប់ Softmax ដ៏ធំសម្បើមជា Prediction Head ចុងក្រោយ ដើម្បីជ្រើសរើសពាក្យ/Token បន្ទាប់ដែលមានប្រូបាប៊ីលីតេខ្ពស់បំផុតពីវចនានុក្រម (ជាទូទៅមានចន្លោះពី $50k$ ដល់ $250k$ Tokens)។
* **យន្តការ Self-Attention ក្នុង Transformers**: ការធ្វើស្តង់ដារតាម Softmax គឺជាបេះដូងគណិតវិទ្យានៃរូបមន្ត Self-Attention ក្នុងស្ថាបត្យកម្ម Transformer៖
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  ដែល Softmax ធ្វើមាត្រដ្ឋាន និងកំណត់កម្រិតពាក់ព័ន្ធរវាង Tokens នីមួយៗដោយស្វ័យប្រវត្តិ។

---

## ៨. សំណួរត្រិះរិះពិចារណា (Reflection Questions)

១. **ភាពប៉ោងតាមបែបគណិតវិទ្យា (Mathematical Convexity)**: ហេតុអ្វីបានជាអនុគមន៍ Log Partition $a(\eta)$ ត្រូវបានធានាថាជាអនុគមន៍ប៉ោង? តើភាពប៉ោងនេះផ្តល់ផលវិជ្ជមានអ្វីខ្លះដល់ការហ្វឹកហាត់ GLM តាមរយៈ Stochastic Gradient Descent?  
២. **ភាពច្រំដែលក្នុង Softmax (Softmax Redundancy)**: ចូរបង្ហាញថាទម្រង់ Softmax មានប៉ារ៉ាម៉ែត្រច្រើនលើសលប់ (Overparameterized) ដោយស្រាយបញ្ជាក់ថាការដកវ៉ិចទ័រថេរ $\psi$ ពីគ្រប់ទម្ងន់ថ្នាក់ $\theta_j$ ទាំងអស់ មិនធ្វើឱ្យប៉ះពាល់ដល់ប្រូបាប៊ីលីតេ $p(y = j \mid x; \theta)$ ឡើយ។ តើភាពច្រំដែលនេះត្រូវបានដោះស្រាយយ៉ាងដូចម្តេចក្នុងស្ថិតិបុរាណ ធៀបនឹងការអនុវត្តជាក់ស្តែងក្នុង Deep Learning ទំនើប?  
៣. **ដែនកំណត់នៃភាពខណ្ឌចែកលីនេអ៊ែរ (Linear Separability)**: នៅក្នុងលំហដែលមានវិមាត្រទាប ម៉ូដែលលីនេអ៊ែរ GLMs (ដូចជា Logistic ឬ Softmax Regression) អាចញែកបានតែទិន្នន័យណាដែលខណ្ឌចែកលីនេអ៊ែរប៉ុណ្ណោះ។ ហេតុអ្វីបានជាលំហដែលមានវិមាត្រខ្ពស់ (ដូចជាលំហ ៧៦៨ វិមាត្ររបស់ GPT-2) ជួយដោះស្រាយបញ្ហានេះបានយ៉ាងមានប្រសិទ្ធភាព និងធ្វើឱ្យ Linear Classifiers ដំណើរការបានល្អអស្ចារ្យ?

---

## ៩. ឯកសារយោង & ការអានបន្ថែម (Further Reading & Resources)

* **Convex Optimization (Boyd & Vandenberghe)**: ជំពូកទី ៣ ស្តីអំពីអនុគមន៍ Log-Convex និងធរណីមាត្រនៃ Partition Functions។
* **The Exponential Family in Statistics (Brown, 1986)**: សៀវភៅគោលដ៏ល្បីល្បាញសម្រាប់ការយល់ដឹងស៊ីជម្រៅអំពីលក្ខណៈសម្បត្តិ Measure-Theoretic នៃ Exponential Families។
* **"Large Language Monkeys" (CS229 Lecture Lore)**: ការស្រាវជ្រាវរបស់សាស្ត្រាចារ្យស្តីពីថាមពលស្ថិតិនៃការទាញសំណាក Token ដោយចៃដន្យ និងរបៀបដែល Temperature Scaling ជះឥទ្ធិពលលើគន្លងនៃការបង្កើតអត្ថបទ។
