# កំណត់ចំណាំមេរៀន: មេរៀនទី ៣: ការវិភាគតម្រែតម្រង់ឡូជីស្ទីក និង Weighted Least Squares (Logistic Regression & Weighted Least Squares)

**វគ្គសិក្សា**: Stanford CS229: Foundations and Paradigms of Machine Learning (មូលដ្ឋានគ្រឹះ និងទម្រង់នៃការរៀនរបស់ម៉ាស៊ីន — និទាឃរដូវ ឆ្នាំ២០២៦)  
**សាស្ត្រាចារ្យ**: Chris Ré (សាស្ត្រាចារ្យផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ) និង Tengyu Ma (សាស្ត្រាចារ្យរងផ្នែកវិទ្យាសាស្ត្រកុំព្យូទ័រ)  
**គេហទំព័រវគ្គសិក្សា**: [cs229.stanford.edu](https://cs229.stanford.edu/index.html-spr26) (គ្រោងមេរៀន និងឯកសារសិក្សា)  

---

## ១. សេចក្តីសង្ខេបនៃគំនិតចម្បងៗ (Summary of Main Ideas)

មេរៀននេះផ្តោតសំខាន់លើការភ្ជាប់ទំនាក់ទំនងយ៉ាងស៊ីជម្រៅរវាង **ស្ថិតិប្រូបាប៊ីលីតេ (Probability Statistics)** និង **ក្បួនដោះស្រាយបង្កើនប្រសិទ្ធភាពក្នុងការរៀនរបស់ម៉ាស៊ីន (Machine Learning Optimization Algorithms)**។ ដំបូងបង្អស់ សាស្ត្រាចារ្យ Chris Ré បានផ្តល់នូវមូលដ្ឋានគ្រឹះយុត្តិកម្មតាមបែបប្រូបាប៊ីលីតេយ៉ាងរឹងមាំសម្រាប់ **មុខងារថ្លៃដើម Least Squares** ពីមេរៀនទី ២ ដោយបង្ហាញថា ការកាត់បន្ថយផលបូកការេនៃកំហុសលំអៀង គឺដូចគ្នាបេះបិទទៅនឹងការធ្វើអតិបរមាកម្មលើលទ្ធភាព (Log-Likelihood) ក្រោមការសន្មតថាទិន្នន័យមានកំហុសរំខាន Gaussian ដែលមានលក្ខណៈឯករាជ្យ និងបំណែងចែកដូចគ្នា (Independent and Identically Distributed - IID Gaussian noise)។

បន្ទាប់មក មេរៀនបានផ្លាស់ប្តូរទិសដៅពីការវិភាគតម្រែតម្រង់ជាលេខបន្តបន្ទាប់ (Continuous Regression) ទៅកាន់ **ការធ្វើចំណាត់ថ្នាក់ទ្វេភាគ (Binary Classification)**។ សាស្ត្រាចារ្យបានគូសបញ្ជាក់ពីចំណុចខ្វះខាតគ្រឹះនៃការយក Linear Regression មកប្រើដោយផ្ទាល់លើបញ្ហាចំណាត់ថ្នាក់ (ដូចជាការបង្ខូចទ្រង់ទ្រាយព្រំដែនសម្រេចចិត្តដោយសារតែទិន្នន័យ Outliers នៅឆ្ងាយ) ហើយបានណែនាំនូវ **ការវិភាគតម្រែតម្រង់ឡូជីស្ទីក (Logistic Regression)**។ តាមរយៈការបញ្ជូនលទ្ធផលបន្សំលីនេអ៊ែរនៃ Features ឆ្លងកាត់ **អនុគមន៍ភ្ជាប់ Sigmoid (Sigmoid Link Function)** ដែលមានរាងកោងរលូន និងកើនឡើងជាលំដាប់ ម៉ូដែលនេះអាចបញ្ចេញតម្លៃប្រូបាប៊ីលីតេត្រឹមត្រូវចន្លោះពី ០ ដល់ ១។ ការទាញដេរីវេ Log-Likelihood នៃម៉ូដែលចំណាត់ថ្នាក់នេះ និងការបង្កើនប្រសិទ្ធភាពតាមរយៈ **Gradient Ascent** បានបង្ហាញនូវលទ្ធផលគួរឱ្យភ្ញាក់ផ្អើលមួយ៖ ទម្រង់ជម្រាលកែកំហុស (Error-correcting gradient) មានទម្រង់រូបមន្ត $y - h(x)$ ដូចគ្នាទាំងស្រុងទៅនឹង Linear Regression!

ជាចុងក្រោយ មេរៀនបានស្វែងយល់លើវិធីសាស្ត្របង្កើនប្រសិទ្ធភាពលំដាប់ទីពីរ គឺ **វិធីសាស្ត្រញូតុន (Newton's Method)**។ សាស្ត្រាចារ្យ Chris Ré បានប្រៀបធៀបល្បឿននៃការរួមគ្នាក្នុងកម្រិតការ៉េ (Quadratic Convergence) ដ៏លឿនបំផុតរបស់ Newton's Method ដែលមិនទាមទារការកំណត់ Learning Rate ទៅនឹងភាពស្មុគស្មាញនៃការគណនាកម្រិតធ្ងន់ $O(nd^2 + d^3)$ ក្នុងមួយជំហាន។ ការប្រៀបធៀបនេះបានពន្យល់យ៉ាងច្បាស់អំពីមូលហេតុដែលក្បួនដោះស្រាយសាមញ្ញ និងស្រាលដូចជា **Stochastic Gradient Descent (SGD)** នៅតែជាកម្លាំងស្នូលដ៏រឹងមាំមិនអាចខ្វះបានក្នុងការពង្រីកមាត្រដ្ឋានម៉ូដែលធំៗទំនើបៗ ដូចជា Large Language Models (LLMs)។

---

## ២. គោលគំនិតសំខាន់ៗ & និយមន័យ (Key Concepts & Definitions)

* **ការធ្វើចំណាត់ថ្នាក់ទ្វេភាគ (Binary Classification)**: កិច្ចការរៀនដោយមានការណែនាំដែលលំហស្លាកសម្គាល់គោលដៅគឺជាតម្លៃដាច់ដោយឡែកកំណត់ត្រឹមជម្រើសពីរ តាមអនុសញ្ញាកំណត់ដោយ $y^{(i)} \in \{0, 1\}$ (ឧ. សំបុត្រ Spam vs. Non-Spam ឬ រូបឆ្មា vs. មិនមែនឆ្មា)។
* **ការសន្មតកំហុសរំខានគ្មានលំអៀង (Unbiased Noise Assumption)**: ការសន្មតថាតម្លៃរំពឹងទុកនៃកំហុសរំខាន $\epsilon^{(i)}$ ស្មើនឹងសូន្យ ($E[\epsilon^{(i)}] = 0$) ដែលមានន័យថាមិនមានលំអៀងជាប្រព័ន្ធក្នុងដំណើរការបង្កើតទិន្នន័យ ឬការវាស់វែងឡើយ។
* **ឯករាជ្យ និងបំណែងចែកដូចគ្នា (Independent and Identically Distributed - IID)**: ការសន្មតគំរូស្នូលដែលទិន្នន័យ ឬកំហុសរំខាននីមួយៗត្រូវបានទាញចេញដោយឯករាជ្យពីបំណែងចែកប្រូបាប៊ីលីតេដូចគ្នា។ ការសន្មតនេះអនុញ្ញាតឱ្យប្រូបាប៊ីលីតេរួមនៃសំណុំទិន្នន័យទាំងមូល អាចបំបែកជាផលគុណនៃប្រូបាប៊ីលីតេទិន្នន័យនីមួយៗបានយ៉ាងងាយស្រួល។
* **អនុគមន៍លទ្ធភាព (Likelihood, $L(\theta)$)**: ប្រូបាប៊ីលីតេនៃសំណុំទិន្នន័យដែលបានសង្កេត គិតជាអនុគមន៍នៃវ៉ិចទ័រប៉ារ៉ាម៉ែត្រ $\theta$ ដោយរក្សាទិន្នន័យសង្កេតទុកនៅថេរ។
* **អនុគមន៍ Log-Likelihood ($l(\theta)$)**: អនុគមន៍លោការីតធម្មជាតិ (Natural Logarithm) នៃ Likelihood។ វាបំប្លែងផលគុណឱ្យទៅជាផលបូក ការពារបញ្ហាលេខតូចពេក (Numerical Underflow) និងជួយសម្រួលដល់ការគណនាដេរីវេ។
* **ការប៉ាន់ស្មានលទ្ធភាពអតិបរមា (Maximum Likelihood Estimation - MLE)**: គោលការណ៍ស្ថិតិដែលយើងជ្រើសរើសប៉ារ៉ាម៉ែត្រ $\theta$ ណាដែលផ្តល់នូវតម្លៃប្រូបាប៊ីលីតេអតិបរមា (Maximize Likelihood / Log-Likelihood) សម្រាប់ទិន្នន័យហ្វឹកហាត់ដែលយើងបានសង្កេតឃើញ។
* **អនុគមន៍ភ្ជាប់ (Link Function)**: អនុគមន៍ម៉ូណូតូន $g$ (Monotonic Function) ដែលផ្គូផ្គងតម្លៃទស្សន៍ទាយលីនេអ៊ែរ ($\theta^T x$) ទៅកាន់លំហធាតុចេញដែលមិនមែនលីនេអ៊ែរ និងមានព្រំដែនកំណត់ (ឧ. ចន្លោះ $(0, 1)$ សម្រាប់ប្រូបាប៊ីលីតេ)។
* **អនុគមន៍ Sigmoid / Logistic Function**: អនុគមន៍ភ្ជាប់ជាក់លាក់មួយកំណត់ដោយ $g(z) = \frac{1}{1 + e^{-z}}$ ដែលបម្លែងតម្លៃពិត $z \in \mathbb{R}$ ណាមួយឱ្យទៅជាតម្លៃប្រូបាប៊ីលីតេក្នុងចន្លោះ $(0, 1)$ យ៉ាងរលូន។
* **ម៉ាទ្រីស Hessian ($H$)**: ម៉ាទ្រីសការេវិមាត្រ $(d+1) \times (d+1)$ ដែលផ្ទុកនូវដេរីវេដោយផ្នែកលំដាប់ទីពីរនៃមុខងារថ្លៃដើម (ឬ Log-Likelihood) ដែលឆ្លុះបញ្ចាំងពីកម្រិតកោងក្នុងលំហ (Curvature) នៃទេសភាពបង្កើនប្រសិទ្ធភាព។
* **ម៉ាទ្រីសច្រាសក្លែងក្លាយ Moore-Penrose ($H^{\dagger}$)**: ទម្រង់ទូទៅនៃម៉ាទ្រីសច្រាសដែលប្រើក្នុង Newton's Method នៅពេលដែល Hessian មិនអាចច្រាសបាន (Singular) ដោយជួយដោះស្រាយអនុលំហប៉ារ៉ាម៉ែត្រមិនកំណត់បានយ៉ាងរលូន។

---

## ៣. រូបមន្តគណិតវិទ្យា និងការទាញសមីការ (Mathematical Formulations & Derivations)

### ក. មូលដ្ឋានគ្រឹះយុត្តិកម្មតាមបែបប្រូបាប៊ីលីតេនៃ Least Squares (MLE)

យើងសន្មតថាស្លាកសម្គាល់គោលដៅត្រូវបានបង្កើតឡើងដោយម៉ូដែលលីនេអ៊ែរពិតបូកនឹងកំហុសរំខាន Gaussian៖
$$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$$
ដែលកំហុសរំខាន $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ មានលក្ខណៈ IID។

នេះមានន័យថា បំណែងចែកមានលក្ខខណ្ឌនៃ $y^{(i)}$ ធៀបនឹងធាតុចូល $x^{(i)}$ និងប៉ារ៉ាម៉ែត្រ $\theta$ មានបំណែងចែកធម្មតាជុំវិញការទស្សន៍ទាយលីនេអ៊ែរ៖
$$y^{(i)} \mid x^{(i)}; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$$

អនុគមន៍ដង់ស៊ីតេប្រូបាប៊ីលីតេ (PDF) សម្រាប់ឧទាហរណ៍ហ្វឹកហាត់មួយគឺ៖
$$p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$$

ក្រោមការសន្មត IID អនុគមន៍ Likelihood នៃសំណុំទិន្នន័យទាំងមូលចំនួន $n$ គឺជាផលគុណនៃប្រូបាប៊ីលីតេនីមួយៗ៖
$$L(\theta) = \prod_{i=1}^n p(y^{(i)} \mid x^{(i)}; \theta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$$

ដើម្បីស្វែងរក $\theta$ ល្អបំផុត យើងយកលោការីតធម្មជាតិដើម្បីទទួលបាន Log-Likelihood $l(\theta)$៖
$$l(\theta) = \log L(\theta) = \sum_{i=1}^n \log \left[ \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) \right]$$
$$l(\theta) = n \log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2} \sum_{i=1}^n \left( y^{(i)} - \theta^T x^{(i)} \right)^2$$

ដើម្បីធ្វើអតិបរមាកម្មលើ $l(\theta)$ ធៀបនឹង $\theta$ តួថេរទីមួយមិនមានឥទ្ធិពលទេ ហើយតួទីពីរមានសញ្ញាដកនៅពីមុខផលបូកការេនៃកំហុស។ ដូច្នេះ ការធ្វើអតិបរមាកម្មលើ Log-Likelihood គឺស្មើនឹងការកាត់បន្ថយផលបូកការេនៃកំហុសឱ្យនៅតូចបំផុត៖
$$\arg\max_\theta l(\theta) = \arg\min_\theta \frac{1}{2} \sum_{i=1}^n \left( y^{(i)} - \theta^T x^{(i)} \right)^2$$

កន្សោមនេះគឺពិតជាមុខងារថ្លៃដើម Least Squares $J(\theta)$ ពីមេរៀនទី ២។ **នេះជាភស្តុតាងបង្ហាញថា Least Squares គឺជា Maximum Likelihood Estimator (MLE) ក្រោមការសន្មតកំហុសរំខាន IID Gaussian**។

### ខ. ការវិភាគតម្រែតម្រង់ឡូជីស្ទីក (Sigmoid Link Function & MLE)

សម្រាប់ការធ្វើចំណាត់ថ្នាក់ទ្វេភាគដែល $y^{(i)} \in \{0, 1\}$ យើងកំណត់ម៉ូដែលប្រូបាប៊ីលីតេនៃថ្នាក់នីមួយៗដោយយកលទ្ធផលលីនេអ៊ែរទៅឆ្លងកាត់អនុគមន៍ Sigmoid៖
$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

ប្រូបាប៊ីលីតេនៃថ្នាក់នីមួយៗត្រូវបានកំណត់ដោយ៖
$$P(y = 1 \mid x; \theta) = h_\theta(x)$$
$$P(y = 0 \mid x; \theta) = 1 - h_\theta(x)$$

យើងអាចសរសេរប្រូបាប៊ីលីតេនេះរួមគ្នាក្នុងទម្រង់បំណែងចែក Bernoulli យ៉ាងខ្លី៖
$$p(y \mid x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$$

សម្រាប់សំណុំទិន្នន័យចំនួន $n$ ដែលជា IID អនុគមន៍ Likelihood គឺ៖
$$L(\theta) = \prod_{i=1}^n (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}}$$

យកលោការីតធម្មជាតិនាំឱ្យយើងទទួលបាន Log-Likelihood $l(\theta)$៖
$$l(\theta) = \sum_{i=1}^n \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

### ការទាញដេរីវេនៃ Log-Likelihood (Deriving the Gradient)

ដើម្បីធ្វើអតិបរមាកម្មលើ $l(\theta)$ យើងត្រូវរកដេរីវេធៀបនឹង $\theta_j$។
ជាដំបូង យើងកត់សម្គាល់លក្ខណៈសម្បត្តិដេរីវេពិសេសនៃអនុគមន៍ Sigmoid៖
$$g'(z) = \frac{d}{dz} \left( \frac{1}{1 + e^{-z}} \right) = \frac{e^{-z}}{(1 + e^{-z})^2} = g(z)(1 - g(z))$$

ដោយប្រើវិធានច្រវាក់ (Chain Rule) ដេរីវេដោយផ្នែកសម្រាប់ឧទាហរណ៍មួយ $(x, y)$ គឺ៖
$$\frac{\partial}{\partial \theta_j} l(\theta) = \left( \frac{y}{h_\theta(x)} - \frac{1-y}{1-h_\theta(x)} \right) \cdot \frac{\partial}{\partial \theta_j} h_\theta(x)$$
$$\frac{\partial}{\partial \theta_j} l(\theta) = \left( \frac{y(1-h_\theta(x)) - (1-y)h_\theta(x)}{h_\theta(x)(1-h_\theta(x))} \right) \cdot h_\theta(x)(1-h_\theta(x)) \cdot x_j$$
$$\frac{\partial}{\partial \theta_j} l(\theta) = (y - h_\theta(x))x_j$$

ធ្វើផលបូកលើសំណុំទិន្នន័យទាំងមូល នាំឱ្យយើងទទួលបានជម្រាលពេញលេញនៃ Log-Likelihood៖
$$\nabla_\theta l(\theta) = \sum_{i=1}^n \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}$$

លទ្ធផលនេះបង្ហាញនូវភាពស្រស់ស្អាតនៃគណិតវិទ្យា៖ **ជម្រាលសម្រាប់ Logistic Regression មានទម្រង់កែកំហុសដូចគ្នាបេះបិទ $(y - h(x))$ ទៅនឹងជម្រាលក្នុង Linear Regression!**

### ភាពផតនៃ Logistic Log-Likelihood (Concavity)

អនុគមន៍ Log-Likelihood $l(\theta)$ គឺជាអនុគមន៍ **ផត (Concave)**៖ ម៉ាទ្រីស Hessian របស់វាគឺ៖
$$H = -\sum_{i=1}^n h_\theta(x^{(i)})\left(1 - h_\theta(x^{(i)})\right) x^{(i)} (x^{(i)})^T$$
ដែលជាម៉ាទ្រីសអវិជ្ជមានពាក់កណ្តាលកំនត់ (Negative Semidefinite) ពីព្រោះ $h(1-h) > 0$ ចំពោះគ្រប់ធាតុចូល ហើយ $v^T x x^T v = (x^T v)^2 \ge 0$។ ជាលទ្ធផល ទាំង Gradient Ascent និង Newton's Method ធានាថានឹងរួមគ្នាមកកាន់ **ចំណុចអតិបរមាសកល (Global Maximum)** ដោយគ្មានបញ្ហាជាប់គាំងក្នុង Local Optima ឡើយ។

---

## ៤. ក្បួនដោះស្រាយបង្កើនប្រសិទ្ធភាពជាជំហានៗ (Step-by-Step Optimization Algorithms)

### ក. Gradient Ascent សម្រាប់ Logistic Regression

ដោយសារយើងចង់ **ធ្វើអតិបរមាកម្ម** លើ Log-Likelihood $l(\theta)$ (ជំនួសឱ្យការកាត់បន្ថយ Loss) យើងប្រើ **Gradient Ascent** (សញ្ញាបូកជំនួសសញ្ញាដក)៖

1. **កំណត់តម្លៃដំបូង (Initialize)**៖ កំណត់វ៉ិចទ័រប៉ារ៉ាម៉ែត្រ $\theta$ (ជាទូទៅកំណត់ដោយចៃដន្យ ឬស្មើ ០)។
2. **រង្វិលជុំរហូតដល់រួមគ្នា (Loop until convergence)**៖
   - **Batch Gradient Ascent**:
     $$\theta := \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}$$
   - **Stochastic Gradient Ascent** (ធ្វើបច្ចុប្បន្នភាពលើទិន្នន័យនីមួយៗ $i$):
     $$\theta := \theta + \alpha \left( y^{(i)} - h_{\theta^{(t)}}(x^{(i)}) \right) x^{(i)}$$
     ដែល $\alpha$ គឺជាអត្រារៀន (Learning Rate)។

### ឧទាហរណ៍គំរូតូចជាក់ស្តែង (Worked Micro-Example)

ឧបមាថាយើងមានចំណុចទិន្នន័យមួយ $(x, y) = (2, 1)$ ដោយកំណត់ $x_0 = 1$ នាំឱ្យ $x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ ហើយចាប់ផ្តើមពី $\theta = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ ជាមួយ $\alpha = 0.1$៖

1. **ទស្សន៍ទាយ (Predict)**៖ $\theta^T x = 0$ នាំឱ្យ $h_\theta(x) = g(0) = \frac{1}{1 + e^{0}} = 0.5$។
2. **កំហុសលំអៀង (Error)**៖ $y - h_\theta(x) = 1 - 0.5 = 0.5$ (ម៉ូដែលមិនទាន់ច្បាស់ ខណៈដែលស្លាកពិតគឺវិជ្ជមាន)។
3. **ធ្វើបច្ចុប្បន្នភាព (Update)**៖
   $$\theta := \begin{bmatrix} 0 \\ 0 \end{bmatrix} + 0.1 \times 0.5 \times \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 0.05 \\ 0.1 \end{bmatrix}$$
4. **ប្រសិទ្ធភាព (Effect)**៖ ការទស្សន៍ទាយថ្មីគឺ $\theta^T x = 0.05 + 0.2 = 0.25$ នាំឱ្យ $h_\theta(x) = g(0.25) \approx 0.562$ — ប្រូបាប៊ីលីតេរំកិល *ខិតជិត* ទៅកាន់ស្លាកពិត (១)។ ការធ្វើជំហានដដែលៗនឹងរុញច្រាន $h_\theta(x) \to 1$ សម្រាប់ចំណុចនេះ។

<div id="plotly-logistic-boundary" class="plotly-chart" aria-label="Interactive Plotly chart: Logistic Regression decision boundary with sigmoid probability gradient"></div>

<p><em>រូបភាព៖ ព្រំដែនសម្រេចចិត្តនៃ Logistic Regression — ប្លង់លក្ខណៈពិសេស 2D ជាមួយឧទាហរណ៍វិជ្ជមាន (រង្វង់ពណ៌ក្រហម) និងឧទាហរណ៍អវិជ្ជមាន (ត្រីកោណពណ៌ខៀវ)។ ខ្សែបន្ទាត់ដាច់ៗគឺជាព្រំដែនសម្រេចចិត្តលីនេអ៊ែរ $\theta^T x = 0$ ហើយព្រួញបង្ហាញពីទិសដៅនៃការកើនឡើងប្រូបាប៊ីលីតេ ដែលលទ្ធផល sigmoid $h_\theta(x)$ ប្រែប្រួលយ៉ាងរលូនពី ០ ទៅ ១។</em></p>

---

### ខ. វិធីសាស្ត្រញូតុន (Newton's Method - Second-Order Optimization)

វិធីសាស្ត្រញូតុន គឺជាបច្ចេកទេសបង្កើនប្រសិទ្ធភាពលំដាប់ទីពីរ (Second-order optimization) ដែលបង្កើតគំរូប៉ាន់ស្មានដឺក្រេទីពីរ (Quadratic approximation) ក្នុងកម្រិតមូលដ្ឋាន ដើម្បីស្វែងរកទីតាំងដែលដេរីវេស្មើនឹងសូន្យ។

#### ១. ការស្វែងរកឫសក្នុងអថេរទោល (1D Root-Finding)

ដើម្បីស្វែងរកឫសនៃអនុគមន៍ $f(\theta) = 0$៖
$$\theta^{(t+1)} := \theta^{(t)} - \frac{f(\theta^{(t)})}{f'(\theta^{(t)})}$$

#### ២. ការធ្វើអតិបរមាកម្មអថេរទោល (Univariate Maximization)

ដើម្បីធ្វើអតិបរមាកម្មលើ Log-Likelihood $l(\theta)$ យើងចង់ស្វែងរកឫសនៃដេរីវេទីមួយ $l'(\theta) = 0$។ ជំនួស $f(\theta) = l'(\theta)$ នាំឱ្យទទួលបាន៖
$$\theta^{(t+1)} := \theta^{(t)} - \frac{l'(\theta^{(t)})}{l''(\theta^{(t)})}$$

#### ៣. ករណីពហុអថេរ (Multivariate Maximization)

នៅក្នុងវិមាត្រខ្ពស់ដែល $\theta \in \mathbb{R}^{d+1}$ ដេរីវេស្កាលែត្រូវបានជំនួសដោយវ៉ិចទ័រជម្រាល $\nabla_\theta l(\theta)$ និងម៉ាទ្រីស Hessian $H$៖
$$\theta^{(t+1)} := \theta^{(t)} - H^{-1} \nabla_\theta l(\theta^{(t)})$$
ដែល $H \in \mathbb{R}^{(d+1) \times (d+1)}$ គឺជាម៉ាទ្រីស Hessian ដែលមានធាតុនីមួយៗកំណត់ដោយ៖
$$H_{jk} = \frac{\partial^2 l(\theta)}{\partial \theta_j \partial \theta_k}$$

<div id="plotly-optimization-compare" class="plotly-chart" aria-label="Interactive Plotly chart: Gradient Descent versus Newton's Method optimization paths"></div>

<p><em>រូបភាព៖ ការប្រៀបធៀបធរណីមាត្រនៃជំហានបង្កើនប្រសិទ្ធភាព — Gradient Descent បោះជំហានតូចៗកាត់កែងទៅនឹងខ្សែកោងកម្រិត ខណៈដែល Newton's Method បង្កើតគំរូប៉ារ៉ាបូលដឺក្រេទីពីរទៅនឹងកម្រិតកោង ហើយលោតឆ្ពោះទៅកាន់ចំណុចអប្បបរមាក្នុងជំហានដ៏តិចបំផុត។</em></p>

---

## ៥. តុល្យភាពជាក់ស្តែង ប្រព័ន្ធ និងការបង្កើនប្រសិទ្ធភាព (Practical, System, & Optimization Trade-offs)

ការជ្រើសរើសក្បួនដោះស្រាយបង្កើនប្រសិទ្ធភាពក្នុង ML ទំនើប គឺផ្អែកលើតុល្យភាពរវាង **ល្បឿនរួមគ្នាតាមបែបស្ថិតិ** (ចំនួនជំហានដើម្បីរកចំណុចល្អបំផុត) និង **ភាពស្មុគស្មាញនៃការគណនាក្នុងមួយជំហាន** (Flops និងអង្គចងចាំ Memory)៖

| លក្ខណៈពិសេស / ក្បួនដោះស្រាយ | Stochastic Gradient Descent (SGD) | វិធីសាស្ត្រញូតុន (Newton's Method) |
| :--- | :--- | :--- |
| **ភាពស្មុគស្មាញពេលវេលាក្នុងមួយជំហាន** | $O(d)$ ក្នុងមួយជំហាន (ឬ $O(Bd)$ សម្រាប់ Minibatch) | $O(n d^2 + d^3)$ ក្នុងមួយជំហាន |
| **ទំហំអង្គចងចាំ (Memory)** | $O(d)$ | $O(d^2)$ ដើម្បីផ្ទុកម៉ាទ្រីស Hessian |
| **អត្រានៃការរួមគ្នា (Convergence)** | Linear / Sublinear (ទាមទារជំហានច្រើន) | Quadratic (រួមគ្នាស្ទើរតែភ្លាមៗក្នុងជំហានតិចតួច) |
| **ការលៃតម្រូវ Step Size ($\alpha$)** | ងាយរងប្រតិកម្មខ្លាំង; ទាមទារការកំណត់កាលវិភាគល្អិតល្អន់ | មិនត្រូវការប៉ារ៉ាម៉ែត្រ Step Size ឡើយ; បោះជំហានតាមគណិតវិទ្យាល្អបំផុត |
| **ព័ត៌មានលំដាប់ទីពីរ (Curvature)** | គ្មាន — ប្រើតែដេរីវេទីមួយ (Gradient) | ប្រើកម្រិតកោងពេញលេញតាមរយៈ Hessian និងម៉ាទ្រីសច្រាស |
| **ការពង្រីកលើវិមាត្រខ្ពស់ ($d \gg 10^5$)** | ពង្រីកបានយ៉ាងល្អអស្ចារ្យ; ជាស្តង់ដារសម្រាប់ម៉ូដែលរាប់ពាន់លានប៉ារ៉ាម៉ែត្រ | មិនអាចអនុវត្តបានទាល់តែសោះ ដោយសារថ្លៃដើមបញ្ច្រាសម៉ាទ្រីស $d^3$ |
| **ករណីប្រើប្រាស់ចម្បង** | Large-scale deep learning, LLMs និងទិន្នន័យកម្រិតអ៊ីនធឺណិត | សំណុំទិន្នន័យខ្នាតតូច ស្ថិតិបុរាណ និងវិទ្យាសាស្ត្រសង្គម |

---

## ៦. Weighted Least Squares & Iteratively Reweighted Least Squares (IRLS)

ចំណងជើងនៃមេរៀនបានលើកឡើងអំពី "Weighted Least Squares" ដែលផ្សារភ្ជាប់មេរៀននេះត្រឡប់ទៅមេរៀនទី ២ និងឆ្ពោះទៅកាន់ Newton's Method។

### ក. Weighted Least Squares (WLS)

ប្រសិនបើកម្រិតប្រែប្រួលនៃកំហុសរំខានមានភាពខុសៗគ្នាក្នុងទិន្នន័យនីមួយៗ (Heteroscedasticity) $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma_i^2)$ នោះការទាញ MLE នឹងនៅរក្សាភាពត្រឹមត្រូវដដែល លើកលែងតែកំហុសលើកជាការេនីមួយៗត្រូវបានថ្លឹងទម្ងន់ដោយកម្រិតសុក្រឹតរបស់វា៖
$$J(\theta) = \frac{1}{2} \sum_{i=1}^n w^{(i)} \left( y^{(i)} - \theta^T x^{(i)} \right)^2, \quad w^{(i)} = \frac{1}{\sigma_i^2}$$

*ការយល់ដឹងវិចារណញាណ*៖ ទិន្នន័យណាដែលមានកំហុសរំខានធំ ($\sigma_i$ ធំ) ផ្តល់ព័ត៌មានមិនសូវច្បាស់លាស់ ដូច្នេះវាទទួលបាន **ទម្ងន់តូចជាង** ក្នុងការហ្វឹកហាត់។ ដំណោះស្រាយទម្រង់បិទក្លាយជា Weighted Normal Equations: $\theta = (X^T W X)^{-1} X^T W y$ ដែល $W = \mathrm{diag}(w^{(1)}, \dots, w^{(n)})$។

### ខ. IRLS: វិធីសាស្ត្រញូតុនលើ Logistic Regression គឺជា Reweighted Least Squares

នៅពេលយើងអនុវត្តវិធីសាស្ត្រញូតុនលើ Logistic Log-Likelihood ការធ្វើបច្ចុប្បន្នភាពនីមួយៗមានទម្រង់ដូចគ្នាបេះបិទទៅនឹងការដោះស្រាយបញ្ហា Weighted Least Squares នៅគ្រប់រង្វិលជុំ។ ជាមួយនឹងទម្ងន់៖
$$w^{(i)} = h_\theta(x^{(i)}) \left( 1 - h_\theta(x^{(i)}) \right)$$
(ដែលជាជម្រាលនៃ sigmoid នៅត្រង់ចំណុចនីមួយៗ), ម៉ាទ្រីស Hessian គឺ $H = -X^T W X$ ហើយជំហានញូតុនគឺស្មើនឹងការគណនា Linear Regression ឡើងវិញលើ "Working Targets" $z^{(i)} = \theta^T x^{(i)} + \frac{y^{(i)} - h_\theta(x^{(i)})}{w^{(i)}}$។ ហេតុនេះហើយទើបវាមានឈ្មោះថា **Iteratively Reweighted Least Squares (IRLS)**៖ ធ្វើរង្វិលជុំដដែលៗ {គណនាទម្ងន់ពីប្រូបាប៊ីលីតេបច្ចុប្បន្ន → ដោះស្រាយ Weighted Least Squares} រហូតដល់រួមគ្នា។ ចំណុចនេះបង្រួបបង្រួមខ្លឹមសារទាំងពីរនៃមេរៀន៖ វិធីសាស្ត្រញូតុនសម្រាប់ Logistic Regression គឺជា Least Squares ដែលធ្វើឡើងដដែលៗជាមួយការថ្លឹងទម្ងន់ឡើងវិញ។

---

## ៧. កម្មវិធីអនុវត្តជាក់ស្តែង (Practical Applications)

* **ស្រទាប់បញ្ចេញលទ្ធផលចំណាត់ថ្នាក់ក្នុង LLMs (Classification Output Layers)**៖ ម៉ូដែល Generative AI ទំនើបៗ (ដូចជា Claude ឬ ChatGPT) ប្រើប្រាស់ទម្រង់ពហុថ្នាក់នៃ Logistic Regression ដែលហៅថា **Softmax** ជាស្រទាប់ចុងក្រោយនៃ Neural Network។ លក្ខណៈពិសេសលីនេអ៊ែរចុងក្រោយ (Logits) ត្រូវបានធ្វើស្តង់ដារតាម Softmax ដើម្បីបញ្ចេញបំណែងចែកប្រូបាប៊ីលីតេលើ Tokens នៃវចនានុក្រម។
* **ការវិភាគតម្រែតម្រង់ក្នុងស្ថិតិបុរាណ (Classical Statistical Regressions)**៖ នៅក្នុងការស្រាវជ្រាវជីវវេជ្ជសាស្ត្រ ឬវិទ្យាសាស្ត្រសង្គម Logistic Regression ត្រូវបានដោះស្រាយដោយប្រើ Newton's Method (ឬក្បួន Quasi-Newton ដូចជា L-BFGS)។ ដោយសារសំណុំទិន្នន័យមានចំនួន Features តូច ($d < 100$) វិធីសាស្ត្រញូតុនត្រូវបានពេញចិត្តបំផុត ពីព្រោះវារួមគ្នាភ្លាមៗដោយមិនបាច់ចំណាយពេលលៃតម្រូវ Learning Rate ឡើយ។

---

## ៨. សំណួរត្រិះរិះពិចារណា (Reflection Questions)

១. **ភាពរងប្រតិកម្មចំពោះ Outliers ក្នុង Least Squares**៖ តាមបែបធរណីមាត្រ ហេតុអ្វីបានជាការយកម៉ូដែល Linear Regression ទៅប្រើលើស្លាកសម្គាល់ទ្វេភាគ ($y \in \{0, 1\}$) តាមរយៈ Least Squares បង្កើតបានព្រំដែនសម្រេចចិត្តខ្សោយ នៅពេលមានចំណុចទិន្នន័យ Outlier ស្ថិតនៅឆ្ងាយខ្លាំង ប៉ុន្តែងាយស្រួលចាត់ថ្នាក់?  
   *ចម្លើយសង្ខេប*៖ Least Squares ដាក់ទណ្ឌកម្មជា *ការេ* លើកំហុសលំអៀង។ ចំណុចដែលនៅឆ្ងាយខ្លាំង (ទោះបីជាចាត់ថ្នាក់ត្រូវរួចហើយក្តី) នឹងបង្កើតតម្លៃសំណល់ធំ ដែលបង្ខំឱ្យបន្ទាត់តម្រែតម្រង់ត្រូវតែផ្អៀងងាកទៅរកវា ដើម្បីបង្រួមកំហុសការេនោះ។ ជាលទ្ធផល ព្រំដែនសម្រេចចិត្តរងការរំកិល និងធ្វើឱ្យចំណុចទិន្នន័យនៅក្បែរព្រំដែនត្រូវចាត់ថ្នាក់ខុស។ ផ្ទុយទៅវិញ Logistic Regression មិនរងផលប៉ះពាល់នេះទេ ពីព្រោះ Sigmoid ឆ្អែត (Saturates) ត្រឹម ១ ដែលធ្វើឱ្យជម្រាលកែកំហុស $(y - h(x))x \approx 0$។  
២. **អត្ថន័យនៃគំរូកំហុសរំខាន (Noise Model Implications)**៖ យើងបានបង្ហាញថាកំហុស IID Gaussian ($\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$) នាំឱ្យកើតមាន Least Squares Cost Function (ការេនៃកំហុស)។ ប្រសិនបើយើងសន្មតថាកំហុសរំខានដើរតាមបំណែងចែក Laplace (Double-exponential) វិញ តើគោលដៅ MLE នឹងផ្លាស់ប្តូរយ៉ាងដូចម្តេច?  
   *ចម្លើយសង្ខេប*៖ អនុគមន៍ដង់ស៊ីតេ Laplace មានទម្រង់ $p(\epsilon) \propto \exp(-|\epsilon|/b)$។ តាមការទាញ MLE វានឹងនាំឱ្យយើងធ្វើអប្បបរមាកម្មលើ **ផលបូកនៃតម្លៃដាច់ខាតនៃកំហុស** (Least Absolute Deviations - LAD) ជំនួសឱ្យផលបូកការេ។ LAD ធន់នឹង Outliers ជាងមុនឆ្ងាយ ប៉ុន្តែមិនអាចធ្វើដេរីវេត្រង់ ០ បានឡើយ។  
៣. **បញ្ហារាំងស្ទះនៃម៉ាទ្រីស Hessian (The Hessian Bottleneck)**៖ ហេតុអ្វីបានជា Newton's Method មិនអាចពង្រីកមាត្រដ្ឋានក្នុងការហ្វឹកហាត់ Large Language Models (LLMs) បាន? ចូរចង្អុលបង្ហាញតួគណិតវិទ្យាជាក់លាក់ក្នុងភាពស្មុគស្មាញនៃជំហានដែលជាឧបសគ្គផ្នែករឹង និងពន្យល់ពីរបៀបដែល Optimizers ទំនើប (ដូចជា Adam) ប៉ាន់ស្មានព័ត៌មានកម្រិតកោងនេះដោយចំណាយធនធានតិច។

---

## ៩. ឯកសារយោង & ការអានបន្ថែម (Further Reading & Resources)

* **Generalized Linear Models (GLMs)**៖ ក្របខ័ណ្ឌគណិតវិទ្យាទូលំទូលាយដែលបង្រួបបង្រួម Linear Regression, Logistic Regression, និងគំរូរាប់ (Count-based models) ដែលនឹងត្រូវបង្រៀនក្នុង Lecture 4។
* **Friday TA Sessions**៖ វគ្គសិក្ខាសាលាសម្រាប់ពិនិត្យមើលឡើងវិញនូវការពន្លាតពហុវិមាត្រ Taylor Expansions, ការរកជម្រាលនៃអនុគមន៍ម៉ាទ្រីស-វ៉ិចទ័រ និងលក្ខណៈសម្បត្តិនៃម៉ាទ្រីស Positive Semidefinite (PSD)។
* **Curvature Approximations in Deep Learning**៖ ឯកសារស្រាវជ្រាវស្តីពីវិធីសាស្ត្រ Quasi-Newton (L-BFGS), ការប៉ាន់ស្មានអង្កត់ទ្រូងនៃ Hessian (AdaGrad), និងរបៀបដែល Optimizers ទំនើបក្នុង Deep Learning រួមបញ្ចូលគ្នារវាង Momentum ជាមួយ Adaptive Learning Rates (Adam)។
