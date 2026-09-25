export interface LectureItem {
  id: string;
  slug: string;
  num: number;
  courseCode: 'cs229' | 'cs336' | 'cs231n';
  courseName: string;
  courseColor: string;
  title: string;
  shortTitle: string;
  url: string;
}

export interface CourseInfo {
  code: 'cs229' | 'cs336' | 'cs231n';
  name: string;
  subtitle: string;
  instructors: string;
  term: string;
  color: string;
  thumbnail: string;
  badge: string;
  summary: string;
  topics: string[];
}

export const coursesInfo: Record<'cs229' | 'cs336' | 'cs231n', CourseInfo> = {
  cs229: {
    code: 'cs229',
    name: 'CS229: Machine Learning',
    subtitle: 'Foundations and Contemporary Paradigms of Machine Learning',
    instructors: 'Prof. Christopher Ré & Tengyu Ma',
    term: 'Spring 2026 • Stanford Computer Science',
    color: '#8C1515',
    thumbnail: 'cs229.jpg',
    badge: 'Flagship Track',
    summary: 'The canonical foundational machine learning curriculum at Stanford. Covers supervised learning (regression, classification, GLMs), deep learning backpropagation, generalization dynamics, unsupervised learning (EM, GMMs, PCA, ICA), reinforcement learning, and modern generative paradigms (Diffusion Models, LoRA fine-tuning).',
    topics: ['Supervised Learning', 'GLMs & Logistic Regression', 'SVMs & Kernels', 'Backprop Dynamics', 'EM & Clustering', 'PCA & SVD', 'Reinforcement Learning', 'Diffusion & LoRA']
  },
  cs336: {
    code: 'cs336',
    name: 'CS336: Language Modeling from Scratch',
    subtitle: 'Architecture, Systems, Distributed Training & Alignment',
    instructors: 'Prof. Percy Liang & Tatsunori Hashimoto',
    term: 'Spring 2026 • Stanford Computer Science',
    color: '#10B981',
    thumbnail: 'cs336.jpg',
    badge: 'Frontier LLMs',
    summary: 'A systems-first, deep-dive curriculum on modern foundation models. Implement and scale transformers from scratch: tokenization (BPE), self-attention, RoPE embeddings, data filtering, distributed training (DDP, ZeRO, FSDP, Pipeline Parallelism), mixed-precision FP8/BF16, FlashAttention, KV-caching, RLHF/DPO alignment, and agent tool use.',
    topics: ['Transformer Core', 'Byte-Pair Encoding', 'Distributed (ZeRO/FSDP)', 'FlashAttention & KV-Cache', 'Quantization & Speculative', 'Supervised Fine-Tuning', 'RLHF & DPO', 'Reasoning & Agents']
  },
  cs231n: {
    code: 'cs231n',
    name: 'CS231N: Deep Learning for Computer Vision',
    subtitle: 'Visual Perception, 3D Reconstruction & Spatial Intelligence',
    instructors: 'Prof. Fei-Fei Li & Ranjay Krishna',
    term: 'Spring 2026 • Stanford Computer Science',
    color: '#3B82F6',
    thumbnail: 'cs231n.jpg',
    badge: '3D Vision & NeRF',
    summary: 'Stanford\'s flagship visual perception curriculum. From biological image formation and classical classification pipelines to deep convolutional networks, residual backbones, Vision Transformers (ViT), object detection, panoptic segmentation, 3D representation, Neural Radiance Fields (NeRF), and 3D Gaussian Splatting.',
    topics: ['Image Classification', 'Losses & Backpropagation', 'Convolutional Networks', 'ResNet & ConvNeXt', 'Vision Transformers', 'Object Detection & Masking', '3D Vision & Geometry', 'NeRF & Gaussian Splats']
  }
};

const rawCs229 = [
  { id: 'lecture-1', num: 1, full: 'Lecture 1: Intro, Linear Regression & Gradient Descent', short: 'Intro, Linear Regression & Gradient Descent' },
  { id: 'lecture-2', num: 2, full: 'Lecture 2: Normal Equations, Locally Weighted Regression', short: 'Normal Equations, Locally Weighted Regression' },
  { id: 'lecture-3', num: 3, full: 'Lecture 3: Logistic Regression & Newton-Raphson Method', short: 'Logistic Regression & Newton-Raphson Method' },
  { id: 'lecture-4', num: 4, full: 'Lecture 4: Generalized Linear Models & Softmax Regression', short: 'Generalized Linear Models & Softmax Regression' },
  { id: 'lecture-5', num: 5, full: 'Lecture 5: Gaussian Discriminant Analysis & Naive Bayes', short: 'Gaussian Discriminant Analysis & Naive Bayes' },
  { id: 'lecture-6', num: 6, full: 'Lecture 6: Support Vector Machines & Kernel Methods', short: 'Support Vector Machines & Kernel Methods' },
  { id: 'lecture-7', num: 7, full: 'Lecture 7: Deep Learning, Backpropagation & Architecture', short: 'Deep Learning, Backpropagation & Architecture' },
  { id: 'lecture-8', num: 8, full: 'Lecture 8: Generalization, Regularization & Double Descent', short: 'Generalization, Regularization & Double Descent' },
  { id: 'lecture-9', num: 9, full: 'Lecture 9: Unsupervised Learning, K-Means & GMMs', short: 'Unsupervised Learning, K-Means & GMMs' },
  { id: 'lecture-10', num: 10, full: 'Lecture 10: Expectation-Maximization (EM) Algorithm', short: 'Expectation-Maximization (EM) Algorithm' },
  { id: 'lecture-11', num: 11, full: 'Lecture 11: Principal Component Analysis & SVD', short: 'Principal Component Analysis & SVD' },
  { id: 'lecture-12', num: 12, full: 'Lecture 12: Independent Component Analysis & Self-Supervision', short: 'Independent Component Analysis & Self-Supervision' },
  { id: 'lecture-13', num: 13, full: 'Lecture 13: Reinforcement Learning & MDPs', short: 'Reinforcement Learning & MDPs' },
  { id: 'lecture-14', num: 14, full: 'Lecture 14: Contemporary Paradigms (Diffusion, LoRA)', short: 'Contemporary Paradigms (Diffusion, LoRA)' },
];

const rawCs336 = [
  { id: 'lecture-1', num: 1, full: 'Lecture 1: Introduction to LLMs & Scaling Laws', short: 'Introduction to LLMs & Scaling Laws' },
  { id: 'lecture-2', num: 2, full: 'Lecture 2: Tokenization & Byte-Pair Encoding (BPE)', short: 'Tokenization & Byte-Pair Encoding (BPE)' },
  { id: 'lecture-3', num: 3, full: 'Lecture 3: Transformer Architecture from Scratch', short: 'Transformer Architecture from Scratch' },
  { id: 'lecture-4', num: 4, full: 'Lecture 4: Self-Attention & RoPE Positional Embeddings', short: 'Self-Attention & RoPE Positional Embeddings' },
  { id: 'lecture-5', num: 5, full: 'Lecture 5: Pre-training Data Curation & Filtering', short: 'Pre-training Data Curation & Filtering' },
  { id: 'lecture-6', num: 6, full: 'Lecture 6: Distributed Training (DDP, ZeRO, FSDP)', short: 'Distributed Training (DDP, ZeRO, FSDP)' },
  { id: 'lecture-7', num: 7, full: 'Lecture 7: Tensor & Pipeline Parallelism', short: 'Tensor & Pipeline Parallelism' },
  { id: 'lecture-8', num: 8, full: 'Lecture 8: Mixed-Precision Training (FP16, BF16, FP8)', short: 'Mixed-Precision Training (FP16, BF16, FP8)' },
  { id: 'lecture-9', num: 9, full: 'Lecture 9: Optimization Dynamics & Learning Rates', short: 'Optimization Dynamics & Learning Rates' },
  { id: 'lecture-10', num: 10, full: 'Lecture 10: Inference Optimization (KV-Cache, FlashAttention)', short: 'Inference Optimization (KV-Cache, FlashAttention)' },
  { id: 'lecture-11', num: 11, full: 'Lecture 11: Speculative Decoding & Quantization', short: 'Speculative Decoding & Quantization' },
  { id: 'lecture-12', num: 12, full: 'Lecture 12: Instruction Tuning & Supervised Fine-Tuning', short: 'Instruction Tuning & Supervised Fine-Tuning' },
  { id: 'lecture-13', num: 13, full: 'Lecture 13: Alignment via RLHF & PPO', short: 'Alignment via RLHF & PPO' },
  { id: 'lecture-14', num: 14, full: 'Lecture 14: Direct Preference Optimization (DPO)', short: 'Direct Preference Optimization (DPO)' },
  { id: 'lecture-15', num: 15, full: 'Lecture 15: Evaluation, Benchmarks & Leaderboards', short: 'Evaluation, Benchmarks & Leaderboards' },
  { id: 'lecture-16', num: 16, full: 'Lecture 16: Retrieval-Augmented Generation (RAG)', short: 'Retrieval-Augmented Generation (RAG)' },
  { id: 'lecture-17', num: 17, full: 'Lecture 17: Reasoning, Tool Use & Autonomous Agents', short: 'Reasoning, Tool Use & Autonomous Agents' },
  { id: 'lecture-guest', num: 99, full: 'Guest Lecture: Frontier Infrastructure at Scale', short: 'Frontier Infrastructure at Scale' },
];

const rawCs231n = [
  { id: 'lecture-1', num: 1, full: 'Lecture 1: Computer Vision Overview & History', short: 'Computer Vision Overview & History' },
  { id: 'lecture-2', num: 2, full: 'Lecture 2: Image Classification Pipeline & k-NN', short: 'Image Classification Pipeline & k-NN' },
  { id: 'lecture-3', num: 3, full: 'Lecture 3: Loss Functions & Optimization', short: 'Loss Functions & Optimization' },
  { id: 'lecture-4', num: 4, full: 'Lecture 4: Neural Networks & Backpropagation', short: 'Neural Networks & Backpropagation' },
  { id: 'lecture-5', num: 5, full: 'Lecture 5: Convolutional Neural Networks (CNNs)', short: 'Convolutional Neural Networks (CNNs)' },
  { id: 'lecture-6', num: 6, full: 'Lecture 6: Training Neural Networks (Part I)', short: 'Training Neural Networks (Part I)' },
  { id: 'lecture-7', num: 7, full: 'Lecture 7: Training Neural Networks (Part II)', short: 'Training Neural Networks (Part II)' },
  { id: 'lecture-8', num: 8, full: 'Lecture 8: Deep Learning Software & Hardware', short: 'Deep Learning Software & Hardware' },
  { id: 'lecture-9', num: 9, full: 'Lecture 9: CNN Architectures (ResNet, ConvNeXt)', short: 'CNN Architectures (ResNet, ConvNeXt)' },
  { id: 'lecture-10', num: 10, full: 'Lecture 10: Recurrent Networks & Attention', short: 'Recurrent Networks & Attention' },
  { id: 'lecture-11', num: 11, full: 'Lecture 11: Vision Transformers (ViT)', short: 'Vision Transformers (ViT)' },
  { id: 'lecture-12', num: 12, full: 'Lecture 12: Detection & Segmentation', short: 'Detection & Segmentation' },
  { id: 'lecture-13', num: 13, full: 'Lecture 13: 3D Vision & Depth Estimation', short: '3D Vision & Depth Estimation' },
  { id: 'lecture-14', num: 14, full: 'Lecture 14: Generative Models (VAEs, GANs)', short: 'Generative Models (VAEs, GANs)' },
  { id: 'lecture-15', num: 15, full: 'Lecture 15: Diffusion Models & Flow Matching', short: 'Diffusion Models & Flow Matching' },
  { id: 'lecture-16', num: 16, full: 'Lecture 16: Visual Transformers & Multimodal Models', short: 'Visual Transformers & Multimodal Models' },
  { id: 'lecture-17', num: 17, full: 'Lecture 17: Video Understanding & Action Recognition', short: 'Video Understanding & Action Recognition' },
  { id: 'lecture-18', num: 18, full: 'Lecture 18: Neural Radiance Fields (NeRF) & 3D Gaussian Splatting', short: 'Neural Radiance Fields (NeRF) & 3D Gaussian Splatting' },
];

function transformRaw(courseCode: 'cs229' | 'cs336' | 'cs231n', list: Array<{ id: string; num: number; full: string; short: string }>): LectureItem[] {
  const info = coursesInfo[courseCode];
  return list.map((item) => ({
    id: item.id,
    slug: item.id,
    num: item.num,
    courseCode,
    courseName: info.name,
    courseColor: info.color,
    title: item.full,
    shortTitle: item.short,
    url: `/courses/${courseCode}/${item.id}/`
  }));
}

export const cs229LecturesList: LectureItem[] = transformRaw('cs229', rawCs229);
export const cs336LecturesList: LectureItem[] = transformRaw('cs336', rawCs336);
export const cs231nLecturesList: LectureItem[] = transformRaw('cs231n', rawCs231n);

export const allLecturesList: LectureItem[] = [
  ...cs229LecturesList,
  ...cs336LecturesList,
  ...cs231nLecturesList
];

export function getLecturesByCourse(courseCode: string): LectureItem[] {
  if (courseCode === 'cs229') return cs229LecturesList;
  if (courseCode === 'cs336') return cs336LecturesList;
  if (courseCode === 'cs231n') return cs231nLecturesList;
  return [];
}

export function getLecture(courseCode: string, lectureSlug: string): LectureItem | undefined {
  return allLecturesList.find(
    (item) => item.courseCode === courseCode && (item.slug === lectureSlug || item.id === lectureSlug)
  );
}
