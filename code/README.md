# Few-Shot Image Classification Based on Prototypical Network

## 项目简介

本项目实现了基于度量学习（Metric-Based）的 Few-Shot Image Classification 系统，并完成课程要求的所有核心任务与扩展挑战。

主要内容包括：

* Prototypical Network（ProtoNet）
* Episodic Training Pipeline
* 5-Way Few-Shot Classification
* 1-shot / 5-shot / 10-shot 对比实验
* Fine-tuning ResNet18 基线模型
* Mixup 数据增强
* CutMix 数据增强
* t-SNE 特征空间可视化

数据集：

* CUB-200-2011 Bird Dataset

---

# 项目目录

```text
project/
│
├── code/
│   ├── README.md
│   ├── requirements.txt
│   ├── main.py
│   ├── dataset.py
│   ├── model.py
│   ├── episode.py
│   ├── train_eval.py
│   ├── finetune.py
│   ├── augment.py
│   ├── methods.py
│   ├── viz.py
│   └── comparison.py
│
├── report/
│   └── report.pdf
│
├── results/
│   ├── loss_curve.png
│   ├── tsne.png
│   ├── confusion_matrix.png
│   └── comparison_table.png
│
└── contribution.txt
```

---

# 实验环境

操作系统：

Windows 11

Python：

Python 3.9

主要依赖：

* torch
* torchvision
* numpy
* matplotlib
* scikit-learn
* pillow
* python-docx
* tqdm

---

# 安装依赖

打开终端：

```bash
pip install -r requirements.txt
```

或者：

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn python-docx tqdm
```

---

# 数据集准备

下载：

CUB-200-2011

解压后目录如下：

```text
CUB_200_2011/
│
├── images/
├── attributes/
├── parts/
├── attributes.txt
├── bounding_boxes.txt
├── classes.txt
├── READEME.md
├── image_class_labels.txt
├── images.txt
└── train_test_split.txt
```

将其放置在项目根目录：

```text
project/
│
├── CUB_200_2011/
└── code/
```

---

# 训练与测试

运行：

```bash
python main.py
```

程序将自动完成：

1. ProtoNet训练
2. Episodic Sampling
3. Baseline实验
4. Mixup实验
5. CutMix实验
6. Fine-tuning实验
7. t-SNE可视化
8. 对比表生成

---

# 输出结果

运行完成后将在 outputs 文件夹中生成：

```text
outputs/
│
├── loss_curve.png
├── tsne.png
├── comparison_table.png
└── confusion_matrix.png
```

---

# 核心任务完成情况

| 任务                | 完成情况 |
| ----------------- | ---- |
| ProtoNet实现        | √    |
| 5-Way 5-Shot分类    | √    |
| Episodic Training | √    |
| Fine-tuning对比     | √    |
| 1/5/10 Shot分析     | √    |

---

# 扩展挑战完成情况

| 挑战       | 完成情况 |
| -------- | ---- |
| Mixup    | √    |
| CutMix   | √    |
| t-SNE可视化 | √    |
| 实验结果分析   | √    |

---

# 实验结果示例

| Method   | 1-shot | 5-shot | 10-shot |
| -------- | ------ | ------ | ------- |
| Baseline | 0.434  | 0.667  | 0.760   |
| Mixup    | 0.329  | 0.452  | 0.519   |
| CutMix   | 0.441  | 0.686  | 0.766   |
| Finetune | 0.385  | 0.421  | 0.452   |

结果表明：

* ProtoNet更适合Few-Shot场景
* Shot数量增加会提升准确率
* CutMix优于Mixup
* Fine-tuning在极少样本条件下表现较差

---

# 作者

人工智能课程林瑞荷小组

2026

## 说明

由于数据集 CUB-200-2011 体积较大，未包含在提交文件中。
