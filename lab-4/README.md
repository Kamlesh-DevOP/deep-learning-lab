# Experiment 4: Comparative Study of Deep CNN Architectures Using Transfer Learning

**Course:** CS3807 -- Deep Learning Laboratory
**Dataset:** CIFAR-10 ($32 \times 32 \times 3$ across 10 classes)

---

## 📌 Project Overview

This laboratory project explores the practical application of transfer learning, fine-tuning, and architectural benchmarking across classic and modern deep Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset.

### Objectives
* **Transfer Learning:** Repurpose pre-trained ImageNet architectures (**MobileNetV2**, **VGG16**, **ResNet50**) using custom classification heads.
* **Fine-Tuning:** Unfreeze top convolutional blocks to adapt feature representations to low-resolution ($32 \times 32$) images.
* **Architecture Comparison:** Benchmark transfer learning models against baseline custom architectures (**LeNet-5**, adapted **AlexNet**).
* **Automated Hyperparameter Optimization:** Implement randomized search across learning rates, batch sizes, optimizers, and layer-freezing configurations.
* **Model Evaluation:** Generate classification reports, macro precision/recall/F1 metrics, loss/accuracy trajectories, and confusion matrices.

---

## 🛠️ Prerequisites

* **Operating System:** Linux, macOS, or Windows 10/11
* **Python Version:** Python 3.9, 3.10, or 3.11
* **Hardware Acceleration (Recommended):** NVIDIA GPU with CUDA 11.8+ / 12.0+ support, or Apple Silicon MPS

---

## 📦 Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/deep-learning-lab4.git
cd deep-learning-lab4
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

# On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### `requirements.txt`
```text
tensorflow>=2.10.0
numpy>=1.22.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
pandas>=1.4.0
```

---

## 📂 Dataset Setup

The experiment uses the raw Python binary batches of the **CIFAR-10** dataset.

1. Download `cifar-10-python.tar.gz`.
2. Extract the archive into the root directory of the project:
```bash
tar -xvzf cifar-10-python.tar.gz
```
3. Ensure the extracted folder is structured as follows:

```text
deep-learning-lab4/
│
├── cifar-10-batches-py/
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   └── test_batch
│
├── plots/
│   ├── accuracy_loss_curves.png
│   ├── confusion_matrix.png
│   ├── optimizer_comparison.png
│   └── cifar10_samples.png
│
├── experiment_4_notebook.ipynb
├── report.tex
├── requirements.txt
└── README.md
```

---

## 🚀 Execution Guide

Run the Jupyter Notebook interactively or execute cells sequentially:

` ` `bash
jupyter notebook experiment_4_notebook.ipynb
` ` `

### Execution Pipeline
1. **Task 1 (Data Ingestion & Preprocessing):** Unpickles binary batches, normalizes pixel values to $[0, 1]$, and one-hot encodes labels across 10 target classes.
2. **Tasks 2 & 3 (MobileNetV2 Transfer Learning):** Instantiates pre-trained `MobileNetV2(include_top=False)` with a custom Global Average Pooling and Dense classification head. Trains with Adam ($\text{LR}=0.001$) for 10 epochs.
3. **Task 4 (Fine-Tuning):** Unfreezes the final 15 layers of MobileNetV2 and trains for an additional 10 epochs.
4. **Task 5 (Evaluation & Metrics):** Plots loss/accuracy curves, generates the confusion matrix, and computes the macro classification report.
5. **Additional Exercises:**
   * Transfer learning with **VGG16** and **ResNet50**.
   * Optimization trajectory comparison: **Adam** vs. **SGD (Momentum = 0.9)**.
   * Native training of **LeNet-5** and **AlexNet (Adapted)**.
6. **Hyperparameter Tuning:** Automated randomized search over learning rate, batch size, optimizer type, dense units, and layer-freezing depth.

---

## 📊 Experimental Results & Summary

### 1. Architectural Benchmark on CIFAR-10

| Architecture | Model Strategy | Total Parameters | Test Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **AlexNet (Adapted)** | Trained from Scratch | $\approx 686\text{K}$ | **75.62\%** |
| **VGG16** | Transfer Learning (Frozen Base) | $\approx 14.78\text{M}$ | **61.53\%** |
| **LeNet-5** | Trained from Scratch | $\approx 62\text{K}$ | **60.77\%** |
| **MobileNetV2** | Transfer Learning (Fine-Tuned) | $\approx 2.42\text{M}$ | **54.43\%** |
| **ResNet50** | Transfer Learning (Frozen Base) | $\approx 23.85\text{M}$ | **39.49\%** |
| **MobileNetV2** | Transfer Learning (Frozen Base) | $\approx 2.42\text{M}$ | **35.10\%** |

---

### 2. MobileNetV2 Performance: Frozen Base vs. Fine-Tuning

| Metric | Frozen Base | Fine-Tuned Stage | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 35.10% | **54.43%** | **+19.33%** |
| **Macro Precision** | 0.3619 | **0.6100** | +0.2481 |
| **Macro Recall** | 0.3573 | **0.5400** | +0.1827 |
| **Macro F1-Score** | 0.3537 | **0.5500** | +0.1963 |

---

### 3. Optimizer Comparison (MobileNetV2 Base)

* **Adam ($\text{LR}=0.001$):** Achieved rapid early convergence, reaching $\sim 35.80\%$ validation accuracy by Epoch 6.
* **SGD ($\text{LR}=0.01, \text{Momentum}=0.9$):** Produced stable, smooth convergence curves reaching $34.68\%$ by Epoch 9.

---

## 🔍 Key Findings & Inferences

* **Resolution Mismatch in Deep Networks:** ResNet50 and MobileNetV2 with completely frozen ImageNet bases struggled on $32\times32$ inputs due to aggressive initial stride downsampling designed for $224\times224$ images.
* **Effectiveness of Fine-Tuning:** Unfreezing the top 15 layers of MobileNetV2 enabled higher-level feature maps to adapt to CIFAR-10 semantics, yielding an absolute accuracy boost of **+19.33%**.
* **Domain Adaptation vs. Native Design:** Custom AlexNet adapted directly for $32\times32$ inputs without aggressive early downsampling achieved the highest overall accuracy (**75.62%**), highlighting the importance of kernel scale alignment with input resolution.