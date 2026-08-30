# CS3807 Deep Learning Laboratory — Experiment 5

## Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning, and Cross-Validation

---

## 📌 Project Summary

This repository contains the complete experimental pipeline for evaluating the key factors that govern Convolutional Neural Network (CNN) performance.

Using the lightweight **MobileNetV2** architecture and the **Oxford-IIIT Pet Dataset**, this project systematically analyzes:

* **Weight Initialization:** Comparing Zero, Random Normal, Xavier/Glorot, and He initializations.
* **Regularization & Batch Normalization:** Assessing generalization gaps using L2 Weight Decay, Dropout, and Batch Normalization.
* **Optimization Dynamics:** Evaluating convergence behavior across SGD, Momentum, RMSProp, and Adam.
* **Hyperparameter Sweeps:** Tuning learning rates, batch sizes, and dropout probabilities.
* **Transfer Learning:** Comparing frozen feature extraction with top-layer fine-tuning.
* **Model Selection:** Using 5-Fold Cross-Validation for robust configuration selection.

### 🏆 Best Results

The optimal configuration achieved:

| Metric                          |     Result |
| ------------------------------- | ---------: |
| Mean Cross-Validation Accuracy  | **86.09%** |
| Final Independent Test Accuracy | **83.18%** |

---

## ⚙️ Prerequisites

To run the notebook and reproduce the experiments, you need a Python environment with PyTorch and the required data science libraries.

### Required Software

* Python **3.8+**
* Jupyter Notebook or JupyterLab
* CUDA Toolkit *(optional, but highly recommended for GPU acceleration)*

### Required Libraries

* `torch`
* `torchvision`
* `numpy`
* `matplotlib`
* `scikit-learn`

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/CS3807-Deep-Learning-Lab.git
cd CS3807-Deep-Learning-Lab/Lab5
```

### 2. Create a Virtual Environment

Creating a virtual environment is recommended to keep dependencies isolated.

```bash
python -m venv venv
```

Activate the environment:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn jupyter
```

---

## 📂 Dataset Setup

The project uses the **Oxford-IIIT Pet Dataset**, which contains images belonging to **37 different pet breeds**.

No manual dataset download or preprocessing is required.

The provided Jupyter notebook uses:

```python
torchvision.datasets.OxfordIIITPet
```

with `download=True`.

On the first execution, the dataset will automatically be downloaded into the local `./data` directory.

### Preprocessing

The pipeline automatically applies standard ImageNet preprocessing:

* Resize images to **224 × 224** pixels
* Convert images to RGB
* Normalize RGB channels using ImageNet normalization statistics

### Dataset Split

The dataset is divided as follows:

* **80% Training**
* **20% Validation**
* **Official Test Split:** Reserved strictly for final evaluation

The training/validation split is created from the official `trainval` split, while the official `test` split remains untouched until final evaluation.

---

## 🚀 Execution Guide

### 1. Launch Jupyter Notebook

From the project directory, run:

```bash
jupyter notebook
```

Alternatively, you can use:

```bash
jupyter lab
```

### 2. Open the Notebook

Open:

```text
dl-lab-5.ipynb
```

### 3. Run the Notebook

Execute the cells sequentially.

Make sure the following are executed successfully before proceeding to the plotting and analysis sections:

1. Dataset download and preprocessing
2. Model-building functions
3. Training utilities
4. Individual experiments
5. Evaluation functions

### 4. Generated Plots

The notebook generates and saves **15 diagnostic plots** in the project directory.

These plots cover:

* Weight initialization
* Regularization
* Batch normalization
* Optimizer comparison
* Learning-rate tuning
* Batch-size tuning
* Dropout tuning
* Transfer learning
* Cross-validation
* Confusion matrix
* Misclassified samples

### 5. Lab Report Output

The final execution cell generates formatted **LaTeX table rows**, which can be directly copied into the laboratory report.

---

## 📊 Repository Structure

```text
CS3807-Deep-Learning-Lab/
│
├── Lab5/
│   │
│   ├── dl-lab-5.ipynb
│   │   # Complete end-to-end PyTorch training notebook
│   │
│   ├── Lab5_Report.tex
│   │   # Formal LaTeX laboratory report
│   │
│   ├── README.md
│   │   # Project overview and instructions
│   │
│   ├── data/
│   │   # Automatically generated dataset directory
│   │
│   └── plots/
│       │
│       ├── plot_1_weight_init_loss.png
│       ├── plot_2_weight_init_acc.png
│       ├── plot_3_4_regularization.png
│       ├── plot_5_batch_normalization.png
│       ├── plot_6_optimizers_loss.png
│       ├── plot_7_optimizers_acc.png
│       ├── plot_8_lr_acc.png
│       ├── plot_9_bs_acc.png
│       ├── plot_10_dropout_acc.png
│       ├── plot_11_fe_vs_ft_acc.png
│       ├── plot_12_fe_vs_ft_loss.png
│       ├── plot_13_cv_accuracy.png
│       ├── plot_14_confusion_matrix.png
│       └── plot_15_misclassified_cases.png
│
└── ...
```

---

## 📈 Final Evaluation

The final model is evaluated on the **official Oxford-IIIT Pet test split**, which is kept separate from training and validation throughout the experimental pipeline.

### Final Performance

**Test Accuracy: 83.18%**

The cross-validation results indicate that the selected configuration generalizes well, achieving a mean validation accuracy of **86.09%** across five folds.

---

## 👨‍💻 Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **MobileNetV2**
* **Scikit-learn**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**
* **LaTeX**

---

## 📚 Dataset

**Oxford-IIIT Pet Dataset**

The dataset consists of images from 37 different breeds of cats and dogs and is widely used for image classification and computer vision research.

---