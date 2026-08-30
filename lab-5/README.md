# CS3807 Deep Learning Laboratory — Experiment 5

## Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning, and Cross-Validation

**Course:** CS3807 — Deep Learning Laboratory   
**Department:** Computer Science and Engineering, Shiv Nadar University Chennai   
**Dataset:** Oxford-IIIT Pet Dataset (37 classes)   
**Architecture:** MobileNetV2   

---

## 📌 Project Overview

This laboratory experiment provides a comprehensive empirical study on the key factors governing Convolutional Neural Network (CNN) performance :
* **Weight Initialization:** Zero, Random Normal, Xavier/Glorot Uniform, and He (Kaiming) Normal .
* **Regularization & Batch Normalization:** Baseline (No regularization), $L_2$ Weight Decay, Dropout, and Batch Normalization .
* **Optimization Dynamics:** SGD, SGD with Momentum, RMSProp, and Adam .
* **CNN Hyperparameter Sweeps:** Learning Rate ($\eta \in \{10^{-3}, 10^{-4}\}$), Batch Size ($B \in \{16, 32, 64\}$), and Dropout Probability ($p \in \{0.0, 0.25, 0.5\}$) .
* **Transfer Learning vs. Fine-Tuning:** Feature extraction (frozen base) vs. layered top-backbone fine-tuning .
* **Model Selection:** 5-Fold Cross-Validation on training data with standard deviation reporting .
* **Final Evaluation:** Generalization verification on an untouched test set, confusion matrix, and misclassification error analysis .

---

## 📁 Repository Structure

```text
├── data/                                # Oxford-IIIT Pet Dataset directory
├── dl-lab-5.ipynb                       # Complete end-to-end PyTorch training notebook
├── Lab5_Report.tex                      # Formal LaTeX laboratory report
├── README.md                            # Project overview and reproduction guide
└── plots/                               # Generated experimental figures
    ├── plot_1_weight_init_loss.png      # Training Loss vs. Epoch across Initializations
    ├── plot_2_weight_init_acc.png       # Val Acc vs. Epoch across Initializations
    ├── plot_3_4_regularization.png      # Regularization & Generalization Gap Curves
    ├── plot_5_batch_normalization.png   # With vs. Without Batch Normalization
    ├── plot_6_optimizers_loss.png       # Training Loss vs. Epoch across Optimizers
    ├── plot_7_optimizers_acc.png        # Val Acc vs. Epoch across Optimizers
    ├── plot_8_lr_acc.png                # Learning Rate vs. Validation Accuracy
    ├── plot_9_bs_acc.png                # Batch Size vs. Validation Accuracy
    ├── plot_10_dropout_acc.png          # Dropout Rate vs. Validation Accuracy
    ├── plot_11_fe_vs_ft_acc.png         # Feature Extraction vs. Fine-Tuning Accuracy
    ├── plot_12_fe_vs_ft_loss.png        # Training & Validation Loss Trajectories
    ├── plot_13_cv_accuracy.png          # 5-Fold CV Accuracy with Error Bars
    ├── plot_14_confusion_matrix.png     # 37-Class Pet Breed Confusion Matrix
    └── plot_15_misclassified_cases.png  # Diagnostic Misclassified Test Samples