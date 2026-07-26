# Lab 2: Implementation of a Multi-Layer Perceptron for Multi-Class Image Classification

This experiment implements a Multi-Layer Perceptron using TensorFlow/Keras to classify grayscale clothing images into 10 categories. It covers the full deep learning workflow — image preprocessing, model construction, training, evaluation, and automated hyperparameter optimization using RandomizedSearchCV.

A baseline MLP (784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)) is trained first, followed by a hyperparameter search over layer count, neuron width, learning rate, batch size, optimizer, activation function, and dropout rate. The best configuration found is retrained and compared against the baseline.

## Dataset

**Fashion-MNIST** — 60,000 training images and 10,000 test images across 10 clothing categories, each a 28×28 grayscale image.

## Repository Contents

- `Lab 2.ipynb` — Colab notebook with all tasks - dataset exploration, preprocessing, model building, training, evaluation, and hyperparameter search.
- `lab-2.pdf` — Full lab report in PDF (theory, procedure, results, plots with inferences, discussion, conclusion).

## What's Implemented

### 1. MLP on Fashion-MNIST

- Data pipeline: load → flatten (28×28 → 784) → normalize [0,1] → one-hot encode
- Baseline architecture: `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)` (109,386 params)
- Full evaluation suite: accuracy, macro-precision/recall/F1, confusion matrix, classification report
- Single-layer perceptron implemented from scratch (NumPy only, no `sklearn.linear_model.Perceptron`)

### 2. Automated Hyperparameter Search

- `RandomizedSearchCV` + SciKeras `KerasClassifier` over an 8-dimensional search space (hidden layers, neurons, learning rate, batch size, epochs, optimizer, activation, dropout)

- 5 sampled candidates × 5-fold CV = 25 fits
- Best model retrained from scratch on the full training set and evaluated independently on the held-out test set

## Results Summary

| Model | Test Accuracy | Macro F1 | Training Time |
|---|---|---|---|
| Baseline MLP | 86.74%  | 0.8680 | 108s |
| Optimized MLP (search) | 88.21% | 0.8824 | 84s |

The hyperparameter search's CV estimate (88.21%) did not fully hold up on the test set — see [report](https://github.com/Kamlesh-DevOP/deep-learning-lab/blob/main/lab-2/Lab_2.pdf) for further discussion.

## Requirements

Designed to run on **Google Colab** (GPU runtime recommended).

```bash
pip install scikit-learn<1.6 scikeras>=0.13 kagglehub
```

> `scikeras` currently requires `scikit-learn < 1.6` (breaking change in `__sklearn_tags__`). Install the pinned versions above, then **restart the runtime** before running the rest of the notebook.

Core stack: `tensorflow`, `scikit-learn`, `scikeras`, `numpy`, `pandas`, `matplotlib`, `seaborn`.

## Running

1. Change the runtime to GPU.
2. Run all cells top to bottom. Total runtime ≈ 50-60 minutes, dominated by the hyperparameter search (~40 mins).
3. Generated figures are saved to the root folder (600 DPI) as the notebook executes.

## Notes

- Random seed fixed (`SEED = 42`) for reproducibility of weight initialization and data splits; exact metrics can still drift slightly run-to-run due to GPU non-determinism in TensorFlow.
- All plots use Times New Roman, 13–15pt, with bold axis labels/legends per lab formatting requirements.
- See the full [report](https://github.com/Kamlesh-DevOP/deep-learning-lab/blob/main/lab-2/Lab_2.pdf) for detailed results, all figures with interpretation, and discussion.