# Lab 2: Implementation of a Multi-Layer Perceptron for Multi-Class Image Classification

This experiment implements a Multi-Layer Perceptron using TensorFlow/Keras to classify grayscale clothing images into 10 categories. It covers the full deep learning workflow — image preprocessing, model construction, training, evaluation, and automated hyperparameter optimization using RandomizedSearchCV.

A baseline MLP (784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)) is trained first, followed by a hyperparameter search over layer count, neuron width, learning rate, batch size, optimizer, activation function, and dropout rate. The best configuration found is retrained and compared against the baseline.

## Dataset

**Fashion-MNIST** — 60,000 training images and 10,000 test images across 10 clothing categories, each a 28×28 grayscale image.

## Repository Contents

- `Lab 2.ipynb` — Colab notebook with all tasks - dataset exploration, preprocessing, model building, training, evaluation, and hyperparameter search.
- `lab-2.pdf` — Full lab report in PDF (theory, procedure, results, plots with inferences, discussion, conclusion).