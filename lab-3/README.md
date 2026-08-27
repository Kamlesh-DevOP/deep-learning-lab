# Lab 3: Implementation of Convolutional Neural Networks (CNNs) for Image Classification

## Dataset
This repository utilizes the **CIFAR-10** dataset .
* **Training Images:** 50,000 
* **Testing Images:** 10,000 
* **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) 
* **Image Size:** 32x32x3 (RGB) 

## Repository Contents
* `Experiment_3.ipynb`: The main PyTorch Jupyter Notebook containing all lab tasks and model training.
* `Experiment__3_Lab_Manual.pdf`: The compiled LaTeX lab manual and report .
* `*.png`: Saved visualizations including sample images, feature maps, performance curves, and confusion matrices.

## What's Implemented
The lab explores the mechanics of Convolutional Neural Networks by implementing the following: 
* **Dataset Exploration:** Loading CIFAR-10 and visualizing class distributions .
* **Convolution Mechanics:** Building custom 2D convolution layers to study the spatial impact of varying kernel sizes (3x3, 5x5, 7x7), strides, and padding techniques .
* **Feature Visualization:** Extracting and displaying internal feature maps from early layers to understand pattern detection .
* **Pooling Comparison:** Benchmarking Max Pooling against Average Pooling .
* **CNN Construction:** Building a deep sequential model (Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax) and training it for 20 epochs using the Adam optimizer .
* **Evaluation:** Generating rigorous classification metrics and confusion matrices .

## Results Summary
The custom CNN architecture achieved the following performance metrics on the CIFAR-10 test set:
* **Training Accuracy:** 75.84%
* **Testing Accuracy:** 68.40%
* **Precision:** 69.69%
* **Recall:** 68.40%
* **F1-score:** 68.68%
* **Total Trainable Parameters:** 25,578
* **Pooling Comparison:** Max Pooling (59.38%) outperformed Average Pooling (54.41%) during baseline 3-epoch testing.

## Requirements
To execute the notebook, ensure you have the following Python libraries installed:
* `torch` and `torchvision`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`

## Running
Change the runtime to GPU. 
Run all cells top to bottom. Total runtime: ~25 minutes. 
Generated figures are saved to the root folder (600 DPI) as the notebook executes.

## Notes
See the full report for detailed results, all figures with interpretation, and discussion.
