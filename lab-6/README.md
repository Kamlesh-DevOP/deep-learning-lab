# CS3807: Deep Learning Laboratory — Experiment 6

## End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding



---

### Student & Course Information

* **Institution:** Shiv Nadar University Chennai


* **Department:** Department of Artificial Intelligence & Data Science


* **Degree & Branch:** B.Tech Artificial Intelligence & Data Science (Semester V)


* **Subject Code & Course Title:** CS3807 -- Deep Learning Laboratory


* **Academic Year:** 2026--2027


* **Student Name:** Kamlesh TJ
* **Roll / Register Number:** 24011101050

---

## 1. Overview & Objectives

This repository contains the complete implementation and experimental analysis for **Experiment 6: End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding**.

The primary objectives of this laboratory are:

1. **Mathematical Verification of BPTT:** Manually and programmatically verify recurrence updates ($h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$) to explore the vanishing/exploding gradient phenomenon during Backpropagation Through Time.


2. **Temporal Sensor Sequence Classification:** Preprocess and structure continuous smartphone accelerometer and gyroscope measurements from the UCI Human Activity Recognition (HAR) dataset into a 3D temporal tensor representation ($X \in \mathbb{R}^{N \times 128 \times 9}$).


3. **Comparative Architecture Benchmarking:** Train and benchmark Vanilla SimpleRNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) models under identical experimental conditions across multiple quantitative metrics (Accuracy, Macro Precision, Macro Recall, Macro F1-score, trainable parameters, and wall-clock training time).


4. **Sequence Context Sensitivity:** Analyze the impact of varying temporal receptive field lengths ($T \in \{32, 64, 128\}$) on classification performance and compute cost.


5. **Video Action Recognition:** Construct a two-stage hybrid spatio-temporal pipeline that extracts spatial embeddings using a frozen pretrained MobileNetV2 backbone ($D = 1280$) and models action sequences using an LSTM/GRU classifier.


6. **Sequence-to-Sequence (Seq2Seq) Modeling:** Implement an encoder--decoder LSTM network on an integer reversal task to demonstrate the difference between token-level and sequence-level accuracy.



---

## 2. Repository Structure

```text
├── README.md                      # Comprehensive project setup and execution guide
├── dl-lab-6.ipynb       # Fully executable Jupyter notebook with all outputs
├── requirements.txt     #Contains the requirements for running the lab
├── figures/                       # Generated laboratory plots
│   ├── plot1_sensor_signals.png
│   ├── plot2_loss_curves.png
│   ├── plot3_accuracy_curves.png
│   ├── plot4_confusion_matrices.png
│   ├── plot5_model_comparison.png
│   ├── plot6_sequence_length.png
│   ├── plot7_video_frames.png
│   ├── plot8_video_curves.png
│   └── plot9_video_confusion_matrix.png
└── Lab_6.pdf        # The formal lab report

```

---

## 3. Environment & Prerequisites

* **Python Version:** Python 3.10 to 3.12 (the notebook was verified on Python 3.12.13)


* **Deep Learning Framework:** TensorFlow 2.16+ / 2.20+


* **Compute Hardware:** CPU or GPU (NVIDIA T4 / P100 recommended for faster video feature extraction; CPU is fully supported)



### Required Libraries

Create a `requirements.txt` file or install the dependencies directly:

```bash
# Install core numerical, machine learning, and visualization libraries
pip install numpy pandas matplotlib seaborn scipy scikit-learn opencv-python tensorflow

```

---

## 4. Dataset Setup & Configuration

### A. Primary Dataset: UCI Human Activity Recognition with Smartphones



The notebook is configured to load the dataset from the Kaggle environment.

* **Kaggle Source:** [Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
* **Expected Paths in Kaggle:**
* Training CSV: `/kaggle/input/datasets/uciml/human-activity-recognition-with-smartphones/train.csv`

* Testing CSV: `/kaggle/input/datasets/uciml/human-activity-recognition-with-smartphones/test.csv`

* *Automatic Fallback:* The notebook includes an auto-detection fallback for `/kaggle/input/human-activity-recognition-with-smartphones/`.




* **Local Setup (Optional):** If running locally, download the two CSV files from Kaggle and update the path variables in Cell 3:
```python
# Set paths to your local directory containing train.csv and test.csv
train_csv_path = "./data/train.csv"
test_csv_path = "./data/test.csv"

```



### B. Video Dataset: UCF-101



* **Expected Kaggle Path:** `/kaggle/input/ucf101/UCF-101`

* **Classes Used:** `Basketball`, `Biking`, `Walking`, `TennisSwing`

* **Built-in Fallback:** If the external UCF-101 video dataset is not attached, Cell 14 automatically generates controlled synthetic video sequence tensors matching the required format ($(B, 10, 224, 224, 3)$) to allow end-to-end execution without interruption.



---

## 5. Step-by-Step Execution Guide

### Option 1: Running on Kaggle (Recommended)

1. **Log in to Kaggle:** Open [Kaggle](https://www.kaggle.com) and go to the **Code** tab.
2. **Create / Upload Notebook:**
* Click **New Notebook**.
* Go to **File** $\rightarrow$ **Import Notebook** $\rightarrow$ upload `dl-lab-6.ipynb`.


3. **Attach the Datasets:**
* In the right sidebar, click **Add Input**.
* Search for `uciml/human-activity-recognition-with-smartphones` and click the **+** icon to add it.
* Search for `ucf101` and add it.




4. **Configure Accelerator:**
* In the right sidebar under **Notebook options**, select **GPU T4 x2** (or keep **CPU**).
* Ensure **Internet** is toggled **Off** or **On** as needed (dataset loading works offline once attached).


5. **Run the Notebook:** Click **Run All** (or execute cells sequentially using `Shift + Enter`).

---

### Option 2: Running Locally or on Google Colab

1. **Clone the Repository:**
```bash
# Clone repository to local machine
git clone https://github.com/[Your-Username]/CS3807-Deep-Learning-Lab-6.git
cd CS3807-Deep-Learning-Lab-6

```


2. **Set Up a Virtual Environment:**
```bash
# Create and activate an isolated virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate          # On Windows

```


3. **Install Dependencies:**
```bash
# Install required packages
pip install -r requirements.txt

```


4. **Update Dataset Paths:**
* Open `dl-lab-6.ipynb` in VS Code or JupyterLab.
* Locate **Cell 3** and point `train_csv_path` and `test_csv_path` to your local folder.


5. **Start Jupyter and Execute:**
```bash
# Launch the Jupyter Notebook server
jupyter notebook dl-lab-6.ipynb

```


Execute cells sequentially from top to bottom.

---

## 6. Notebook Pipeline & Cell-by-Cell Breakdown

| Cell Index | Section in Lab Manual | Description & Output

 |
| --- | --- | --- |
| **Cell 1** | Setup & Initialization | Imports TensorFlow, NumPy, Pandas, Scipy, OpenCV, and Matplotlib; fixes random seeds.

 |
| **Cell 2** | Section 8: BPTT Exercise | Manually and programmatically derives $h_1, h_2, h_3$ for $x = [0.5, 0.7, 0.2]$; verifies output matches analytical values ($< 10^{-4}$ error).

 |
| **Cell 3** | Section 3 & 5: Dataset Loading | Loads tabular Kaggle CSVs, encodes the 6 activity labels ($0\dots5$), and maps feature vectors into a 3D temporal tensor ($N \times 128 \times 9$).

 |
| **Cell 4** | Section 5: Preprocessing | Performs a stratified 70% train / 15% validation / 15% test split ($N = 3000$); applies channel-wise z-score normalization using training statistics.

 |
| **Cell 5** | Section 6: Plot 1 | Visualizes sensor signal trajectories across dynamic (`WALKING`) vs. static (`SITTING`, `LAYING`) activities over 128 time steps.

 |
| **Cell 6** | Sections 9--12: Architecture Builder | Implements a modular Keras model builder for `SimpleRNN`, `LSTM`, and `GRU` with a 32-unit recurrent layer, 0.2 Dropout, Dense(16, ReLU), and Softmax(6).

 |
| **Cell 7** | Section 12: Controlled Training | Trains SimpleRNN, LSTM, and GRU models across 30 epochs with batch size 32 using the Adam optimizer ($\eta = 10^{-3}$) under identical conditions.

 |
| **Cell 8** | Section 13: Plots 2 & 3 | Generates side-by-side training and validation loss curves (Plot 2) and accuracy curves (Plot 3) across epochs.

 |
| **Cell 9** | Sections 14 & 16: Evaluation Tables | Computes test Accuracy, Macro Precision, Macro Recall, Macro F1, parameter counts, training times, and displays the Section 16 architectural comparison table.

 |
| **Cell 10** | Section 15: Plot 4 | Plots separate $6 \times 6$ confusion matrices for SimpleRNN, LSTM, and GRU to evaluate pairwise activity misclassifications.

 |
| **Cell 11** | Section 16: Plot 5 | Creates a dual-axis bar/line chart illustrating the trade-off among predictive performance, model complexity, and training cost.

 |
| **Cell 12** | Section 17: Plot 6 | Evaluates sequence length truncation ($T \in \{32, 64, 128\}$) against Macro F1-score and training runtime.

 |
| **Cell 13** | Section 28: Additional Exercises | Benchmarks unit scaling (16, 32, 64 units), a stacked 2-layer LSTM, and a Bidirectional LSTM.

 |
| **Cell 14** | Sections 18--21: Video Action Recognition | Uses frozen MobileNetV2 features ($10 \times 1280$) coupled with an LSTM/GRU to classify video actions; outputs Plots 7, 8, 9 and prediction confidence.

 |
| **Cell 15** | Sections 22--24: Sequence-to-Sequence | Implements an Encoder-Decoder LSTM for an integer reversal task; displays 5 qualitative samples and compares token vs. sequence accuracy.

 |
| **Cell 16** | Section 25: Consolidated Summary | Produces the final consolidated evaluation table summarizing all models developed throughout the experiment.

 |

---

## 7. Key Results & Benchmark Summary

### A. Primary Sequence Classification Performance ($T = 128$)



| Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Parameters | Training Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| **Vanilla SimpleRNN** | 84.89 | 85.12 | 84.78 | 84.85 | **1,974** | **38.45** |
| **LSTM** | 92.44 | 92.68 | 92.35 | 92.42 | 6,006 | 72.18 |
| **GRU** | **93.33** | **93.55** | **93.28** | **93.37** | 4,758 | 61.32 |

### B. Sequence Length Sensitivity ($T \in \{32, 64, 128\}$)



| Sequence Length ($T$) | SimpleRNN F1 (%) | LSTM F1 (%) | GRU F1 (%) |
| --- | --- | --- | --- |
| **32** | 72.15 | 81.42 | 82.20 |
| **64** | 79.30 | 87.65 | 88.40 |
| **128** | **84.85** | **92.42** | **93.37** |

### C. Sequence-to-Sequence Evaluation



* **Token Accuracy:** $96.85\%$

* **Sequence-Level Accuracy:** $85.38\%$

* **Theoretical Relation:** Because a sequence is marked incorrect if even a single token is mispredicted, sequence accuracy scales exponentially with sequence length $L$: $\text{Acc}_{\text{seq}} \approx (\text{Acc}_{\text{token}})^L \approx (0.9685)^5 \approx 85.21\%$.



---

## 8. Troubleshooting & Common Issues

* **Dataset Path Not Found:** If you see a `FileNotFoundError` in Cell 3, verify that the dataset `human-activity-recognition-with-smartphones` is attached under `/kaggle/input/`.


* **Keras 3 Shape Warning:** If using TensorFlow 2.16+, input layers prefer explicit tuples. The notebook handles this via `layers.Input(shape=(128, 9))`.


* **Memory Limits during Video Processing:** Cell 14 resizes frames to $224 \times 224$ and extracts features in batches of 16. If running on low-RAM instances, reduce the number of video clips processed per class.

---
