# Experiment 1 - Single-Layer Perceptron 

This repository contains a manual implementation of the perceptron learning algorithm, applied to two tasks:

A single-layer perceptron built using NumPy, trained to classify banknotes as authentic or forged based on four statistical features extracted from banknote images. The pipeline covers dataset exploration, preprocessing, training with the perceptron learning rule, evaluation using standard classification metrics, a learning-rate comparison, and a comparison against Scikit-learn's Perceptron implementation.

The same perceptron learning algorithm implemented independently for the AND, OR, and NOT gates, with weights displayed after every update and the decision boundary to visualize how the perceptron converges for each gate.

## 1. Repository Contents

- **Code:** Google Colab notebook containing the complete implementation for both the banknote classification task and the logic gate experiments, along with outputs.
- **Report:** A complete PDF file documenting the objective, theory, methodology, results, and analysis for both experiments.

---

## 2. What the notebook does

The notebook is organized to mirror the lab manual's task structure, plus additional tasks:

1. **Setup** — imports, a fixed random seed, and one shared Matplotlib style (Times New Roman, bold axes/legends, 600 dpi EPS export) used by every plot.
2. **Task 1 — Dataset Exploration:** shape, dtypes, missing-value check, class balance, descriptive statistics.
3. **Task 2 — EDA:** feature histograms, correlation heatmap, scatter plots, boxplots (all clubbed into multi-panel figures).
4. **Task 3 — Preprocessing:** feature standardization (fit on train only) and an 80/20 stratified train/test split.
5. **Task 4 — Perceptron (from scratch):** a single `Perceptron` class — weight/bias initialization, step activation, forward pass, and the classic perceptron learning rule.
6. **Task 5 — Training:** per-epoch training log (misclassified count, weights, bias) plus training-dynamics plots.
7. **Task 6 — Evaluation:** accuracy, precision, recall, F1-score, and a confusion matrix on the held-out test set.
8. **Task 7 — Learning Rate Study:** the same model retrained at η = 0.001, 0.01, and 0.1, compared side by side.
9. **Optional — Decision Boundary:** a 2-feature (variance, skewness) perceptron with its separating hyperplane plotted directly.
10. **Additional Task — Scikit-learn comparison:** a sanity check against `sklearn.linear_model.Perceptron` on the same split.
11. **Additional Task — Step vs. Sigmoid:** a visual side-by-side of the two activation shapes.
12. **Additional Task — XOR:** the perceptron trained on the XOR truth table, demonstrating non-convergence.
13. **Additional Task — Normalization effect:** raw vs. standardized features compared under identical hyperparameters (with a documented caveat about shared RNG state across sequential runs — see the report's Discussion section).
14. **Additional Task — Logic Gates (OR / AND / NOT / XOR):** weights and decision boundary plotted after **every individual weight update** (not just per epoch) for each gate, including a worked explanation of the periodic weight-cycling pattern that prevents XOR from converging.
15. **Performance Summary Table:** a consolidated table of the final model's configuration and metrics.

Every figure the notebook produces is saved automatically to `<name>.eps` at 600 dpi as it's generated — no manual export step is needed.

---

## 3. How to run it

### 3.1 Prerequisites

- Python 3.10+ (tested with 3.12)
- Jupyter (Notebook, JupyterLab, or VS Code's Jupyter extension)

### 3.2 Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter ipykernel
```

### 3.3 Get the code and data

```bash
git clone https://github.com/Kamlesh-DevOP/deep-learning-lab.git
cd .\deep-learning-lab\lab-1\
```

### 3.4 Run the notebook

**Option A — Jupyter Notebook / JupyterLab:**

```bash
jupyter notebook Lab1.ipynb
```

Then use **Run → Run All Cells** (or run cells top to bottom with `Shift+Enter`).

**Option B — VS Code:** open the `.ipynb` file directly and click **Run All** at the top of the notebook.

### 3.5 Expected output

- All intermediate results (dataset summary, training logs, evaluation metrics, comparison tables) print inline as the notebook runs.
- 15 figures are written as to the same folder as `.eps` files, each named to match its section (e.g. `perceptron_confusion_matrix.eps`).
- Internet connection is required to download the dataset from ucimlrepo.

### 3.6 Runtime

The full notebook runs in well under a few minutes on a standard laptop CPU; there is no GPU dependency anywhere in this experiment.

---