# Breast Cancer Classification using Machine Learning — WBCD

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning pipeline for breast cancer classification, developed during a 30-day AICTE SAMARTHAN internship at **IIIT Bhagalpur**, under the **Department of Computer Science & Engineering**, mentored by **Dr. Suneel Kumar**.

This project reproduces and extends the feature-selection-driven classification methodology on the **Wisconsin Breast Cancer Dataset (WBCD)**, adding metaheuristic optimization, automated hyperparameter tuning, ensemble learning, explainability, and statistical validation.

---

## Internship Details

| | |
|---|---|
| **Program** | AICTE SAMARTHAN Internship Connect 2026–27 |
| **Host Institute** | IIIT Bhagalpur (Bhagalpur College of Engineering Campus, Sabour, Bihar) |
| **Department** | Computer Science & Engineering — AI, Data Science, Cybersecurity & Blockchain |
| **Mentor** | Dr. Suneel Kumar, Guest Faculty, CSE |
| **Mode** | Offline, on-campus |
| **Duration** | 30 days (June 15 – July 15, 2026) |
| **Intern** | Abdul Basir, B.Tech CSE, USTM, Meghalaya |

---

## Project Overview

Breast cancer is one of the most prevalent cancers among women globally, and early, accurate diagnosis significantly improves treatment outcomes. This project builds a complete ML pipeline to classify tumors as **benign** or **malignant** using the Wisconsin Breast Cancer dataset, comparing multiple feature selection strategies and classification algorithms to identify the most effective combination.

**Reference paper reproduced:** Sharma, S. & Mishra, R. (2022) — CFS + IG + SFS feature selection with a Voting Ensemble (ANN + SVM + LR), reporting 99.41% accuracy on WBCD.

---

## Dataset

**Wisconsin Breast Cancer Dataset (Original)** — UCI Machine Learning Repository

- **699 samples**, 9 cytological features + class label
- **Class distribution:** 458 benign, 241 malignant
- **Features:** clump thickness, cell size uniformity, cell shape uniformity, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli, mitoses
- **Missing values:** 16 rows (`bare_nuclei`), imputed using median
- **Source:** [UCI WBCD](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data)

> Note: This is distinct from the 30-feature WDBC dataset (`sklearn.datasets.load_breast_cancer()`). This project deliberately uses the original 9-feature WBCD to match the reference paper's methodology.

---

## Pipeline Architecture

```
WBCD Dataset → Preprocessing → Feature Selection (7 methods) →
10 Classifiers → Hyperparameter Tuning (Optuna) → 10-Fold CV →
Ensembles (Voting + Stacking) → Evaluation → Explainability (SHAP) →
Statistical Validation → Model Export
```

### 1. Data Preprocessing
- Stratified 65/15/20 train/validation/test split
- StandardScaler normalization
- Variance threshold filtering

### 2. Feature Selection (7 Methods Compared)
| Method | Type | Description |
|---|---|---|
| CFS | Filter | Correlation-based feature selection |
| Information Gain | Filter | Mutual information with target |
| SFS (Forward/Backward) | Wrapper | Sequential greedy selection |
| ANOVA F-test + RFE | Filter + Wrapper | Statistical + recursive elimination |
| **PSO** | Metaheuristic | Particle Swarm Optimization (30 particles, 50 iterations) |
| **WOA** | Metaheuristic | Whale Optimization Algorithm (30 whales, 50 iterations) |

### 3. Classification Models (10 Total)
ANN (PyTorch, sklearn-wrapped) · SVM · Random Forest · XGBoost · Logistic Regression · Naive Bayes · KNN · Decision Tree · AdaBoost · Gradient Boosting

### 4. Hyperparameter Optimization
Optuna (TPE sampler) tuning across all 10 classifiers, including the neural network's learning rate, weight decay, and epoch count.

### 5. Ensemble Methods
- **Voting Classifier:** ANN + SVM + LR (soft voting) — matches the reference paper's specification
- **Stacking Classifier:** SVM + RF + LR (base) → XGBoost (meta-learner)

### 6. Evaluation & Validation
- Stratified 10-fold cross-validation with SMOTE applied per-fold (no data leakage)
- Classifier × Feature-Selection-Method accuracy matrix (10×7)
- Threshold tuning for optimal F1
- Bootstrap 95% confidence intervals (1000 resamples)
- McNemar's statistical significance test
- Ablation study (component-wise contribution analysis)

### 7. Explainability
SHAP (TreeExplainer) — bar, beeswarm, and waterfall plots identifying which cytological features drive malignancy predictions.

---

## Key Results

| Metric | Value |
|---|---|
| **Best Model** | K-Nearest Neighbors (KNN) |
| **Best CV AUC-ROC** | 0.9968 |
| **Best CV Accuracy** | 97.57% |
| **Best Feature Selection Method** | Sequential Forward Selection (5 features) |
| **Voting Ensemble (ANN+SVM+LR) AUC** | 0.9914 |
| **Stacking Ensemble AUC** | 0.9710 |
| **Optimal Decision Threshold** | 0.41 (vs. default 0.5) |
| **PSO Selected Features** | 6 |
| **WOA Selected Features** | 6 |

Full results, confusion matrices, ROC curves, and the classifier × FS-method heatmap are available in `figures/` and `data/cv_results.csv`.

---

## Repository Structure

```
bc-classification-wbcd/
├── notebooks/
│   └── bc_classification.ipynb       # Full pipeline (34 cells)
├── data/
│   ├── fs_comparison.csv
│   ├── cv_results.csv
│   └── clf_fs_matrix.csv
├── figures/                         # ROC, confusion matrices, SHAP plots, heatmaps
├── src/
│   └── models/                      # Saved models (gitignored — see below)
├── reports/
│   └── daily_logs/
├── requirements.txt
├── README.md
└── .gitignore
```

> Trained model files (`.pkl`, `.pth`, `.onnx`) and generated figures are excluded from version control via `.gitignore` — regenerate them by running the notebook.

---

## Tech Stack

- **Language:** Python 3.12
- **ML/DL:** scikit-learn, XGBoost, PyTorch
- **Optimization:** Optuna, PySwarms (custom PSO/WOA implementations)
- **Imbalance Handling:** imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Environment:** Google Colab (CUDA GPU)

---

## How to Run

1. Open `notebooks/bc_classification.ipynb` in Google Colab
2. Run all cells sequentially — the pipeline auto-installs dependencies, mounts Drive, and loads WBCD directly from UCI (no manual download needed)
3. Outputs (figures, CSVs, trained models) are saved to the configured `BASE_DIR`

```bash
# Or locally:
pip install -r requirements.txt
jupyter notebook notebooks/bc_internship01.ipynb
```

---

## Future Work

- Extend to image-based modalities (mammography via CBIS-DDSM, histopathology via BreakHis) using EfficientNet/U-Net (placeholder branches included in the notebook)
- Multimodal fusion of tabular + imaging features
- Deployment as a REST API for clinical decision support

---

## Reference

Sharma, S., & Mishra, R. (2022). *Breast cancer classification using machine learning techniques*. [Reproduced and extended methodology]
Majidpour, J., et al., 2024 NSGA-II-DL: *Metaheuristic optimal feature selection with deep learning framework for HER2 classification in Breast Cancer. IEEE Access*.

---

## Acknowledgements

Developed under the AICTE SAMARTHAN Internship Connect 2026–27 program, IIIT Bhagalpur, with guidance from Dr. Suneel Kumar, Department of CSE.

---

## License

This project is for academic and research purposes as part of an AICTE-sponsored internship.
