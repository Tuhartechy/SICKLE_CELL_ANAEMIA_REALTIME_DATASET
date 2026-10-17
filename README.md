# Ensemble Machine Learning Approach for Predicting Pain Severity in Sickle Cell Disease

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)]()
[![Framework](https://img.shields.io/badge/Framework-scikit--learn%20%7C%20xgboost%20%7C%20streamlit-orange.svg)]()

> **College Major Project & Clinical Pilot Feasibility Study**  
> *A pilot investigation utilizing real hospital clinical data ($N=40$ raw patients, $N=23$ supervised labelled patients) to evaluate baseline machine learning models and ensemble architectures for predicting pain severity classes without target leakage.*

---

## 📁 Project Directory Structure

```text
sickle-pain/
├── assets/
│   ├── sickle_pain_scale.jpg   # Clinical pain severity scaling infographic
│   └── sickle_circulation.jpg  # Normal vs. sickled RBC microvascular diagram
├── data/
│   ├── patients.csv            # Original raw hospital dataset (read-only)
│   ├── patients_clean.csv      # Labelled dataset (N=23) with derived Pain_Class
│   └── rows_for_review.csv     # Flagged rows (N=20: 17 unlabelled + 3 conflicts)
├── models/
│   ├── best_model.joblib       # Serialized winning pipeline (Stacking Ensemble)
│   └── metadata.json           # Model configuration, feature names, medians & modes
├── results/
│   ├── model_comparison.csv    # Cross-validation & LOOCV benchmark metrics
│   ├── macro_f1_comparison.png # Macro F1 bar chart with error bars
│   ├── confusion_matrix_stacking_ensemble.png  # Out-of-fold confusion matrix
│   ├── feature_importance.png  # Random Forest Gini + Permutation Importance
│   └── report_notes.md         # Comprehensive technical and analytical report notes
├── app.py                      # Streamlit interactive clinical frontend
├── preprocess.py               # Data cleaning, range checks, target mapping & pipeline
├── train.py                    # Model training, cross-validation, LOOCV & serialization
├── requirements.txt            # Project dependencies
└── README.md                   # Setup guide and documentation
```

---

## ⚙️ Environment Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows PowerShell, macOS terminal, or Linux bash

### 2. Create Virtual Environment & Install Dependencies
Open a terminal in the project directory:

```bash
# Navigate to the project root
cd sickle-pain

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Preprocessing & Data Validation
Runs data cleaning, clinical range verification, Genotype and Sex standardization, target label mapping (`Pain_Class`), and exports clean and review CSV files.

```bash
python preprocess.py
```
- **Outputs**:
  - `data/patients_clean.csv`: 23 labelled patients ($8$ Mild, $7$ Moderate, $8$ Severe).
  - `data/rows_for_review.csv`: 20 patients flagged for audit (17 unlabelled due to lack of score or non-severity descriptions + 3 score-vs-text conflicts).
  - Terminal audit report showing feature missingness, corrections, and distributions.

---

### Step 2: Model Training & Cross-Validation
Trains 5 baseline classifiers (Random Forest, XGBoost, SVM, Logistic Regression, KNN) and 2 ensemble architectures (Soft Voting and Stacking with Logistic Regression meta-learner).

```bash
python train.py
```
- **Strict Anti-Leakage Design**: Every model is encapsulated in a scikit-learn `Pipeline` utilizing `make_preprocessor()`. Median imputation, standard scaling, and one-hot encoding are fitted **exclusively within training folds**.
- **Validation**: Repeated Stratified 5-Fold Cross-Validation (5 Repeats = 25 evaluations) + Leave-One-Out Cross-Validation (LOOCV = 23 iterations).
- **Outputs**:
  - `results/model_comparison.csv`
  - `results/macro_f1_comparison.png`
  - `results/confusion_matrix_stacking_ensemble.png`
  - `results/feature_importance.png`
  - `models/best_model.joblib`
  - `models/metadata.json`

---

### Step 3: Launch Streamlit Web Application
Starts the interactive clinical demonstration interface.

```bash
streamlit run app.py
```
- **Features in the App**:
  - **Tab 1 ("Predict")**: Input patient demographics, vitals, and lab values. Any lab value can be left blank to demonstrate automated train-fold median imputation. Includes one-click presets for Acute Crisis and Stable profiles. Displays predicted pain severity and probability distributions.
  - **Tab 2 ("Model Comparison")**: Live benchmarks table with metric highlighting and embedded high-resolution evaluation figures.
  - **Tab 3 ("About")**: Full documentation of study objectives, label rules, clinical limitations, and research disclaimer.

---

## 📊 Summary of Experimental Results ($N=23$)

| Model | CV Macro F1 ($\pm \text{Std}$) | CV Balanced Acc | CV Accuracy | LOOCV Macro F1 | LOOCV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble** *(Best)* | **0.2173 $\pm$ 0.2144** | 0.2667 | 0.2700 | 0.2573 | 0.2609 |
| **SVM (RBF)** | 0.1956 $\pm$ 0.1356 | **0.2800** | 0.2680 | 0.2142 | 0.2174 |
| **Soft Voting Ensemble** | 0.1938 $\pm$ 0.1453 | 0.2533 | 0.2480 | 0.2000 | 0.2174 |
| **XGBoost** | 0.1907 $\pm$ 0.1390 | 0.2400 | 0.2340 | 0.1347 | 0.1304 |
| **Random Forest** | 0.1693 $\pm$ 0.1206 | 0.2267 | 0.2320 | 0.2489 | 0.2609 |
| **Logistic Regression** | 0.1640 $\pm$ 0.1600 | 0.2000 | 0.2020 | **0.2679** | 0.2609 |
| **KNN ($k=3$)** | 0.1564 $\pm$ 0.1312 | 0.2267 | 0.2180 | 0.0868 | 0.0870 |

*Note: In accordance with scientific integrity standards, standard deviations are explicitly published. The high variance directly reflects the statistical properties of evaluating a 3-class problem on a small pilot sample ($N=23$).*

---

## 🛡️ Anti-Leakage Protocol
1. `Patient_ID`, `Pain_Score`, and `Pain_Severity` are **never** used as predictive features.
2. Imputation and scaling transformers are never fitted globally; all preprocessing steps are encapsulated inside scikit-learn cross-validation pipelines.

---

## ⚠️ Academic & Clinical Disclaimer
This project is an **academic research prototype** and educational pilot study. It is **NOT** a certified medical device, diagnostic system, or clinical decision tool. It must **NOT** be used to guide, triage, or alter patient management.
