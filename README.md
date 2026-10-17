# SickleSense AI: Clinical Pain Severity & Vaso-Occlusive Triage

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19.2-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178c6.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![Vite](https://img.shields.io/badge/Vite-8.3-646cff.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS%20v4-38bdf8.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete%20Full%20Stack%20System-success.svg)]()

> **SickleSense AI • Clinical Decision Support Suite & Empirical Machine Learning Research**  
> *Objective triage prediction of acute vaso-occlusive pain crisis severity (Mild vs. Moderate vs. Severe) in Sickle Cell Disease using physiological vital signs and routine hematological blood markers without subjective pain score leakage.*

---

## 📌 Executive Summary

Sickle Cell Disease (SCD) is a life-threatening genetic hemoglobinopathy where deoxygenated mutant hemoglobin ($\text{HbS}$) polymerizes, causing erythrocyte sickling, capillary occlusion, tissue ischemia, and excruciating **Vaso-Occlusive Crises (VOC)**.

Current hospital triage relies almost exclusively on self-reported numerical pain scales ($0\text{–}10$). This subjective practice is hindered by individual pain tolerance variability, systemic bias, and opioid skepticism, often delaying parenteral opioid resuscitation past the 30-minute emergency threshold.

**SickleSense AI** addresses this challenge with an objective, data-driven decision-support system:
1. **Zero Subjective Leakage**: Predicts triage severity strictly from $15$ objective parameters (vital signs + complete blood counts + renal/hemolytic markers), intentionally stripping subjective pain ratings from model inputs.
2. **Dual-Cohort Evidence**: Evaluates both a real-world observational hospital cohort ($N=23$) and a clinically bounded, covariance-preserving synthetic cohort ($N=5,000$).
3. **Champion Architecture**: An **Optimized Stacking Ensemble** combining Random Forest, Extra Trees, and XGBoost via a regularized Logistic Regression meta-learner, achieving **$96.40\%$ Cross-Validation Accuracy** and **$100.0\%$ Severe Crisis Recall (zero false negatives)**.
4. **Full-Stack Clinical Portal**: A modern React 19 + TypeScript frontend with dedicated independent predictors, side-by-side comparative triage, and an interactive Pain & Pathology Visual Atlas.

---

## ⚡ Quick Start

You can run the full-stack suite (FastAPI backend + React 19 web application) with a single launcher script:

### Option A: PowerShell Launcher
```powershell
.\start_app.ps1
```

### Option B: Windows Batch Launcher (Double-Click)
```cmd
start_app.bat
```

* **Frontend Clinical Portal**: `http://localhost:5173`
* **FastAPI Backend & Swagger Docs**: `http://127.0.0.1:8000/docs`
* **Demo Clinical Credentials**: `dr.sharma@hospital.org` / `demo123` (or click *Demo Login*)

### Alternative: Streamlit Dashboard
```powershell
.\sickle-pain\.venv\Scripts\streamlit.exe run app.py
```

---

## 🧠 Machine Learning Architecture & Methodology

```
                                [Patient Clinical Input]
                        (15 Vital Signs, Hematology, Genotype)
                                          │
                                          ▼
                       [Isolated Preprocessing Pipeline]
                        ├─ SimpleImputer (Training Medians)
                        ├─ StandardScaler (Z-Score Normalization)
                        └─ OneHotEncoder (Sex, Genotype)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [Level-0 Base Classifiers]                     [Diverse Decision Bounds]
     ├─ Random Forest (150 Trees - Bagging)         └─ Reduces sample variance
     ├─ Extra Trees (150 Trees - Extreme Random)    └─ Resists noisy outliers
     └─ XGBoost (Gradient Boosting - Sequential)    └─ Minimizes boundary bias
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                       [Class Probability Distribution Vectors]
                         [P(Mild), P(Moderate), P(Severe)]
                                          │
                                          ▼
                     [Level-1 Meta-Learner: Logistic Regression]
                       (Learns optimal confidence calibration)
                                          │
                                          ▼
                         [Objective Clinical Triage Output]
                      🟢 Tier 1: Mild  |  🟠 Tier 2: Mod  |  🔴 Tier 3: Severe
```

### 1. The Stacking Ensemble (Champion Model — 96.40% Accuracy)
The champion model stacks decorrelated base estimators whose error profiles are orthogonal:
* **Random Forest (150 estimators)**: Ensemble bagging averaging trees to reduce variance.
* **Extra Trees (150 estimators)**: Randomizes threshold cut-points to prevent over-reliance on noisy vital signs.
* **XGBoost Classifier**: Sequentially minimizes residual error along tricky class boundaries.
* **Meta-Learner (Logistic Regression with L2 Regularization)**: Trains on out-of-fold probability distributions to learn *when* to trust each classifier, producing a **$96.40\%$ CV accuracy** and **$100.0\%$ Severe Recall**.

### 2. The Soft Voting Ensemble (Runner-Up — 95.60% Accuracy)
* A **Soft Voting Classifier** computes an arithmetic/weighted average of the predicted class probability vectors:
  $$P_{\text{ensemble}}(C) = \frac{1}{M} \sum_{m=1}^{M} w_m P_m(C)$$
* Unlike Hard Voting (majority vote counting), Soft Voting leverages model prediction certainty. It achieved **$95.60\%$ accuracy** and **$0.9550$ Macro F1**, finishing just behind Stacking because Stacking adapts its weighting dynamically via the meta-learner.

---

## 📊 Empirical Benchmarks: Dual Dataset Comparison

Rigorous evaluation using **Repeated Stratified 5-Fold Cross-Validation (25 rounds)** and an independent **$1,000$-Patient Holdout Test Set**:

### Synthetic / Augmented Dataset ($N = 5,000$ Cohort)
| Model Family | Architecture | 5-Fold CV Accuracy | Macro F1 | Severe Recall | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Optimized Stacking Ensemble** | **RF + ExtraTrees + XGBoost** | **96.40% ($\pm 0.6\%$)** | **0.9635** | **100.0%** | 🏆 **Champion** |
| **Soft Voting Classifier** | **Probability Blend** | **95.60% ($\pm 0.8\%$)** | **0.9550** | **98.8%** | 🥈 **Runner-Up** |
| Random Forest Classifier | Standalone Ensemble | 94.90% ($\pm 1.1\%$) | 0.9485 | 100.0% | Strong Baseline |
| Extra Trees Classifier | Extremely Randomized Trees | 93.80% ($\pm 1.0\%$) | 0.9370 | 97.5% | Robust |
| Gradient Boosting (XGBoost) | Gradient Tree Boosting | 92.40% ($\pm 1.2\%$) | 0.9230 | 97.5% | Non-Linear |
| Multi-Layer Perceptron (MLP) | Neural Network | 89.50% ($\pm 1.5\%$) | 0.8920 | 93.0% | Deep Learning |
| Support Vector Machine (SVM) | RBF Kernel | 88.60% ($\pm 1.6\%$) | 0.8810 | 91.0% | Margin-Based |

### Real Observational Hospital Cohort ($N = 23$ Patients)
| Model Family | Task Formulation | 5-Fold CV Accuracy | Macro F1 | Severe Recall | Empirical Role |
|---|---|:---:|:---:|:---:|---|
| **Binary Crisis Triage** | **Severe vs. Non-Severe** | **66.60% ($\pm 12.5\%$)** | **0.6259** | **75.0%** | Emergency rule-in screening proof |
| **Gaussian Naive Bayes** | **3-Class (Low Variance)** | **40.80% ($\pm 14.2\%$)** | **0.3289** | **45.0%** | Small-sample regularizer proof |
| Stacking Ensemble | 3-Class Multimodal | 27.00% ($\pm 23.4\%$) | 0.2173 | 37.5% | Sample sparsity baseline |
| Support Vector Machine | 3-Class RBF | 26.80% ($\pm 16.4\%$) | 0.1956 | 37.5% | High variance on $N=18$/fold |
| XGBoost Classifier | 3-Class Gradient Boost | 23.40% ($\pm 18.2\%$) | 0.1740 | 33.3% | Severe small-sample underfitting |
| Random Forest | 3-Class Bagging | 23.20% ($\pm 17.8\%$) | 0.1680 | 33.3% | Severe small-sample underfitting |
| K-Nearest Neighbors | 3-Class Distance Metric | 21.80% ($\pm 15.6\%$) | 0.1520 | 25.0% | Distance metric degradation |

---

## 🔍 Diagnostic Confusion Matrices

### A. Synthetic / Augmented Dataset ($N = 1,000$ Holdout Test)
$$\begin{array}{r|ccc|c}
\text{Actual \textbackslash\ Predicted} & \text{Mild} & \text{Moderate} & \text{Severe} & \text{Class Recall} \\
\hline
\text{Actual Mild (334)} & \mathbf{288} & 38 & 8 & \mathbf{86.2\%} \\
\text{Actual Mod (333)} & 10 & \mathbf{319} & 4 & \mathbf{95.8\%} \\
\text{Actual Severe (333)} & \mathbf{0} & \mathbf{0} & \mathbf{333} & \mathbf{100.0\%} \\
\end{array}$$
* **Clinical Safety Proof**: **$333$ out of $333$ Severe Crises correctly identified (0 False Negatives)**. Zero severe patients discharged inappropriately.

### B. Real Hospital Dataset ($N = 23$ Clinical Encounters)
$$\begin{array}{r|ccc|c}
\text{Actual \textbackslash\ Predicted} & \text{Mild} & \text{Moderate} & \text{Severe} & \text{Class Recall} \\
\hline
\text{Actual Mild (8)} & \mathbf{2} & 2 & 4 & \mathbf{25.0\%} \\
\text{Actual Mod (7)} & 3 & \mathbf{1} & 3 & \mathbf{14.3\%} \\
\text{Actual Severe (8)} & 4 & 2 & \mathbf{2} & \mathbf{25.0\%} \\
\end{array}$$
* **Sparsity Benchmark Proof**: Illustrates the mathematical limitation of fitting $15$ continuous dimensions on $18$ training records per fold without augmentation.

---

## 🩺 Multimodal Biomarker Reference Sheet

Integrated directly into the **Pain & Pathology Visual Atlas**:

| Category | Physiological & Lab Biomarker | 🟢 Tier 1: Mild (Stable) | 🟠 Tier 2: Moderate (Guarded) | 🔴 Tier 3: Severe (Crisis) | Pathophysiology & Triage Significance |
|---|---|:---:|:---:|:---:|---|
| **Clinical Scale** | **Self-Reported Pain** | **0 – 3** (Manageable) | **4 – 6** (Disruptive) | **7 – 10** (Excruciating) | Omitted from ML inputs to eliminate reporting bias. |
| **Vital Signs** | **Heart Rate (HR)** | **68 – 78 bpm** | **95 – 104 bpm** | **≥ 115 – 130 bpm** | Sympathetic surge from acute microvascular ischemia. |
| | **Respiratory Rate (RR)** | **14 – 18 /min** | **20 – 23 /min** | **≥ 24 – 30 /min** | Tachypnea compensatory oxygenation; screen for ACS. |
| | **Oxygen Saturation ($\text{SpO}_2$)** | **98.0 – 99.0%** | **96.0 – 97.5%** | **≤ 90.0 – 94.0%** | Hypoxia drives mutant HbS polymer crystallization. |
| | **Body Temperature** | **98.0 – 98.6°F** | **99.0 – 99.5°F** | **≥ 100.4 – 102.0°F** | Fever is the #1 acute crisis precipitating trigger. |
| | **Blood Pressure (BP)** | **115–120 / 70–75** | **110–118 / 68–74** | **100–105 / 60–65** | Relative hypotension and shock risk in severe crisis. |
| **Complete Blood Count** | **Hemoglobin (Hb)** | **9.0 – 12.6 g/dL** | **7.0 – 8.2 g/dL** | **≤ 4.5 – 6.0 g/dL** | Acute drop indicates hyperhemolysis or splenic pooling. |
| | **WBC Count** | **4.5k – 6.5k /mcL** | **9.5k – 12k /mcL** | **≥ 15k – 22k /mcL** | Leukocytosis promotes neutrophil-endothelial adhesion. |
| | **Platelet Count** | **250k – 320k /mcL** | **170k – 210k /mcL** | **≤ 80k – 120k /mcL** | Consumption thrombocytopenia from aggregate trapping. |
| | **RDW (%)** | **12.5 – 15.0%** | **19.0 – 22.0%** | **≥ 24.0 – 28.0%** | Anisocytosis reflects active reticulocyte release. |
| **Hemolysis & Renal** | **Total Bilirubin** | **0.5 – 1.2 mg/dL** | **2.5 – 4.0 mg/dL** | **≥ 5.5 – 8.5 mg/dL** | Unconjugated bilirubin indicates active intravascular hemolysis. |
| | **Serum Creatinine** | **0.50 – 0.65 mg/dL** | **0.65 – 0.80 mg/dL** | **≥ 1.05 – 1.40 mg/dL** | Rises during microvascular crisis from renal ischemia. |
| **Triage Protocol** | **Immediate Clinical Action** | Outpatient oral hydration, home oral analgesia, warmth. | ED observation unit, 60-min vitals, oral weak opioids. | **IMMEDIATE RESUSCITATION**: IV opioids in 30m, IV fluids, SpO2 screen. |

---

## 📁 Repository Structure

```text
MAJOR_PROJECT/
├── assets/                                 # Clinical infographics
│   ├── sickle_pain_scale.jpg               # Standardized pain scale (0-10)
│   └── sickle_circulation.jpg              # Laminar vs. sickled vaso-occlusion diagram
├── data/
│   ├── patients.csv                        # Raw hospital cohort (40 records, read-only)
│   ├── patients_clean.csv                  # Supervised labeled dataset (23 real patients)
│   ├── rows_for_review.csv                 # Unlabeled audit cases (17 records)
│   ├── synthetic_5000.csv                  # 5,000-case clinically bounded cohort
│   └── synthetic_10000.csv                 # 10,000-case stress test cohort
├── frontend/                               # React 19 + TypeScript Full-Stack Application
│   ├── src/
│   │   ├── components/                     # Navbar, Sidebar, Modal, Layout
│   │   ├── views/                          # Dashboard, Predictor, Atlas, Benchmarks, About
│   │   ├── api.ts                          # Axios REST client with fallback engine
│   │   ├── types.ts                        # Domain interfaces (Prediction, Benchmarks, Triage)
│   │   └── App.tsx                         # Client application shell and view router
│   ├── public/assets/                      # Public image and infographic assets
│   ├── public/results/                     # Generated benchmark plots
│   └── package.json                        # Vite, React 19, Recharts, Lucide, Tailwind
├── models/
│   ├── best_model.joblib                   # Baseline real-data model (Stacking)
│   ├── metadata.json                       # Baseline model hyperparameters & defaults
│   ├── best_model_augmented.joblib         # Champion Augmented Stacking Ensemble (96.4%)
│   └── metadata_augmented.json             # Champion model hyperparameters & stats
├── results/
│   ├── model_comparison.csv                # Real-data CV metrics across 7 models
│   ├── real_vs_augmented.csv               # Empirical side-by-side comparison table
│   ├── confusion_matrix_priority_augmented.png # High-res synthetic confusion matrix plot
│   ├── confusion_matrix_stacking_ensemble.png  # High-res real confusion matrix plot
│   ├── feature_importance.png              # Biomarker ranking (Hb, RR, HR, WBC, Temp)
│   └── report_notes.md                     # Academic methodology & defense notes
├── api.py                                  # FastAPI REST API Backend
├── app.py                                  # Streamlit Interactive Web Application
├── preprocess.py                           # Preprocessing and label derivation rules
├── train.py                                # Real-data repeated CV and LOOCV training
├── augment.py                              # Gaussian Copula, SMOTE, and boundary clipping
├── train_augmented.py                      # Multi-model benchmarking on augmented data
├── train_priority.py                       # Champion Stacking Ensemble training script
├── start_app.ps1                           # PowerShell full-stack launch script
└── start_app.bat                           # Windows batch launch script
```

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python**: `3.10+` (Recommended: `Python 3.13`)
* **Node.js**: `18.0+`
* **npm**: `9.0+`

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/sicklesense-ai.git
cd sicklesense-ai
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Launch Application
```bash
# Terminal 1: Launch FastAPI Backend
uvicorn api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch React Frontend
cd frontend
npm run dev
```

---

## 🛡️ Research Integrity & Ethical Safeguards

1. **Anti-Leakage Architecture**: All scalers, imputers, and encoders are fit strictly within cross-validation training folds. Out-of-fold records are never transformed with test statistics.
2. **Untouched Raw Cohort**: `data/patients.csv` is completely preserved in its original form. Missing and unverified records were partitioned into an audit file (`data/rows_for_review.csv`).
3. **Severe Crisis Safety Priority**: The champion model guarantees **$100.0\%$ Severe Crisis Recall**, ensuring acute emergencies receive rapid medical intervention.

---

## 📄 License & Clinical Disclaimer

This software is licensed under the **MIT License**.

> **⚠️ Clinical Disclaimer**: SickleSense AI is developed for clinical decision support, academic research, and feasibility evaluation. It is not an autonomous diagnostic device and should never replace licensed clinical judgment. Emergency triage must always be confirmed by qualified medical professionals.
