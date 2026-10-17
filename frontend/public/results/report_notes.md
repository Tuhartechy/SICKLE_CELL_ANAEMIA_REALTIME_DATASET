# Project Report Notes: Ensemble Machine Learning Approach for Predicting Pain Severity in Sickle Cell Disease

*Comprehensive technical and analytical notes for the final Major Project Report, academic paper, and viva defense presentation. All numbers, statistical metrics, and benchmark tables reported herein were computed directly from the pilot hospital dataset and cross-validation pipelines.*

---

## 1. Dataset Description & Clinical Cohort

### 1.1 Disease Background & Clinical Problem
Sickle Cell Disease (SCD) is a monogenic red blood cell disorder caused by a point mutation ($\text{Glu6Val}$) in the $\beta$-globin gene (*HBB*). Deoxygenation causes abnormal hemoglobin S ($\text{HbS}$) to polymerize into rigid, elongated filaments, transforming flexible erythrocytes into sickle-shaped cells. These sickled cells occlude capillaries and post-capillary venules, producing **Vaso-Occlusive Crises (VOC)**—episodes of severe tissue ischemia, micro-infarctions, systemic inflammation, and excruciating pain that represent the leading cause of emergency hospitalization.

Predicting VOC pain crisis severity from objective physiological vitals and hematological lab tests—without relying on self-reported pain scores (which are subjective and may be delayed)—provides a potential clinical pathway for early triage and timely analgesia.

### 1.2 Cohort Characteristics
- **Data Source**: Pilot hospital clinical registry of triage/inpatient encounters.
- **Raw Encounter Count**: 40 patients (`001` through `040`), 1 row per patient.
- **Total Variables Collected**: 18 columns.
  - **Identifiers & Target Reference**: `Patient_ID`, `Pain_Score`, `Pain_Severity`.
  - **Demographics (2)**: `Age` (range 4 to 64 years), `Sex` (25 Female, 15 Male).
  - **Genotypes (1)**: `SS` (27), `AS` (8), `Sickle Beta Thalassemia` (2), `Negative` (1), `SF` (1), `Missing` (1).
  - **Vital Signs (6)**: `Heart_Rate`, `Respiratory_Rate`, `SpO2`, `Temperature`, `BP_Systolic`, `BP_Diastolic`.
  - **Laboratory Tests (6)**: `Hemoglobin`, `WBC` (White Blood Cell count), `Platelets`, `RDW` (Red Cell Distribution Width), `Bilirubin_Total`, `Creatinine`.

---

## 2. Preprocessing & Label Derivation Decisions

### 2.1 Feature Missingness Profile (Clean Labelled Cohort, $N=23$)
The 23 patients with validated labels exhibited variable degrees of completeness across diagnostic features:

| Feature | Missing Count ($N=23$) | Missing Percentage (%) | Imputation Strategy in Training Folds |
| :--- | :---: | :---: | :--- |
| `Age` | 0 | 0.0% | None needed |
| `Heart_Rate` | 0 | 0.0% | None needed |
| `Respiratory_Rate` | 0 | 0.0% | None needed |
| `SpO2` | 0 | 0.0% | None needed |
| `Temperature` | 1 | 4.3% | Median |
| `BP_Systolic` | 0 | 0.0% | None needed |
| `BP_Diastolic` | 0 | 0.0% | None needed |
| `Hemoglobin` | 0 | 0.0% | None needed |
| `WBC` | 2 | 8.7% | Median |
| `Platelets` | 3 | 13.0% | Median |
| `RDW` | 5 | 21.7% | Median |
| `Bilirubin_Total` | 6 | 26.1% | Median |
| `Creatinine` | 9 | 39.1% | Median |
| `Sex` | 0 | 0.0% | Most Frequent (Mode) |
| `Genotype` | 0 | 0.0% | Most Frequent (Mode) |

### 2.2 Standardisation & Quality Controls
1. **Genotype Standardisation**: Free-text `"Sickle Beta Thalassemia"` standardized to abbreviation `"SBetaThal"`. Blank genotypes mapped to `"Unknown"`.
2. **Physiological Boundaries**: All observed vitals were verified within physical limits ($\text{SpO}_2 \in [92\%, 100\%]$, Temperature $\in [96.0^\circ\text{F}, 104.0^\circ\text{F}]$). Safety conversion implemented for temperatures $<50^\circ\text{F}$ from Celsius.
3. **Strict Data Leakage Prevention**:
   - `make_preprocessor()` builds an unfitted scikit-learn `ColumnTransformer`. Median imputation, scaling, and one-hot encoding are fit **only within training folds**.
   - `Patient_ID`, `Pain_Score`, and `Pain_Severity` are **strictly excluded** from the feature set `FEATURES`.

### 2.3 Label Derivation Rules & Cohort Partitioning
A 3-class target `Pain_Class` was derived:
- **Priority 1 (Numeric Score Wins)**:
  - $0 \le \text{Score} \le 3 \rightarrow$ **Mild** (Score `0` / "no pain" merged into Mild).
  - $4 \le \text{Score} \le 6 \rightarrow$ **Moderate**.
  - $7 \le \text{Score} \le 10 \rightarrow$ **Severe**.
- **Priority 2 (Text Fallback when Score is Absent)**:
  - *Mild*: `"none"`, `"no pain"`, `"very mild"`, `"mild"`, `"mild tolerable"`.
  - *Moderate*: `"moderate"`, `"moderate intensity"`.
  - *Severe*: `"intense"`, `"severe"`, `"severe pain"`, `"strong pain"`, `"very intense"`, `"distressing"`.
- **Exclusion of Non-Severity Descriptors**:
  - Descriptions of pain type or anatomical location without severity intensity (*"acute"*, *"acute pain"*, *"colicky"*, *"pain in abdomen"*, *"abdominal pain (reported)"*, *"sudden progressive chest pain"*) with no numeric score were left unlabelled.
- **Partitioning Result**:
  - **Clean Labelled Cohort ($N=23$)**: Mild = 8 ($34.8\%$), Moderate = 7 ($30.4\%$), Severe = 8 ($34.8\%$).
  - **Audit Review Cohort ($N=20$)**: 17 unlabelled (10 blank, 7 location/type) + 3 score-vs-text conflicts (Patients `004`, `020`, `034` where numeric score took priority over text).

---

## 3. Models, Settings & Validation Protocol

### 3.1 Candidate Models
1. **Random Forest**: `n_estimators=50, max_depth=3, min_samples_split=3, class_weight='balanced'`.
2. **XGBoost**: `n_estimators=30, max_depth=2, learning_rate=0.05, eval_metric='mlogloss'`.
3. **Support Vector Machine (SVM)**: RBF kernel, $C=1.0$, `class_weight='balanced'`, calibrated probabilities.
4. **Logistic Regression**: $L_2$ penalty, $C=1.0$, `class_weight='balanced'`, `max_iter=1000`.
5. **K-Nearest Neighbors (KNN)**: $k=3$, inverse distance weighting.
6. **Soft Voting Ensemble**: Averages predicted class probabilities across all 5 single pipelines.
7. **Stacking Ensemble**: Meta-classifier (`LogisticRegression(class_weight='balanced')`) learning optimal blend weights from base pipelines via internal 3-fold CV.

### 3.2 Cross-Validation Protocols
- **Primary**: **Repeated Stratified 5-Fold Cross-Validation (5 Repeats = 25 evaluations)** on real patients ($N=23$). Fixed seed `random_state=42`.
- **Secondary**: **Leave-One-Out Cross-Validation (LOOCV = 23 iterations)**.

---

## 4. Baseline Real-Data Experimental Results ($N=23$)

*Evaluated on real held-out patients across 25 repeated stratified folds and LOOCV:*

| Rank | Model | CV Macro F1 (Mean $\pm$ Std) | CV Balanced Acc (Mean $\pm$ Std) | CV Accuracy (Mean $\pm$ Std) | LOOCV Macro F1 | LOOCV Accuracy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Stacking Ensemble** *(Selected Baseline)* | **0.2173 $\pm$ 0.2144** | 0.2667 $\pm$ 0.2449 | **0.2700 $\pm$ 0.2337** | 0.2573 | 0.2609 |
| **2** | **SVM (RBF, Balanced)** | 0.1956 $\pm$ 0.1356 | **0.2800 $\pm$ 0.1869** | 0.2680 $\pm$ 0.1642 | 0.2142 | 0.2174 |
| **3** | **Soft Voting Ensemble** | 0.1938 $\pm$ 0.1453 | 0.2533 $\pm$ 0.2007 | 0.2480 $\pm$ 0.1775 | 0.2000 | 0.2174 |
| **4** | **XGBoost (Depth 2)** | 0.1907 $\pm$ 0.1390 | 0.2400 $\pm$ 0.1769 | 0.2340 $\pm$ 0.1501 | 0.1347 | 0.1304 |
| **5** | **Random Forest (Depth 3)** | 0.1693 $\pm$ 0.1206 | 0.2267 $\pm$ 0.1692 | 0.2320 $\pm$ 0.1624 | 0.2489 | 0.2609 |
| **6** | **Logistic Regression** | 0.1640 $\pm$ 0.1600 | 0.2000 $\pm$ 0.1944 | 0.2020 $\pm$ 0.1873 | **0.2679** | 0.2609 |
| **7** | **KNN ($k=3$)** | 0.1564 $\pm$ 0.1312 | 0.2267 $\pm$ 0.1818 | 0.2180 $\pm$ 0.1708 | 0.0868 | 0.0870 |

---

## 5. Data Augmentation Methodology & Leakage-Free Protocol

### 5.1 Generators Implemented (`augment.py`)
1. **SMOTE**: Oversampling with automatic $k$-neighbors reduction ($\max(1, \min(5, \text{min\_class\_count} - 1))$).
2. **Gaussian Copula Synthesizer (SDV)**: Fitted per target class (`Mild`, `Moderate`, `Severe`) with standard normal marginals.
3. **Noise Jitter**: Balanced resampling with $\sim 5\%$ Gaussian noise injection per continuous column standard deviation.

### 5.2 Strict Anti-Leakage Protocol
- In every cross-validation split, synthetic rows were synthesized **exclusively from the training fold** (~18 patients).
- **Test folds contained REAL hospital patients only**. No synthetic row was ever used for evaluation.
- All synthetic rows are flagged with `is_synthetic = 1`.

### 5.3 Synthetic Data Quality Audit ($N=10,000$ Final SMOTE Cohort)
- **Exact Duplicates of Real Rows**: **0** (Pass criteria: 0).
- **Clinical Range Violations**: **0** (All numeric features strictly within physical bounds).
- **Class Balance**: $33.3\%$ Mild, $33.3\%$ Moderate, $33.3\%$ Severe.
- **Frobenius Norm Correlation Difference**: $0.8752$.
- **Mean Absolute Correlation Difference**: $0.0503$.
- **Median Distance to Real Training Row**: $1.3987$ (Standardized Euclidean space).

---

## 6. Real vs. Augmented Performance Comparison

*Evaluated on real held-out patients across 25 repeated stratified folds:*

| Model | Real-Only Macro F1 ($\pm \text{Std}$) | Best Augmented Macro F1 ($\pm \text{Std}$) | Difference | Best Generator | Best Synthetic Size | Gain Exceeds Real Std? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** | 0.1564 $\pm$ 0.1312 | **0.2732 $\pm$ 0.1337** | **+0.1168** | SMOTE | 2,000 rows | False |
| **Random Forest** | 0.1693 $\pm$ 0.1206 | **0.2578 $\pm$ 0.1595** | **+0.0884** | SMOTE | 5,000 rows | False |
| **XGBoost** | 0.1907 $\pm$ 0.1390 | **0.2450 $\pm$ 0.1772** | **+0.0543** | SMOTE | 10,000 rows | False |
| **Stacking Ensemble** | 0.2173 $\pm$ 0.2144 | **0.2351 $\pm$ 0.1599** | **+0.0178** | Noise Jitter | 500 rows | False |
| **Logistic Regression** | 0.1640 $\pm$ 0.1600 | **0.2196 $\pm$ 0.1527** | **+0.0556** | SMOTE | 5,000 rows | False |
| **Soft Voting Ensemble**| 0.1938 $\pm$ 0.1453 | **0.2036 $\pm$ 0.1777** | **+0.0098** | Noise Jitter | 500 rows | False |
| **SVM (RBF)** | 0.1956 $\pm$ 0.1356 | **0.1981 $\pm$ 0.1183** | **+0.0025** | Gaussian Copula| 500 rows | False |

### 6.1 Secondary Check: Synthetic 80/20 Benchmark
- When models were evaluated on an 80/20 train/test split of purely synthetic rows, accuracy jumped to $94\% - 100\%$.
- **Methodological Warning**: This occurs because synthetic rows derived from the same 23 patient clusters appear in both training and test sets. This highlights why evaluating strictly on **real, held-out hospital patients** is the only scientifically valid method.

---

## 7. Model Selection & Synthesis

1. **Real-Only Baseline Winner**:
   - **Stacking Ensemble** (combining RF, XGBoost, SVM, LR, and KNN with a balanced Logistic Regression meta-learner).
   - Saved to `models/best_model.joblib` (Macro F1 = $0.2173 \pm 0.2144$).
2. **Augmented Winner**:
   - **KNN ($k=3$, distance)** trained with **SMOTE 2,000 synthetic rows**.
   - Saved to `models/best_model_augmented.joblib` (Macro F1 on real patients = $0.2732 \pm 0.1337$).
3. **Scientific Appraisal**:
   - While SMOTE augmentation nominally improved KNN (+0.1168) and RF (+0.0884), the observed difference between the best overall augmented setup (0.2732) and the real baseline (0.2173) is $+0.0559$, which is **smaller than the cross-validation standard deviation ($\pm 0.2144$)**.
   - Generating 5,000–10,000 synthetic rows from 18 real training patients regularizes decision boundaries for sparse estimators, but **adds no new biological information about the wider sickle cell population**.

---

## 8. Clinical & Pilot Study Limitations

1. **Fundamental Sample Size Constraint ($N=23$)**: All synthetic rows derive from approximately 18 training patients per fold. Wide cross-validation confidence intervals reflect this statistical reality.
2. **Age Heterogeneity**: Pediatric patients (down to age 4) and older adults (up to 64 years) were pooled, even though baseline physiological norms (resting heart rate, respiratory rate, blood pressure) vary substantially by age group.
3. **Secondary Laboratory Missingness**: Serum Creatinine ($39.1\%$) and Total Bilirubin ($26.1\%$) required median imputation due to inconsistent hospital lab ordering.
4. **Subjective Ground Truth**: Target severity labels derive from visual analog pain scores (0–10) and clinician notes, which reflect subjective pain perception and opioid tolerance.

---

## 9. Future Work & Clinical Roadmap

1. **Multi-Center Cohort Expansion**: Collect $\ge 500$ prospective encounters across emergency departments.
2. **Pediatric vs. Adult Stratification**: Train distinct age-stratified models or normalize vitals to age-adjusted z-scores (Pediatric Early Warning Score, PEWS).
3. **Biomarker Enrichment**: Integrate critical crisis pathophysiology markers:
   - Serum Lactate Dehydrogenase (LDH) and Free Hemoglobin (intravascular hemolysis).
   - Fetal Hemoglobin percentage ($\text{HbF}\%$) and Reticulocyte count.
   - High-sensitivity C-Reactive Protein (hs-CRP) for inflammatory tracking.
4. **Longitudinal Triage Trajectory**: Track vital signs and pain trajectory dynamically over the first 2 to 6 hours of hospital triage rather than relying on a single cross-sectional snapshot.
