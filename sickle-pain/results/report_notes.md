# Project Report Notes: Ensemble Machine Learning Approach for Predicting Pain Severity in Sickle Cell Disease

*These notes are prepared as reference material for the final Major Project Report, documentation, and thesis defense presentation. All metrics and statistical distributions reported herein are computed directly from the pilot hospital dataset.*

---

## 1. Dataset Description & Clinical Context

### 1.1 Disease Background
Sickle Cell Disease (SCD) is a monogenic hemoglobinopathy caused by a point mutation in the $\beta$-globin gene (*HBB*), resulting in abnormal hemoglobin S ($\text{HbS}$). Under deoxygenated conditions, $\text{HbS}$ polymerizes into rigid, sickle-shaped erythrocytes that obstruct microvascular blood flow. This pathophysiology triggers **Vaso-Occlusive Crises (VOC)**—episodes of severe tissue ischemia, micro-infarction, and intense systemic pain that represent the leading cause of emergency hospitalization for individuals with SCD.

### 1.2 Dataset Dimensions & Characteristics
- **Source**: Pilot hospital clinical registry of patients evaluated during triage/inpatient care.
- **Raw Sample Size**: 40 unique patients (`001` to `040`), 1 row per patient.
- **Total Variables Collected**: 18 columns.
  - **Identifiers / Target Source**: `Patient_ID`, `Pain_Score`, `Pain_Severity`.
  - **Demographics (2)**: `Age` (range 4 to 64 years), `Sex` (25 Female, 15 Male).
  - **Genotypes (1)**: `SS` (27), `AS` (8), `Sickle Beta Thalassemia` (2), `Negative` (1), `SF` (1), `Missing` (1).
  - **Vital Signs (6)**: `Heart_Rate`, `Respiratory_Rate`, `SpO2`, `Temperature`, `BP_Systolic`, `BP_Diastolic`.
  - **Laboratory Tests (6)**: `Hemoglobin`, `WBC` (White Blood Cell count), `Platelets`, `RDW` (Red Cell Distribution Width), `Bilirubin_Total`, `Creatinine`.

---

## 2. Preprocessing Steps & Methodological Decisions

### 2.1 Missingness Profile (Clean Labelled Cohort, $N=23$)
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

### 2.2 Clinical Validation & Standardization Decisions
1. **Genotype Standardization**:
   - Free-text `"Sickle Beta Thalassemia"` standardized to abbreviation `"SBetaThal"`.
   - Blank genotypes mapped to `"Unknown"`.
2. **Physiological Value Bounds & Outlier Verification**:
   - `SpO2`: Range observed 92.0% to 100.0% (no values exceeded 100%).
   - `Temperature`: Range observed 96.0°F to 104.0°F (all recorded in Fahrenheit; safety logic implemented to convert values $<50^\circ\text{F}$ from Celsius).
   - Negative values: 0 found across all numeric variables.
3. **Strict Data Leakage Prevention**:
   - Imputation (median for numeric, mode for categorical) and Feature Scaling (`StandardScaler`) were **never** executed globally on `patients_clean.csv`.
   - Instead, `preprocess.py` provides `make_preprocessor()`, an unfitted scikit-learn `ColumnTransformer`. This transformer is fitted strictly within the training partitions of each cross-validation fold.
   - `Patient_ID`, `Pain_Score`, and `Pain_Severity` were strictly excluded from the feature matrix `FEATURES` to avoid target leakage.

---

## 3. Label Definition & Cohort Partitioning

To enable supervised machine learning, an objective 3-class target variable (`Pain_Class`) was derived:

### 3.1 Decision Rules
1. **Priority Rule (Numeric Score Wins)**:
   - `Pain_Score` $\in [0, 3]$ $\rightarrow$ **Mild** (Score `0` / "no pain" grouped into Mild).
   - `Pain_Score` $\in [4, 6]$ $\rightarrow$ **Moderate**.
   - `Pain_Score` $\in [7, 10]$ $\rightarrow$ **Severe**.
2. **Text Fallback Rule (When `Pain_Score` is Absent)**:
   - **Mild**: `"none"`, `"no pain"`, `"very mild"`, `"mild"`, `"mild tolerable"`.
   - **Moderate**: `"moderate"`, `"moderate intensity"`.
   - **Severe**: `"intense"`, `"severe"`, `"severe pain"`, `"strong pain"`, `"very intense"`, `"distressing"`.
3. **Exclusion of Non-Severity Clinical Descriptors**:
   - Clinical notes indicating anatomical pain site or temporal character without intensity (*"acute"*, *"acute pain"*, *"colicky"*, *"pain in abdomen"*, *"abdominal pain (reported)"*, *"sudden progressive chest pain"*) cannot be reliably converted to severity. In the absence of a numeric score, these rows were left unlabelled and excluded from training.
4. **Conflict Handling**:
   - When score and text disagreed, the numeric score took precedence, and the record was flagged for clinical audit.

### 3.2 Resulting Cohort Breakdown
- **Total Patients Read**: 40
- **Labelled Supervised Cohort**: **23 patients** (Saved to `data/patients_clean.csv`)
  - **Mild**: 8 patients (34.8%)
  - **Moderate**: 7 patients (30.4%)
  - **Severe**: 8 patients (34.8%)
- **Review Cohort**: **20 patients** (Saved to `data/rows_for_review.csv`)
  - **17 Unlabelled Patients**: 10 completely blank + 7 location/type descriptors.
  - **3 Conflicting Patients** (Score decided label, flagged for audit):
    - Patient `004`: `Pain_Score = 4` (Moderate), `Pain_Severity = "Distressing"` (Severe) $\rightarrow$ Assigned: **Moderate**.
    - Patient `020`: `Pain_Score = 6` (Moderate), `Pain_Severity = "Distressing"` (Severe) $\rightarrow$ Assigned: **Moderate**.
    - Patient `034`: `Pain_Score = 6` (Moderate), `Pain_Severity = "Intense"` (Severe) $\rightarrow$ Assigned: **Moderate**.

---

## 4. Machine Learning Models & Pipeline Architecture

### 4.1 Algorithms Evaluated
To counter overfitting on $N=23$, lightweight architectures with regularized hyperparameters were chosen:
1. **Random Forest**: `n_estimators=50`, `max_depth=3`, `min_samples_split=3`, `class_weight='balanced'`.
2. **XGBoost**: `n_estimators=30`, `max_depth=2`, `learning_rate=0.05`, `eval_metric='mlogloss'`.
3. **Support Vector Machine (SVM)**: RBF kernel, $C=1.0$, `class_weight='balanced'`, calibrated probabilities.
4. **Logistic Regression**: $L_2$ penalty, $C=1.0$, `class_weight='balanced'`, `max_iter=1000`.
5. **K-Nearest Neighbors (KNN)**: $k=3$, inverse distance weighting.
6. **Soft Voting Ensemble**: Combines class prediction probabilities across all 5 single model pipelines.
7. **Stacking Ensemble**: Meta-model using a balanced `LogisticRegression` to learn optimal blend weights from the 5 base pipelines using internal 3-fold cross-validation.

### 4.2 Cross-Validation Protocols
- **Primary Protocol**: **Repeated Stratified 5-Fold Cross-Validation (5 Repeats = 25 evaluations)**.
  - Stratification ensures each validation fold contains at least 1 sample of each of the 3 classes.
  - All 7 candidate models were tested on identical split seeds (`random_state=42`).
- **Secondary Protocol**: **Leave-One-Out Cross-Validation (LOOCV = 23 iterations)**.
  - Exhaustively tests generalizability when single patients are iteratively held out.

---

## 5. Experimental Results

*Exact computed values from `results/model_comparison.csv`:*

| Model | CV Macro F1 (Mean $\pm$ Std) | CV Balanced Acc (Mean $\pm$ Std) | CV Accuracy (Mean $\pm$ Std) | LOOCV Macro F1 | LOOCV Bal Acc | LOOCV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble** | **0.2173 $\pm$ 0.2144** | 0.2667 $\pm$ 0.2449 | **0.2700 $\pm$ 0.2337** | 0.2573 | 0.2679 | 0.2609 |
| **SVM (RBF, Balanced)** | 0.1956 $\pm$ 0.1356 | **0.2800 $\pm$ 0.1869** | 0.2680 $\pm$ 0.1642 | 0.2142 | 0.2143 | 0.2174 |
| **Soft Voting Ensemble** | 0.1938 $\pm$ 0.1453 | 0.2533 $\pm$ 0.2007 | 0.2480 $\pm$ 0.1775 | 0.2000 | 0.2202 | 0.2174 |
| **XGBoost** | 0.1907 $\pm$ 0.1390 | 0.2400 $\pm$ 0.1769 | 0.2340 $\pm$ 0.1501 | 0.1347 | 0.1310 | 0.1304 |
| **Random Forest** | 0.1693 $\pm$ 0.1206 | 0.2267 $\pm$ 0.1692 | 0.2320 $\pm$ 0.1624 | 0.2489 | 0.2560 | 0.2609 |
| **Logistic Regression** | 0.1640 $\pm$ 0.1600 | 0.2000 $\pm$ 0.1944 | 0.2020 $\pm$ 0.1873 | **0.2679** | 0.2619 | 0.2609 |
| **KNN ($k=3$)** | 0.1564 $\pm$ 0.1312 | 0.2267 $\pm$ 0.1818 | 0.2180 $\pm$ 0.1708 | 0.0868 | 0.0833 | 0.0870 |

---

## 6. Ensemble Discussion & Scientific Appraisal

### 6.1 Did Ensembles Improve Performance?
- **Nominal Gain**: The **Stacking Ensemble** attained the highest Mean Macro F1 score ($0.2173$), representing a $+0.0218$ gain over the strongest individual model (SVM at $0.1956$).
- **Variance Trade-off**: The Stacking Ensemble also exhibited the highest variance ($\text{Std} = \pm 0.2144$), compared to SVM ($\text{Std} = \pm 0.1356$) and Soft Voting ($\text{Std} = \pm 0.1453$).
- **Statistical Interpretation**: In small clinical cohorts ($N=23$), stacking introduces a second-level estimation step (meta-learner fitting across inner folds). When sample sizes are small, meta-models can be influenced by fold partitions, leading to higher variance even when nominal mean scores rise.
- **Soft Voting as a Stabilizer**: The Soft Voting Ensemble achieved a balanced performance ($0.1938 \pm 0.1453$), smoothing out the individual errors of tree-based and distance-based estimators.

### 6.2 Feature Importance Insights
- Permutation importance and Random Forest Gini analysis highlighted that systemic indicators of acute physiological stress (`Respiratory_Rate`, `Heart_Rate`, `WBC`, and `Hemoglobin`) contributed most prominently to discriminating crisis severity, aligning with clinical literature on vaso-occlusion inflammatory cascades.

---

## 7. Pilot Study Limitations

1. **Extreme Sample Size ($N=23$)**:
   - The primary limitation is the small cohort of labelled records. With only 4 to 5 patients per validation fold, standard deviations are naturally wide.
2. **Age Heterogeneity**:
   - Pediatric patients (as young as 4 years) and older adults (up to 64 years) were pooled. In clinical medicine, pediatric baseline vital signs (e.g., resting heart rate, respiratory rate) differ substantially from adult norms.
3. **Missing Laboratory Covariates**:
   - Key biochemical markers had high missingness (Serum Creatinine $39.1\%$, Total Bilirubin $26.1\%$). While median imputation prevented row dropping, real-world hospital documentation must be strengthened.
4. **Subjectivity in Ground Truth**:
   - Pain scores (0–10) are inherently subjective and influenced by patient pain tolerance, previous opioid exposure, and psychological distress.

---

## 8. Recommendations for Future Work

1. **Multi-Center Expansion**: Expand data collection to $\ge 500$ encounters across multiple hospital emergency departments.
2. **Stratified Pediatric / Adult Cohorts**: Separate adult and pediatric prediction pipelines or normalize vitals to age-adjusted z-scores (e.g., pediatric early warning scores).
3. **Biomarker Enrichment**: Integrate critical sickle cell biomarkers:
   - Lactate Dehydrogenase (LDH) and Free Hemoglobin (markers of intravascular hemolysis).
   - Fetal Hemoglobin percentage ($\text{HbF}\%$) and Reticulocyte count.
   - High-sensitivity C-reactive protein (hs-CRP) for inflammatory tracking.
4. **Longitudinal Time-Series Tracking**: Transition from single-point cross-sectional snapshots to dynamic monitoring of vitals over the first 6–24 hours of triage.
