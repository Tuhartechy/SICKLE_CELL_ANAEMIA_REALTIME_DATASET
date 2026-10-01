"""
Preprocess module for Sickle Cell Disease Pain Severity Prediction.

Handles raw data ingestion, text standardization, physiological value validation,
target variable ('Pain_Class') derivation, and exports:
  - data/patients_clean.csv (labelled rows for model training/validation)
  - data/rows_for_review.csv (unlabelled and conflicting rows)

Provides make_preprocessor() to construct an unfitted scikit-learn ColumnTransformer
to prevent data leakage prior to cross-validation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- FEATURE AND TARGET DEFINITIONS ---
# Patient_ID, Pain_Score, and Pain_Severity are strictly excluded from FEATURES.
NUMERIC_FEATURES = [
    'Age',
    'Heart_Rate',
    'Respiratory_Rate',
    'SpO2',
    'Temperature',
    'BP_Systolic',
    'BP_Diastolic',
    'Hemoglobin',
    'WBC',
    'Platelets',
    'RDW',
    'Bilirubin_Total',
    'Creatinine'
]

CATEGORICAL_FEATURES = [
    'Sex',
    'Genotype'
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'Pain_Class'

# Verify no leakage
assert 'Patient_ID' not in FEATURES, "Target/Identifier leakage: Patient_ID in FEATURES"
assert 'Pain_Score' not in FEATURES, "Target leakage: Pain_Score in FEATURES"
assert 'Pain_Severity' not in FEATURES, "Target leakage: Pain_Severity in FEATURES"


def make_preprocessor():
    """
    Builds an unfitted scikit-learn ColumnTransformer.
    - Numeric: median imputation + StandardScaler
    - Categorical: most-frequent imputation + OneHotEncoder(handle_unknown='ignore')
    
    This transformer must ONLY be fitted inside cross-validation training folds
    to prevent data leakage.
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def map_text_severity(text):
    """Maps free-text pain descriptions to categories or flags as unlabelled."""
    if not isinstance(text, str):
        return 'Blank'
    t = text.strip().lower()
    if t in ['none', 'no pain', 'very mild', 'mild', 'mild tolerable']:
        return 'Mild'
    elif t in ['moderate', 'moderate intensity']:
        return 'Moderate'
    elif t in ['intense', 'severe', 'severe pain', 'strong pain', 'very intense', 'distressing']:
        return 'Severe'
    elif t in ['acute', 'acute pain', 'colicky', 'pain in abdomen', 'abdominal pain (reported)', 'sudden progressive chest pain']:
        return 'Pain Type / Location'
    elif t == '' or t == 'nan':
        return 'Blank'
    else:
        return f'Other: {t}'


def map_pain_class(score, text):
    """
    Determines Pain_Class (Mild, Moderate, Severe), Conflict, or Unlabelled.
    Rules:
      - Score decides if present: 0-3 Mild, 4-6 Moderate, 7-10 Severe.
      - If no score, text decides: Mild, Moderate, Severe lists.
      - If pain type/location or blank with no score -> Unlabelled.
      - If score and text disagree -> Score wins, flagged as Conflict.
    """
    score_val = None
    if pd.notnull(score) and str(score).strip() != '' and str(score).strip().lower() != 'nan':
        try:
            score_val = float(score)
        except ValueError:
            score_val = None
            
    text_val = str(text).strip() if pd.notnull(text) else ''
    text_mapped = map_text_severity(text_val)
    
    score_mapped = None
    if score_val is not None:
        if 0.0 <= score_val <= 3.0:
            score_mapped = 'Mild'
        elif 4.0 <= score_val <= 6.0:
            score_mapped = 'Moderate'
        elif 7.0 <= score_val <= 10.0:
            score_mapped = 'Severe'
        else:
            score_mapped = 'Invalid'
            
    if score_mapped is not None and score_mapped != 'Invalid':
        final_class = score_mapped
        if text_mapped in ['Mild', 'Moderate', 'Severe']:
            if score_mapped != text_mapped:
                return final_class, 'Conflict', f'Score {score_val} -> {score_mapped} (WINS), but text is "{text_val}" ({text_mapped})'
            else:
                return final_class, 'Consistent', f'Score {score_val} and text "{text_val}" both map to {score_mapped}'
        else:
            return final_class, 'Score Only', f'Score {score_val} -> {score_mapped}; text was "{text_val}"'
    else:
        # No score -> check text
        if text_mapped in ['Mild', 'Moderate', 'Severe']:
            return text_mapped, 'Text Only', f'No score; text "{text_val}" maps to {text_mapped}'
        else:
            return None, 'Unlabelled', f'No score; text "{text_val}" ({text_mapped}) does not specify severity'


def clean_patient_data(raw_csv_path, output_clean_path, output_review_path):
    """
    Cleans raw patient dataset, validates ranges, maps labels,
    and saves clean and review CSV files.
    """
    corrections = []
    
    # Read preserving string representation so 'None' is not automatically converted to NaN
    df_raw = pd.read_csv(raw_csv_path, keep_default_na=False, dtype={'Patient_ID': str})
    rows_read = len(df_raw)
    
    # 1. Clean strings and standardise categorical variables
    df = df_raw.copy()
    for col in df.select_dtypes(include=['object', 'string', 'str']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Standardise Sex
    def standardize_sex(val):
        v = val.upper()
        if v in ['F', 'FEMALE']:
            return 'F'
        elif v in ['M', 'MALE']:
            return 'M'
        return 'Unknown'
    df['Sex'] = df['Sex'].apply(standardize_sex)
    
    # Standardise Genotype
    def standardize_genotype(val):
        v = val.strip()
        if v == 'Sickle Beta Thalassemia':
            return 'SBetaThal'
        elif v in ['SS', 'AS', 'SF', 'Negative']:
            return v
        elif v == '' or v.lower() == 'nan':
            return 'Unknown'
        return v
    
    orig_genotypes = df['Genotype'].copy()
    df['Genotype'] = df['Genotype'].apply(standardize_genotype)
    for idx, (orig, new) in enumerate(zip(orig_genotypes, df['Genotype'])):
        if orig != new:
            corrections.append(f"Row {idx+1} (Patient {df.loc[idx, 'Patient_ID']}): Standardized Genotype from '{orig}' to '{new}'")

    # 2. Convert and validate numeric columns
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')
        
    # Out of range / anomaly checks
    for idx, row in df.iterrows():
        pid = row['Patient_ID']
        
        # Temperature check: if < 50, treat as Celsius and convert
        temp = row['Temperature']
        if pd.notnull(temp):
            if temp < 50.0:
                temp_f = round((temp * 9.0 / 5.0) + 32.0, 2)
                corrections.append(f"Row {idx+1} (Patient {pid}): Temperature {temp}°C converted to {temp_f}°F")
                df.loc[idx, 'Temperature'] = temp_f
            elif temp < 90.0 or temp > 115.0:
                corrections.append(f"Row {idx+1} (Patient {pid}): Impossible Temperature {temp}°F set to NaN")
                df.loc[idx, 'Temperature'] = np.nan
                
        # SpO2 check
        spo2 = row['SpO2']
        if pd.notnull(spo2):
            if spo2 > 100.0 or spo2 < 50.0:
                corrections.append(f"Row {idx+1} (Patient {pid}): Impossible SpO2 {spo2}% set to NaN")
                df.loc[idx, 'SpO2'] = np.nan
                
        # Check non-negatives across all numeric features
        for col in NUMERIC_FEATURES:
            val = row[col]
            if pd.notnull(val) and val < 0:
                corrections.append(f"Row {idx+1} (Patient {pid}): Negative value {val} in {col} set to NaN")
                df.loc[idx, col] = np.nan

    # 3. Label mapping
    pain_classes = []
    statuses = []
    reasons = []
    
    for idx, row in df.iterrows():
        score_raw = row['Pain_Score']
        text_raw = row['Pain_Severity']
        p_class, status, reason = map_pain_class(score_raw, text_raw)
        pain_classes.append(p_class)
        statuses.append(status)
        reasons.append(reason)
        
    df['Pain_Class'] = pain_classes
    df['Label_Status'] = statuses
    df['Review_Reason'] = reasons
    
    # 4. Split into clean labelled dataset and review dataset
    clean_mask = df['Pain_Class'].notnull()
    clean_df = df[clean_mask].copy()
    
    # Review dataset: all unlabelled rows + all conflicting rows
    review_mask = (df['Pain_Class'].isna()) | (df['Label_Status'] == 'Conflict')
    review_df = df[review_mask].copy()
    
    # Ensure export directory exists
    os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_review_path), exist_ok=True)
    
    # Save clean dataset (drop temporary columns used for audit, keep features + Patient_ID + Target)
    clean_cols_to_save = ['Patient_ID'] + FEATURES + [TARGET]
    clean_df[clean_cols_to_save].to_csv(output_clean_path, index=False)
    
    # Save review dataset (include IDs, scores, raw texts, decision, reason)
    review_cols_to_save = ['Patient_ID', 'Pain_Score', 'Pain_Severity', 'Pain_Class', 'Label_Status', 'Review_Reason']
    review_df[review_cols_to_save].to_csv(output_review_path, index=False)
    
    # 5. Print Preprocessing Report
    print("=" * 70)
    print("                PREPROCESSING AND AUDIT REPORT")
    print("=" * 70)
    print(f"Total Rows Read:              {rows_read}")
    print(f"Labelled Rows Exported:       {len(clean_df)} -> {output_clean_path}")
    print(f"Rows Dropped (Unlabelled):    {len(df[~clean_mask])}")
    print(f"Rows Written for Review:      {len(review_df)} -> {output_review_path}")
    print(f"  - Conflicting rows included with score-derived label: {len(df[df['Label_Status'] == 'Conflict'])}")
    print(f"  - Unlabelled rows excluded:                          {len(df[df['Pain_Class'].isna()])}")
    
    print("\n--- Target Class Distribution (Labelled Patients) ---")
    class_counts = clean_df['Pain_Class'].value_counts()
    for cls_name, cnt in class_counts.items():
        pct = (cnt / len(clean_df)) * 100
        print(f"  {cls_name:<10}: {cnt:>2} patients ({pct:.1f}%)")
        
    print("\n--- Genotype Distribution (Labelled Patients) ---")
    geno_counts = clean_df['Genotype'].value_counts()
    for geno, cnt in geno_counts.items():
        pct = (cnt / len(clean_df)) * 100
        print(f"  {geno:<12}: {cnt:>2} patients ({pct:.1f}%)")
        
    print("\n--- Missing Percentage per Feature (Clean Dataset, N=23) ---")
    missing_tbl = []
    for f in FEATURES:
        m_cnt = clean_df[f].isna().sum()
        m_pct = (m_cnt / len(clean_df)) * 100
        missing_tbl.append({'Feature': f, 'Missing_Count': m_cnt, 'Missing_Pct': f"{m_pct:.1f}%"})
    missing_df = pd.DataFrame(missing_tbl)
    print(missing_df.to_string(index=False))
    
    print("\n--- Data Corrections & Normalisations Applied ---")
    if corrections:
        for c in corrections:
            print(f"  * {c}")
    else:
        print("  * None (all data within standard physical bounds)")
        
    print("\n--- Conflicting Rows Flagged for Review ---")
    for _, r in df[df['Label_Status'] == 'Conflict'].iterrows():
        print(f"  * Patient {r['Patient_ID']}: Score={r['Pain_Score']}, Text='{r['Pain_Severity']}' | {r['Review_Reason']}")
        
    print("=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY (No data leakage introduced)")
    print("=" * 70)
    
    return clean_df, review_df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'patients.csv')
    clean_path = os.path.join(base_dir, 'data', 'patients_clean.csv')
    review_path = os.path.join(base_dir, 'data', 'rows_for_review.csv')
    
    clean_patient_data(raw_path, clean_path, review_path)
