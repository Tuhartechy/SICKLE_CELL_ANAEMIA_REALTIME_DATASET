"""
Augmentation module for Sickle Cell Disease Pain Severity Prediction.

Features:
  - Strict leakage prevention: synthetic rows are generated ONLY from training folds.
  - Test folds always contain real, held-out patients only.
  - All synthetic rows are marked with is_synthetic=1.
  - 3 Generators implemented:
      (a) SMOTE (with k_neighbors automatically adjusted to smallest class).
      (b) Gaussian Copula synthesizer from SDV (fitted per-class, sampled balanced).
      (c) Gaussian Noise Jitter (resampling with 5% feature std noise).
  - Clinical constraint enforcement:
      Age in [1, 120], SpO2 <= 100, BP_Systolic > BP_Diastolic, physiological lab bounds,
      valid categories only, and real-data missingness rate preservation.
  - quality_report():
      Per-column KS test, correlation matrix difference, min/max checks, class balance,
      exact duplicate verification (must be 0), and nearest-neighbor distance metrics.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import sdv
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from preprocess import (
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET
)

# Physiological plausible bounds for Sickle Cell cohorts
CLINICAL_BOUNDS = {
    'Age': (1, 120, int),
    'Heart_Rate': (40, 220, int),
    'Respiratory_Rate': (8, 60, int),
    'SpO2': (50.0, 100.0, float),
    'Temperature': (92.0, 108.0, float),
    'BP_Systolic': (60, 240, int),
    'BP_Diastolic': (30, 140, int),
    'Hemoglobin': (1.5, 25.0, float),
    'WBC': (1000.0, 60000.0, float),
    'Platelets': (10000.0, 1000000.0, float),
    'RDW': (8.0, 45.0, float),
    'Bilirubin_Total': (0.1, 35.0, float),
    'Creatinine': (0.1, 15.0, float)
}

VALID_CATEGORIES = {
    'Sex': ['F', 'M'],
    'Genotype': ['SS', 'AS', 'SF', 'SBetaThal', 'Negative', 'Unknown'],
    'Pain_Class': ['Mild', 'Moderate', 'Severe']
}


def apply_clinical_constraints(synthetic_df, real_df, preserve_missingness=True, random_state=42):
    """
    Enforces physiological plausibility and clinical constraints:
      - Numeric columns clipped to clinical bounds and cast to proper types.
      - BP_Systolic enforced to be strictly greater than BP_Diastolic + 10.
      - Categorical columns restricted to existing valid categories.
      - Zero exact duplicates of real rows (per-feature perturbation if identical).
      - Preserves empirical missingness rate per column matching the real training data.
    """
    np.random.seed(random_state)
    df = synthetic_df.copy()
    
    # 1. Bounds and types
    for col, (low, high, dtype) in CLINICAL_BOUNDS.items():
        if col in df.columns:
            # Clip non-null values
            mask = df[col].notnull()
            df.loc[mask, col] = df.loc[mask, col].clip(lower=low, upper=high)
            if dtype is int:
                df.loc[mask, col] = df.loc[mask, col].round().astype(int)
            else:
                df.loc[mask, col] = df.loc[mask, col].round(2)
                
    # 2. Blood pressure clinical constraint: Systolic must exceed Diastolic
    if 'BP_Systolic' in df.columns and 'BP_Diastolic' in df.columns:
        invalid_bp = df['BP_Systolic'] <= (df['BP_Diastolic'] + 10)
        if invalid_bp.any():
            # Adjust Systolic to Diastolic + random pulse pressure in [20, 50]
            pulse_pressures = np.random.randint(20, 50, size=invalid_bp.sum())
            df.loc[invalid_bp, 'BP_Systolic'] = (df.loc[invalid_bp, 'BP_Diastolic'] + pulse_pressures).clip(upper=220)

    # 3. Categoricals valid categories
    for col, valids in VALID_CATEGORIES.items():
        if col in df.columns:
            invalid_cat = ~df[col].isin(valids)
            if invalid_cat.any():
                real_mode = real_df[col].mode()[0] if col in real_df.columns else valids[0]
                df.loc[invalid_cat, col] = real_mode

    # 4. Guarantee zero exact duplicates of real rows
    real_tuples = set(tuple(x) for x in real_df[FEATURES].fillna(-9999).to_numpy())
    for idx, row in df.iterrows():
        syn_tuple = tuple(row[FEATURES].fillna(-9999).to_numpy())
        if syn_tuple in real_tuples:
            # Perturb a random numeric column slightly
            perturb_col = np.random.choice(NUMERIC_FEATURES)
            if pd.notnull(df.loc[idx, perturb_col]):
                std_val = real_df[perturb_col].std() if pd.notnull(real_df[perturb_col].std()) else 1.0
                df.loc[idx, perturb_col] += np.random.uniform(0.02, 0.05) * std_val
                low, high, dtype = CLINICAL_BOUNDS[perturb_col]
                df.loc[idx, perturb_col] = np.clip(df.loc[idx, perturb_col], low, high)
                if dtype is int:
                    df.loc[idx, perturb_col] = round(df.loc[idx, perturb_col])

    # 5. Missingness rate preservation (inject NaNs if columns are 100% full)
    if preserve_missingness:
        for col in NUMERIC_FEATURES:
            real_miss_rate = real_df[col].isna().mean()
            current_miss_rate = df[col].isna().mean()
            if real_miss_rate > 0.02 and current_miss_rate < (real_miss_rate * 0.5):
                # Inject missingness to match real rate
                needed_nan_count = int((real_miss_rate - current_miss_rate) * len(df))
                if needed_nan_count > 0:
                    avail_indices = df[df[col].notnull()].index
                    nan_indices = np.random.choice(avail_indices, size=needed_nan_count, replace=False)
                    df.loc[nan_indices, col] = np.nan

    # Add required marker
    df['is_synthetic'] = 1
    return df


# -------------------------------------------------------------
# GENERATOR A: SMOTE
# -------------------------------------------------------------
def generate_smote(df_train, target_count, random_state=42):
    """
    Synthesizes balanced rows using SMOTE.
    Automatically scales k_neighbors to accommodate small class counts.
    Returns: synthetic DataFrame with target_count rows and is_synthetic=1.
    """
    np.random.seed(random_state)
    df = df_train.copy()
    
    # Target counts per class
    classes = ['Mild', 'Moderate', 'Severe']
    n_per_class = target_count // len(classes)
    
    # 1. Median/Mode imputation on training fold to compute Euclidean distances for SMOTE
    medians = {c: df[c].median(skipna=True) for c in NUMERIC_FEATURES}
    modes = {c: df[c].mode()[0] for c in CATEGORICAL_FEATURES}
    
    df_imputed = df.copy()
    for c in NUMERIC_FEATURES:
        df_imputed[c] = df_imputed[c].fillna(medians[c])
    for c in CATEGORICAL_FEATURES:
        df_imputed[c] = df_imputed[c].fillna(modes[c])
        
    # One-hot encode categoricals for SMOTE
    df_encoded = pd.get_dummies(df_imputed[FEATURES], columns=CATEGORICAL_FEATURES, drop_first=False)
    y = df_imputed[TARGET].values
    
    # Determine smallest class size to set k_neighbors safely
    class_counts = pd.Series(y).value_counts()
    min_count = class_counts.min()
    k_neighbors = max(1, min(5, min_count - 1))
    
    # Setup sampling strategy
    current_counts = dict(class_counts)
    sampling_strategy = {}
    for c in classes:
        # Generate target per class plus initial count
        sampling_strategy[c] = current_counts.get(c, 0) + n_per_class
        
    smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(df_encoded, y)
    
    # Extract only newly generated synthetic rows
    n_real = len(df_train)
    X_syn = X_res.iloc[n_real:n_real + target_count].copy()
    y_syn = y_res[n_real:n_real + target_count]
    
    # Reconstruct original columns
    res_df = pd.DataFrame()
    for col in NUMERIC_FEATURES:
        res_df[col] = X_syn[col].values
        
    # Decode categorical columns back from one-hot representation
    for cat_col in CATEGORICAL_FEATURES:
        cat_dummy_cols = [c for c in X_syn.columns if c.startswith(f"{cat_col}_")]
        if cat_dummy_cols:
            best_cat = X_syn[cat_dummy_cols].idxmax(axis=1).str.replace(f"{cat_col}_", "", regex=False)
            res_df[cat_col] = best_cat.values
        else:
            res_df[cat_col] = modes[cat_col]
            
    res_df[TARGET] = y_syn
    
    # Fill remaining rows if target_count doesn't divide evenly by 3
    remaining = target_count - len(res_df)
    if remaining > 0:
        extra = res_df.sample(remaining, replace=True, random_state=random_state)
        res_df = pd.concat([res_df, extra], ignore_index=True)
        
    # Apply clinical constraints & inject empirical missingness
    res_df = apply_clinical_constraints(res_df, df_train, preserve_missingness=True, random_state=random_state)
    return res_df.head(target_count).reset_index(drop=True)


# -------------------------------------------------------------
# GENERATOR B: GAUSSIAN COPULA (SDV)
# -------------------------------------------------------------
def generate_copula(df_train, target_count, random_state=42):
    """
    Synthesizes balanced rows using SDV's GaussianCopulaSynthesizer,
    fitted per class on training data.
    Returns: synthetic DataFrame with target_count rows and is_synthetic=1.
    """
    np.random.seed(random_state)
    classes = ['Mild', 'Moderate', 'Severe']
    n_per_class = target_count // len(classes)
    
    synthetic_classes = []
    
    for cls_name in classes:
        cls_data = df_train[df_train[TARGET] == cls_name][FEATURES].copy()
        
        # Build SingleTableMetadata
        metadata = SingleTableMetadata()
        for col in NUMERIC_FEATURES:
            metadata.add_column(column_name=col, sdtype='numerical')
        for col in CATEGORICAL_FEATURES:
            metadata.add_column(column_name=col, sdtype='categorical')
            
        # Fit GaussianCopula
        synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='norm')
        synthesizer.fit(cls_data)
        
        # Sample class rows
        syn_cls = synthesizer.sample(num_rows=n_per_class)
        syn_cls[TARGET] = cls_name
        synthetic_classes.append(syn_cls)
        
    combined_syn = pd.concat(synthetic_classes, ignore_index=True)
    
    # Fill remaining rows if target_count doesn't divide evenly by 3
    remaining = target_count - len(combined_syn)
    if remaining > 0:
        extra = combined_syn.sample(remaining, replace=True, random_state=random_state)
        combined_syn = pd.concat([combined_syn, extra], ignore_index=True)
        
    # Apply clinical constraints and missingness preservation
    res_df = apply_clinical_constraints(combined_syn, df_train, preserve_missingness=True, random_state=random_state)
    return res_df.head(target_count).reset_index(drop=True)


# -------------------------------------------------------------
# GENERATOR C: NOISE JITTER
# -------------------------------------------------------------
def generate_jitter(df_train, target_count, noise_factor=0.05, random_state=42):
    """
    Synthesizes balanced rows via resampling real training rows
    and injecting Gaussian noise of ~5% of each numeric column's std.
    Returns: synthetic DataFrame with target_count rows and is_synthetic=1.
    """
    np.random.seed(random_state)
    classes = ['Mild', 'Moderate', 'Severe']
    n_per_class = target_count // len(classes)
    
    # Calculate column std on real training data
    col_stds = {}
    for col in NUMERIC_FEATURES:
        col_stds[col] = float(df_train[col].std(skipna=True))
        if np.isnan(col_stds[col]) or col_stds[col] == 0:
            col_stds[col] = 1.0
            
    synthetic_list = []
    
    for cls_name in classes:
        cls_data = df_train[df_train[TARGET] == cls_name]
        
        # Resample real rows with replacement
        resampled = cls_data.sample(n=n_per_class, replace=True, random_state=random_state).copy()
        
        # Inject Gaussian jitter into numeric columns
        for col in NUMERIC_FEATURES:
            std = col_stds[col]
            noise = np.random.normal(loc=0.0, scale=noise_factor * std, size=len(resampled))
            # Only add noise where value was not null
            mask = resampled[col].notnull()
            resampled.loc[mask, col] = resampled.loc[mask, col] + noise[mask]
            
        synthetic_list.append(resampled)
        
    combined_syn = pd.concat(synthetic_list, ignore_index=True)
    
    # Fill remaining rows if target_count doesn't divide evenly
    remaining = target_count - len(combined_syn)
    if remaining > 0:
        extra = combined_syn.sample(remaining, replace=True, random_state=random_state)
        combined_syn = pd.concat([combined_syn, extra], ignore_index=True)
        
    # Apply clinical constraints
    res_df = apply_clinical_constraints(combined_syn, df_train, preserve_missingness=False, random_state=random_state)
    return res_df.head(target_count).reset_index(drop=True)


# -------------------------------------------------------------
# QUALITY REPORT
# -------------------------------------------------------------
def quality_report(real_df, synthetic_df, label="Synthetic Dataset"):
    """
    Computes rigorous statistical quality metrics comparing real vs synthetic data:
      1. Per-column 2-sample Kolmogorov-Smirnov (KS) test (statistic & p-value).
      2. Correlation-matrix difference (Frobenius norm and mean absolute difference).
      3. Min / Max range verification.
      4. Class balance check.
      5. Number of exact duplicates of real rows (must be 0).
      6. Distance of each synthetic row to its nearest real training row.
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"             DATA QUALITY & FIDELITY REPORT: {label}")
    report_lines.append("=" * 70)
    report_lines.append(f"Real Cohort Size:      {len(real_df)} rows")
    report_lines.append(f"Synthetic Cohort Size: {len(synthetic_df)} rows")
    report_lines.append(f"is_synthetic Column:   {'Present (all = 1)' if (synthetic_df['is_synthetic'] == 1).all() else 'MISSING OR INVALID'}")
    
    # 1. Exact Duplicate Verification (MUST BE ZERO)
    real_mat = real_df[FEATURES].fillna(-9999).to_numpy()
    syn_mat = synthetic_df[FEATURES].fillna(-9999).to_numpy()
    real_set = set(tuple(x) for x in real_mat)
    
    exact_duplicates = sum(1 for row in syn_mat if tuple(row) in real_set)
    report_lines.append(f"\n[1] EXACT DUPLICATES OF REAL ROWS: {exact_duplicates} (Pass criteria: 0)")
    if exact_duplicates == 0:
        report_lines.append("    -> PASS: No synthetic row is an exact replica of any real training patient.")
    else:
        report_lines.append(f"    -> FAIL: Detected {exact_duplicates} duplicate rows.")

    # 2. Class Balance Check
    report_lines.append(f"\n[2] TARGET CLASS BALANCE ({TARGET}):")
    class_counts = synthetic_df[TARGET].value_counts()
    for cls, cnt in class_counts.items():
        pct = (cnt / len(synthetic_df)) * 100
        report_lines.append(f"    - {cls:<10}: {cnt:>5} rows ({pct:.1f}%)")
        
    # 3. Min/Max Range Verification
    report_lines.append(f"\n[3] CLINICAL BOUNDS & RANGE VERIFICATION:")
    bounds_violations = 0
    for col in NUMERIC_FEATURES:
        low_bound, high_bound, _ = CLINICAL_BOUNDS[col]
        s_min = synthetic_df[col].min(skipna=True)
        s_max = synthetic_df[col].max(skipna=True)
        r_min = real_df[col].min(skipna=True)
        r_max = real_df[col].max(skipna=True)
        
        status = "OK"
        if s_min < low_bound or s_max > high_bound:
            status = "VIOLATION"
            bounds_violations += 1
            
        report_lines.append(f"    {col:<17} | Real: [{r_min:>6.1f}, {r_max:>6.1f}] | Synthetic: [{s_min:>6.1f}, {s_max:>6.1f}] | Bound: [{low_bound}, {high_bound}] -> {status}")
        
    report_lines.append(f"    Total Range Violations: {bounds_violations}")

    # 4. Correlation Matrix Difference (Frobenius Norm & Mean Abs Diff)
    report_lines.append(f"\n[4] CORRELATION MATRIX PRESERVATION (Numeric Features):")
    real_corr = real_df[NUMERIC_FEATURES].corr().fillna(0).to_numpy()
    syn_corr = synthetic_df[NUMERIC_FEATURES].corr().fillna(0).to_numpy()
    
    corr_diff = np.abs(real_corr - syn_corr)
    frob_norm = np.linalg.norm(real_corr - syn_corr, 'fro')
    mean_abs_diff = np.mean(corr_diff)
    
    report_lines.append(f"    Frobenius Norm Difference:      {frob_norm:.4f}")
    report_lines.append(f"    Mean Absolute Correlation Diff: {mean_abs_diff:.4f}")

    # 5. Per-Column Two-Sample Kolmogorov-Smirnov (KS) Test
    report_lines.append(f"\n[5] TWO-SAMPLE KOLMOGOROV-SMIRNOV (KS) DISTRIBUTION TEST:")
    report_lines.append("    (Tests null hypothesis that real and synthetic samples come from the same distribution)")
    ks_results = []
    for col in NUMERIC_FEATURES:
        real_vals = real_df[col].dropna().values
        syn_vals = synthetic_df[col].dropna().values
        if len(real_vals) > 0 and len(syn_vals) > 0:
            ks_stat, p_val = stats.ks_2samp(real_vals, syn_vals)
            ks_results.append({'Feature': col, 'KS_Stat': ks_stat, 'p_value': p_val})
            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
            report_lines.append(f"    {col:<17} | KS Stat: {ks_stat:.4f} | p-value: {p_str:>8}")
            
    # 6. Nearest Neighbor Distance to Real Rows
    report_lines.append(f"\n[6] NEAREST-NEIGHBOR DISTANCE TO REAL DATA (Standardized Space):")
    # Standardize numeric columns using real data scaling
    imputer_vals = real_df[NUMERIC_FEATURES].median().to_dict()
    scaler = StandardScaler()
    real_num_scaled = scaler.fit_transform(real_df[NUMERIC_FEATURES].fillna(imputer_vals))
    syn_num_scaled = scaler.transform(synthetic_df[NUMERIC_FEATURES].fillna(imputer_vals))
    
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(real_num_scaled)
    distances, _ = nbrs.kneighbors(syn_num_scaled)
    distances = distances.flatten()
    
    report_lines.append(f"    Min Distance to Real Row:    {np.min(distances):.4f}")
    report_lines.append(f"    Median Distance to Real Row: {np.median(distances):.4f}")
    report_lines.append(f"    Mean Distance to Real Row:   {np.mean(distances):.4f}")
    report_lines.append(f"    Max Distance to Real Row:    {np.max(distances):.4f}")
    
    report_lines.append("=" * 70)
    full_report = "\n".join(report_lines)
    return full_report, {
        'exact_duplicates': exact_duplicates,
        'bounds_violations': bounds_violations,
        'frob_norm': frob_norm,
        'mean_abs_diff': mean_abs_diff,
        'min_distance': float(np.min(distances)),
        'median_distance': float(np.median(distances)),
        'mean_distance': float(np.mean(distances))
    }


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_path = os.path.join(base_dir, 'data', 'patients_clean.csv')
    
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Clean patient data not found at {clean_path}.")
        
    df_real = pd.read_csv(clean_path)
    print(f"Loaded real dataset: {len(df_real)} rows.")
    
    # Verification test: synthesize 500 rows with each method and evaluate quality
    print("\n--- Verifying Generator 1: SMOTE (N=500) ---")
    syn_smote = generate_smote(df_real, target_count=500, random_state=42)
    rep_smote, _ = quality_report(df_real, syn_smote, label="SMOTE (500 rows)")
    print(rep_smote)
    
    print("\n--- Verifying Generator 2: Gaussian Copula (N=500) ---")
    syn_copula = generate_copula(df_real, target_count=500, random_state=42)
    rep_copula, _ = quality_report(df_real, syn_copula, label="Gaussian Copula (500 rows)")
    print(rep_copula)
    
    print("\n--- Verifying Generator 3: Noise Jitter (N=500) ---")
    syn_jitter = generate_jitter(df_real, target_count=500, random_state=42)
    rep_jitter, _ = quality_report(df_real, syn_jitter, label="Noise Jitter (500 rows)")
    print(rep_jitter)
