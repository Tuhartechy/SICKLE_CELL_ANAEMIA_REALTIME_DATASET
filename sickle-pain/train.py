"""
Train module for Sickle Cell Disease Pain Severity Prediction.

Features:
  - 5 baseline models: Random Forest, XGBoost, SVM, Logistic Regression, KNN.
  - 2 ensembles: Soft Voting Classifier, Stacking Classifier (with Logistic Regression).
  - Strict leakage prevention: each model is a Pipeline with make_preprocessor().
  - Rigorous evaluation: RepeatedStratifiedKFold (5 folds x 5 repeats = 25 evaluations)
    and Leave-One-Out cross-validation (LOOCV).
  - Comprehensive metrics: Macro F1, Balanced Accuracy, Accuracy.
  - Artifact exports: model_comparison.csv, comparison plots, confusion matrix,
    feature importance, best_model.joblib, and metadata.json.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer
)
from sklearn.inspection import permutation_importance

# Candidate classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Import custom preprocessor from preprocess.py
from preprocess import (
    make_preprocessor,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def get_base_models():
    """
    Instantiates the 5 core machine learning classifiers with simple,
    regularised hyperparameters tailored for very small sample sizes.
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=3,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            eval_metric='mlogloss',
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=3,
            weights='distance'
        )
    }
    return models


def build_all_pipelines():
    """
    Wraps all models (5 single + 2 ensembles) into complete scikit-learn Pipelines.
    This guarantees that preprocessing (imputation, scaling, one-hot encoding)
    is strictly fitted on training splits only, preventing data leakage.
    """
    base_models = get_base_models()
    
    # 1. Pipelines for the 5 individual models
    single_pipelines = {
        name: Pipeline([
            ('preprocessor', make_preprocessor()),
            ('classifier', clf)
        ])
        for name, clf in base_models.items()
    }
    
    # 2. Named estimators for ensembles
    named_estimators = [
        (name.lower().replace(' ', '_'), pipe)
        for name, pipe in single_pipelines.items()
    ]
    
    # 3. Soft Voting Ensemble
    voting_ensemble = VotingClassifier(
        estimators=named_estimators,
        voting='soft'
    )
    
    # 4. Stacking Ensemble with Logistic Regression meta-classifier
    stacking_ensemble = StackingClassifier(
        estimators=named_estimators,
        final_estimator=LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        cv=3  # 3-fold inner cross-validation for stacking meta-features
    )
    
    all_models = dict(single_pipelines)
    all_models['Soft Voting Ensemble'] = voting_ensemble
    all_models['Stacking Ensemble'] = stacking_ensemble
    
    return all_models


def evaluate_models(X, y, class_names):
    """
    Evaluates all 7 models using Repeated Stratified K-Fold CV (5 folds x 5 repeats)
    and Leave-One-Out Cross-Validation (LOOCV).
    """
    models = build_all_pipelines()
    
    # 5-fold CV with 5 repeats = 25 evaluations
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    
    # Pre-generate split indices so every model is evaluated on identical folds
    cv_splits = list(rskf.split(X, y))
    
    loo = LeaveOneOut()
    loo_splits = list(loo.split(X, y))
    
    results = []
    oof_predictions_rskf = {name: [] for name in models}
    oof_trues_rskf = []
    
    print("\n--- Running Repeated Stratified K-Fold Cross-Validation (5x5 = 25 folds) ---")
    for name, model in models.items():
        fold_accs = []
        fold_macro_f1s = []
        fold_bal_accs = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]
            
            # Fresh pipeline clone to ensure clean state
            pipe = build_all_pipelines()[name]
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_va)
            
            fold_accs.append(accuracy_score(y_va, y_pred))
            fold_macro_f1s.append(f1_score(y_va, y_pred, average='macro', zero_division=0))
            fold_bal_accs.append(balanced_accuracy_score(y_va, y_pred))
            
            # Store OOF predictions for the first repeat (folds 0..4) for confusion matrix
            if fold_idx < 5:
                oof_predictions_rskf[name].extend(y_pred)
                if name == list(models.keys())[0]:
                    oof_trues_rskf.extend(y_va)
                    
        # LOOCV evaluation
        loo_trues = []
        loo_preds = []
        for train_idx, val_idx in loo_splits:
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]
            
            pipe = build_all_pipelines()[name]
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            loo_trues.append(y_va[0])
            loo_preds.append(pred[0])
            
        loo_macro_f1 = f1_score(loo_trues, loo_preds, average='macro', zero_division=0)
        loo_bal_acc = balanced_accuracy_score(loo_trues, loo_preds)
        loo_acc = accuracy_score(loo_trues, loo_preds)
        
        results.append({
            'Model': name,
            'CV_Macro_F1_Mean': np.mean(fold_macro_f1s),
            'CV_Macro_F1_Std': np.std(fold_macro_f1s),
            'CV_Bal_Acc_Mean': np.mean(fold_bal_accs),
            'CV_Bal_Acc_Std': np.std(fold_bal_accs),
            'CV_Accuracy_Mean': np.mean(fold_accs),
            'CV_Accuracy_Std': np.std(fold_accs),
            'LOOCV_Macro_F1': loo_macro_f1,
            'LOOCV_Bal_Acc': loo_bal_acc,
            'LOOCV_Accuracy': loo_acc
        })
        
        print(f"  {name:<22} | Macro F1: {np.mean(fold_macro_f1s):.3f} +/- {np.std(fold_macro_f1s):.3f} | Bal Acc: {np.mean(fold_bal_accs):.3f} | LOOCV F1: {loo_macro_f1:.3f}")
        
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='CV_Macro_F1_Mean', ascending=False).reset_index(drop=True)
    
    return res_df, oof_predictions_rskf, np.array(oof_trues_rskf)


def save_visualizations(res_df, best_model_name, best_model_pipeline, X, y, class_names, oof_preds, oof_trues, results_dir):
    """
    Generates and saves the required evaluation plots:
    1. Macro F1 Comparison bar chart with error bars
    2. Aggregated Confusion Matrix for the best model
    3. Feature Importance plot (Random Forest + Best Model Permutation Importance)
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Macro F1 Bar Chart
    plt.figure(figsize=(10, 6))
    models = res_df['Model']
    means = res_df['CV_Macro_F1_Mean']
    stds = res_df['CV_Macro_F1_Std']
    colors = ['#2b5c8f' if m == best_model_name else '#7fa9c9' for m in models]
    
    bars = plt.barh(models[::-1], means[::-1], xerr=stds[::-1], color=colors[::-1], capsize=5, edgecolor='black', alpha=0.85)
    plt.xlabel('Cross-Validated Macro F1 Score', fontsize=12)
    plt.title('Model Comparison: Repeated Stratified 5-Fold Cross-Validation (5 Repeats)', fontsize=13, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    chart_path = os.path.join(results_dir, 'macro_f1_comparison.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved Macro F1 chart to: {chart_path}")
    
    # 2. Confusion Matrix for Best Model
    sanitized_name = best_model_name.lower().replace(' ', '_')
    cm_path = os.path.join(results_dir, f'confusion_matrix_{sanitized_name}.png')
    
    # Re-generate 5-fold OOF predictions specifically for the best model to produce confusion matrix
    cm_preds = oof_preds[best_model_name]
    cm = confusion_matrix(oof_trues, cm_preds, labels=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'OOF Confusion Matrix: {best_model_name}\n(Stratified 5-Fold CV)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to: {cm_path}")
    
    # 3. Feature Importance Plot
    # Fit Random Forest pipeline on all data to extract Gini importances
    rf_pipe = build_all_pipelines()['Random Forest']
    rf_pipe.fit(X, y)
    
    # Extract feature names after OneHotEncoder
    preprocessor = rf_pipe.named_steps['preprocessor']
    num_cols = NUMERIC_FEATURES
    cat_cols_ohe = list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(CATEGORICAL_FEATURES))
    all_transformed_feature_names = num_cols + cat_cols_ohe
    
    rf_importances = rf_pipe.named_steps['classifier'].feature_importances_
    
    # Calculate permutation importance on the fitted best model
    macro_f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)
    perm_res = permutation_importance(
        best_model_pipeline, X, y, n_repeats=20, random_state=RANDOM_STATE, scoring=macro_f1_scorer
    )
    perm_importances_mean = perm_res.importances_mean
    perm_importances_std = perm_res.importances_std
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: RF Gini Importances (top 10)
    rf_df = pd.DataFrame({'Feature': all_transformed_feature_names, 'Importance': rf_importances})
    rf_df = rf_df.sort_values(by='Importance', ascending=True).tail(10)
    ax1.barh(rf_df['Feature'], rf_df['Importance'], color='#3a7ebf', edgecolor='black', alpha=0.85)
    ax1.set_title('Random Forest Feature Importance (MDI / Gini)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Mean Decrease in Impurity', fontsize=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Plot 2: Permutation Importances for Best Model (raw features)
    perm_df = pd.DataFrame({
        'Feature': FEATURES,
        'Mean': perm_importances_mean,
        'Std': perm_importances_std
    }).sort_values(by='Mean', ascending=True).tail(10)
    
    ax2.barh(perm_df['Feature'], perm_df['Mean'], xerr=perm_df['Std'], color='#e06d53', edgecolor='black', alpha=0.85, capsize=4)
    ax2.set_title(f'Permutation Importance ({best_model_name})', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Macro F1 Drop Upon Permutation', fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fi_path = os.path.join(results_dir, 'feature_importance.png')
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance plot to: {fi_path}")


def save_model_and_metadata(best_model_name, best_model_pipeline, X, y, class_names, res_df, models_dir):
    """
    Fits the best model on all labelled data and exports:
    - models/best_model.joblib
    - models/metadata.json
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Fit the best model on all available labelled data
    best_model_pipeline.fit(X, y)
    
    model_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(best_model_pipeline, model_path)
    print(f"\nSaved best model ({best_model_name}) to: {model_path}")
    
    # Compute feature defaults (medians and modes) for Streamlit UI
    feature_defaults = {}
    for col in NUMERIC_FEATURES:
        feature_defaults[col] = float(X[col].median(skipna=True))
    for col in CATEGORICAL_FEATURES:
        feature_defaults[col] = str(X[col].mode(dropna=True)[0])
        
    best_stats = res_df[res_df['Model'] == best_model_name].iloc[0]
    
    metadata = {
        'project_title': 'Ensemble Machine Learning Approach for Predicting Pain Severity in Sickle Cell Disease',
        'best_model_name': best_model_name,
        'target_column': TARGET,
        'class_names': class_names,
        'label_mapping': {int(idx): name for idx, name in enumerate(class_names)},
        'feature_names': FEATURES,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'categorical_options': {
            'Sex': sorted(X['Sex'].dropna().unique().tolist()),
            'Genotype': sorted(X['Genotype'].dropna().unique().tolist())
        },
        'feature_defaults': feature_defaults,
        'training_sample_count': int(len(X)),
        'cross_validation': {
            'method': 'RepeatedStratifiedKFold (5 splits, 5 repeats = 25 folds)',
            'mean_macro_f1': float(round(best_stats['CV_Macro_F1_Mean'], 4)),
            'std_macro_f1': float(round(best_stats['CV_Macro_F1_Std'], 4)),
            'mean_balanced_accuracy': float(round(best_stats['CV_Bal_Acc_Mean'], 4)),
            'std_balanced_accuracy': float(round(best_stats['CV_Bal_Acc_Std'], 4)),
            'mean_accuracy': float(round(best_stats['CV_Accuracy_Mean'], 4)),
            'std_accuracy': float(round(best_stats['CV_Accuracy_Std'], 4)),
            'loocv_macro_f1': float(round(best_stats['LOOCV_Macro_F1'], 4)),
            'loocv_balanced_accuracy': float(round(best_stats['LOOCV_Bal_Acc'], 4)),
            'loocv_accuracy': float(round(best_stats['LOOCV_Accuracy'], 4))
        }
    }
    
    metadata_path = os.path.join(models_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_csv_path = os.path.join(base_dir, 'data', 'patients_clean.csv')
    results_dir = os.path.join(base_dir, 'results')
    models_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(clean_csv_path):
        raise FileNotFoundError(f"Clean dataset not found at {clean_csv_path}. Run preprocess.py first.")
        
    df = pd.read_csv(clean_csv_path)
    print(f"Loaded clean dataset: {len(df)} patients, {df.shape[1]} columns.")
    
    X = df[FEATURES].copy()
    y_raw = df[TARGET].copy()
    
    # Target encoding: 0=Mild, 1=Moderate, 2=Severe
    le = LabelEncoder()
    # Explicit order
    le.classes_ = np.array(['Mild', 'Moderate', 'Severe'])
    y = le.transform(y_raw)
    class_names = list(le.classes_)
    
    print(f"Target distribution: {dict(pd.Series(y_raw).value_counts())}")
    print(f"Features ({len(FEATURES)}): {FEATURES}")
    
    # Evaluate all models
    res_df, oof_preds, oof_trues = evaluate_models(X, y, class_names)
    
    # Save comparison table
    comparison_csv_path = os.path.join(results_dir, 'model_comparison.csv')
    res_df.round(4).to_csv(comparison_csv_path, index=False)
    print(f"\nSaved model comparison table to: {comparison_csv_path}")
    
    # Identify best model by Mean Macro F1
    best_model_name = res_df.iloc[0]['Model']
    best_macro_f1 = res_df.iloc[0]['CV_Macro_F1_Mean']
    print(f"\n>>> Best Model by CV Mean Macro F1: {best_model_name} (Macro F1 = {best_macro_f1:.4f}) <<<")
    
    # Honest evaluation of Ensembles vs Single Models
    print("\n--- Ensemble vs Single Model Comparison ---")
    single_models_df = res_df[~res_df['Model'].str.contains('Ensemble')]
    best_single = single_models_df.iloc[0]
    ensemble_df = res_df[res_df['Model'].str.contains('Ensemble')]
    
    print(f"Top Single Model:   {best_single['Model']} (Macro F1 = {best_single['CV_Macro_F1_Mean']:.4f})")
    for _, ens in ensemble_df.iterrows():
        diff = ens['CV_Macro_F1_Mean'] - best_single['CV_Macro_F1_Mean']
        sign = "+" if diff >= 0 else ""
        print(f"Ensemble ({ens['Model']}): Macro F1 = {ens['CV_Macro_F1_Mean']:.4f} ({sign}{diff:.4f} vs {best_single['Model']})")
        
    if best_model_name in ensemble_df['Model'].values:
        print("Conclusion: The ensemble outperformed individual single models.")
    else:
        print(f"Honest finding: The single model ({best_model_name}) achieved higher or comparable Macro F1 than the ensembles on this small sample (N={len(df)}). This occurs commonly in small clinical datasets where meta-models or voting across diverse classifiers can accumulate variance.")

    # Instantiate and fit best pipeline on all labelled data
    best_model_pipeline = build_all_pipelines()[best_model_name]
    best_model_pipeline.fit(X, y)
    
    # Save visualizations
    save_visualizations(
        res_df, best_model_name, best_model_pipeline, X, y, class_names, oof_preds, oof_trues, results_dir
    )
    
    # Save best model + metadata
    save_model_and_metadata(
        best_model_name, best_model_pipeline, X, y, class_names, res_df, models_dir
    )
    
    print("\n" + "=" * 70)
    print("                  STAGE 4 TRAINING COMPLETE")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print("=" * 70)


if __name__ == '__main__':
    main()
