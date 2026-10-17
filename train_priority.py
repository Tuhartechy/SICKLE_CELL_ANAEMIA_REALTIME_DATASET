"""
Train Priority Production Model on the Augmented Dataset.
Evaluates 80/20 test split and 5-Fold Stratified Cross-Validation.
Target Performance: 94% - 96% Accuracy.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import preprocess


def main():
    print("Loading augmented dataset (data/synthetic_5000.csv)...")
    df_syn = pd.read_csv('data/synthetic_5000.csv')
    features = [c for c in df_syn.columns if c not in ['Patient_ID', 'Pain_Score', 'Pain_Severity', 'Pain_Class', 'is_synthetic']]
    X = df_syn[features]
    y = df_syn['Pain_Class']

    preprocessor = preprocess.make_preprocessor()
    model = Pipeline([
        ('prep', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=75, max_depth=3, min_samples_leaf=15, random_state=42))
    ])

    print("Evaluating 80/20 Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_acc = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, average='macro'))
    print(f"--> Test Accuracy (80/20): {test_acc*100:.2f}%")
    print(f"--> Test Macro F1: {test_f1:.4f}")
    print("\nClassification Report (80/20 Test Set):")
    print(classification_report(y_test, y_pred))

    print("Running 5-Fold Stratified Cross-Validation on the 5,000 augmented cohort...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

    cv_acc = float(np.mean(acc_scores))
    cv_acc_std = float(np.std(acc_scores))
    cv_f1 = float(np.mean(f1_scores))
    cv_f1_std = float(np.std(f1_scores))

    print(f"--> 5-Fold CV Accuracy: {cv_acc*100:.2f}% +/- {cv_acc_std*100:.2f}%")
    print(f"--> 5-Fold CV Macro F1: {cv_f1:.4f} +/- {cv_f1_std:.4f}")

    print("Fitting model on full 5,000 dataset for deployment...")
    model.fit(X, y)

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Mild', 'Moderate', 'Severe'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Mild', 'Moderate', 'Severe'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f"Priority Model Confusion Matrix\nTest Acc: {test_acc*100:.1f}%, 5-Fold CV: {cv_acc*100:.1f}%")
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('sickle-pain/results', exist_ok=True)
    plt.savefig('results/confusion_matrix_priority_augmented.png', dpi=300)
    plt.savefig('sickle-pain/results/confusion_matrix_priority_augmented.png', dpi=300)
    plt.close()

    # Metadata
    metadata = {
        'project_title': 'Ensemble Machine Learning Approach for Predicting Pain Severity in Sickle Cell Disease (Priority Production Pipeline)',
        'best_model_name': 'Random Forest (Priority Production Pipeline)',
        'test_accuracy_80_20': round(test_acc, 4),
        'test_macro_f1_80_20': round(test_f1, 4),
        'cv_accuracy_mean': round(cv_acc, 4),
        'cv_accuracy_std': round(cv_acc_std, 4),
        'cv_macro_f1_mean': round(cv_f1, 4),
        'cv_macro_f1_std': round(cv_f1_std, 4),
        'augmentation_method': 'SMOTE + Clinical Bounds',
        'total_training_rows': len(df_syn),
        'synthetic_training_rows': len(df_syn),
        'target_column': 'Pain_Class',
        'class_names': ['Mild', 'Moderate', 'Severe'],
        'label_mapping': {'0': 'Mild', '1': 'Moderate', '2': 'Severe'},
        'feature_names': features,
        'numeric_features': [c for c in features if c not in ['Sex', 'Genotype']],
        'categorical_features': ['Sex', 'Genotype'],
        'feature_defaults': {
            'Age': 25, 'Heart_Rate': 90, 'Respiratory_Rate': 20, 'SpO2': 98.0,
            'Temperature': 98.6, 'BP_Systolic': 110, 'BP_Diastolic': 70,
            'Hemoglobin': 8.5, 'WBC': 9000.0, 'Platelets': 250000.0, 'RDW': 20.0,
            'Bilirubin_Total': 2.0, 'Creatinine': 0.55, 'Sex': 'F', 'Genotype': 'SS'
        }
    }

    for d in ['models', 'sickle-pain/models']:
        os.makedirs(d, exist_ok=True)
        joblib.dump(model, os.path.join(d, 'best_model_augmented.joblib'))
        with open(os.path.join(d, 'metadata_augmented.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    print("Model and metadata saved successfully!")


if __name__ == '__main__':
    main()
