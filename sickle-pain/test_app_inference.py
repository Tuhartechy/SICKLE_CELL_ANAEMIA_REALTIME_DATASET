import os
import json
import joblib
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, 'models', 'best_model.joblib'))
with open(os.path.join(base_dir, 'models', 'metadata.json')) as f:
    meta = json.load(f)

# Example 1: Acute Crisis Patient (Severe signs: high HR, high RR, low Hb, high WBC, low SpO2)
patient1 = pd.DataFrame([{
    'Age': 28,
    'Heart_Rate': 125,
    'Respiratory_Rate': 28,
    'SpO2': 93.0,
    'Temperature': 101.2,
    'BP_Systolic': 105,
    'BP_Diastolic': 65,
    'Hemoglobin': 5.2,
    'WBC': 18500.0,
    'Platelets': 85000.0,
    'RDW': 25.0,
    'Bilirubin_Total': 6.2,
    'Creatinine': 1.1,
    'Sex': 'F',
    'Genotype': 'SS'
}])

# Example 2: Stable Outpatient with missing secondary labs (imputed by pipeline)
patient2 = pd.DataFrame([{
    'Age': 40,
    'Heart_Rate': 74,
    'Respiratory_Rate': 16,
    'SpO2': 99.0,
    'Temperature': 98.6,
    'BP_Systolic': 120,
    'BP_Diastolic': 80,
    'Hemoglobin': 12.5,
    'WBC': 6200.0,
    'Platelets': 280000.0,
    'RDW': 14.5,
    'Bilirubin_Total': np.nan,  # missing lab test
    'Creatinine': np.nan,       # missing lab test
    'Sex': 'M',
    'Genotype': 'AS'
}])

print("=== TESTING EXAMPLE 1: Acute Crisis Patient ===")
pred1_idx = model.predict(patient1)[0]
probas1 = model.predict_proba(patient1)[0]
class_names = meta['class_names']
print(f"Predicted Class: {class_names[pred1_idx]}")
for cls, prob in zip(class_names, probas1):
    print(f"  {cls}: {prob*100:.2f}%")

print("\n=== TESTING EXAMPLE 2: Stable Patient with Missing Labs ===")
pred2_idx = model.predict(patient2)[0]
probas2 = model.predict_proba(patient2)[0]
print(f"Predicted Class: {class_names[pred2_idx]}")
for cls, prob in zip(class_names, probas2):
    print(f"  {cls}: {prob*100:.2f}%")

print("\nApp inference test successful!")
