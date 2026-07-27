import joblib
import pandas as pd
import numpy as np

model  = joblib.load('model_artifacts/model.pkl')
scaler = joblib.load('model_artifacts/scaler.pkl')

SCALER_COLS = [
    'age', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level', 'gender_enc',
    'smoke_No Info', 'smoke_current', 'smoke_ever',
    'smoke_former', 'smoke_never', 'smoke_not current'
]

TOP_FEATURES = [
    'HbA1c_level', 'blood_glucose_level', 'age', 'bmi',
    'hypertension', 'smoke_No Info', 'heart_disease', 'smoke_former'
]

def predict(data):
    df     = pd.DataFrame([data])[SCALER_COLS]
    scaled = pd.DataFrame(scaler.transform(df), columns=SCALER_COLS)
    final  = scaled[TOP_FEATURES]
    proba  = model.predict_proba(final)[0][1]
    label  = "🔴 High" if proba >= 0.6 else "🟡 Medium" if proba >= 0.3 else "🟢 Low"
    return proba, label

# Scan HbA1c and glucose combinations to find medium risk zone
print("Scanning for MEDIUM RISK Zone (30–60%)...")
print(f"{'HbA1c':>8} {'Glucose':>9} {'Probability':>13} {'Risk':>10}")
print("-" * 45)

for hba1c in [5.5, 5.8, 6.0, 6.2, 6.4, 6.5, 6.8]:
    for glucose in [100, 115, 130, 140, 155, 160]:
        data = {
            'age': 45.0, 'hypertension': 0.0, 'heart_disease': 0.0,
            'bmi': 28.0, 'HbA1c_level': hba1c, 'blood_glucose_level': float(glucose),
            'gender_enc': 1.0, 'smoke_No Info': 0.0, 'smoke_current': 0.0,
            'smoke_ever': 0.0, 'smoke_former': 0.0, 'smoke_never': 1.0,
            'smoke_not current': 0.0
        }
        proba, label = predict(data)
        if 0.20 <= proba <= 0.75:  # show boundary zone
            print(f"{hba1c:>8.1f} {glucose:>9} {proba*100:>12.1f}%  {label:>10}")

print("Full Probability Map:")
print(f"{'HbA1c':>8} {'Glucose':>9} {'Probability':>13}")
print("-" * 35)
for hba1c in [4.5, 5.0, 5.5, 6.0, 6.5, 6.8, 7.0, 7.5, 8.0]:
    for glucose in [85, 100, 130, 160, 200]:
        data = {
            'age': 45.0, 'hypertension': 0.0, 'heart_disease': 0.0,
            'bmi': 28.0, 'HbA1c_level': hba1c, 'blood_glucose_level': float(glucose),
            'gender_enc': 1.0, 'smoke_No Info': 0.0, 'smoke_current': 0.0,
            'smoke_ever': 0.0, 'smoke_former': 0.0, 'smoke_never': 1.0,
            'smoke_not current': 0.0
        }
        proba, label = predict(data)
        print(f"{hba1c:>8.1f} {glucose:>9} {proba*100:>12.1f}%  {label}")

