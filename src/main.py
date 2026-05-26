# src/main.py
import pandas as pd
import joblib
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

print("Загрузка модели...")
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "rb") as f:
    FEATURE_COLUMNS = pickle.load(f)
print(f"✅ Модель загружена. {len(FEATURE_COLUMNS)} признаков")

# Словарь для возраста
AGE_MAPPING = {
    '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3,
    '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7,
    '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11,
    '80 or older': 12
}

class PatientData(BaseModel):
    BMI: float
    Smoking: str
    AlcoholDrinking: str
    Stroke: str
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: str
    Sex: str
    AgeCategory: str
    Race: str
    Diabetic: str
    PhysicalActivity: str
    GenHealth: str
    SleepTime: float
    Asthma: str
    KidneyDisease: str
    SkinCancer: str

def preprocess_patient(patient: PatientData):
    data = {
        'BMI': [patient.BMI],
        'Smoking': [patient.Smoking],
        'AlcoholDrinking': [patient.AlcoholDrinking],
        'Stroke': [patient.Stroke],
        'PhysicalHealth': [patient.PhysicalHealth],
        'MentalHealth': [patient.MentalHealth],
        'DiffWalking': [patient.DiffWalking],
        'Sex': [patient.Sex],
        'AgeCategory': [patient.AgeCategory],
        'Race': [patient.Race],
        'Diabetic': [patient.Diabetic],
        'PhysicalActivity': [patient.PhysicalActivity],
        'GenHealth': [patient.GenHealth],
        'SleepTime': [patient.SleepTime],
        'Asthma': [patient.Asthma],
        'KidneyDisease': [patient.KidneyDisease],
        'SkinCancer': [patient.SkinCancer]
    }
    df = pd.DataFrame(data)
    df['Age_num'] = df['AgeCategory'].map(AGE_MAPPING)
    df['Obesity'] = (df['BMI'] > 30).astype(int)
    df['RiskFactor'] = ((df['Smoking'] == 'Yes').astype(int) +
                        (df['Diabetic'] == 'Yes').astype(int) +
                        (df['PhysicalActivity'] == 'No').astype(int))
    df['Age_BMI'] = df['Age_num'] * df['BMI']
    df = pd.get_dummies(df, drop_first=True)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLUMNS]
    return df

app = FastAPI(title="Heart Disease Prediction API")

@app.get("/")
def root():
    return {"status": "running", "message": "Heart Disease API"}

@app.get("/health")
def health():
    return {"status": "healthy", "n_features": len(FEATURE_COLUMNS)}

@app.post("/predict")
def predict(patient: PatientData):
    try:
        df = preprocess_patient(patient)
        probability = float(model.predict_proba(df)[0][1])
        prediction = int(probability > 0.5)
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "message": "ВЫСОКИЙ РИСК" if prediction == 1 else "НИЗКИЙ РИСК"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))