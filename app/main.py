from fastapi import FastAPI
import pandas as pd
import joblib

from app.schemas import CustomerData

app = FastAPI(title="Churn Prediction API")

# Modeli yükle
model_data = joblib.load("models/churn_pipeline.joblib")
model = model_data["model"]
feature_columns = model_data["columns"]


@app.get("/")
def home():
    return {"message": "Churn Prediction API çalışıyor"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: CustomerData):
    # Gelen veriyi dict'e çevir
    input_dict = data.model_dump()

    # DataFrame'e dönüştür
    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Eğitimdeki kolonlarla hizala
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Tahmin
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    label = "Churn" if int(prediction) == 1 else "No Churn"

    return {
        "prediction": int(prediction),
        "prediction_label": label,
        "churn_probability": round(float(probability), 4)
    }
