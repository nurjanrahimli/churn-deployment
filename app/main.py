from datetime import datetime
import os
from fastapi import FastAPI
import pandas as pd
import joblib

from app.schemas import CustomerData

app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    description="Customer Churn Prediction API"
)

# Eğitilmiş pipeline'ı yükle
model = joblib.load("models/churn_pipeline.joblib")


@app.get("/")
def home():
    return {
        "message": "Churn Prediction API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    log_line = (
        f"{datetime.now()} | "
        f"Prediction={prediction} | "
        f"Probability={probability:.4f}\n"
    )

    with open("prediction_logs.txt", "a") as f:
        f.write(log_line)

    return {
        "prediction": int(prediction),
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4)
    }

