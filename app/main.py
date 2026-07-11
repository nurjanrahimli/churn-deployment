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

    # İstek verisini DataFrame'e çevir
    input_df = pd.DataFrame([data.model_dump()])

    # Tahmin
    prediction = model.predict(input_df)[0]

    # Olasılık
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4)
    }
