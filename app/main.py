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

    print("Gelen kolonlar:")
    print(input_df.columns.tolist())

    print("Model beklediği kolonlar:")
    print(model.feature_names_in_)

    prediction = model.predict(input_df)[0]

    return {"prediction": int(prediction)}
