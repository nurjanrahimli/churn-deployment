from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}   
    
    
def test_predict():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 70.0
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["Churn", "No Churn"]
    assert result["prediction_label"] == (
        "Churn" if result["prediction"] == 1 else "No Churn"
    )
    assert 0 <= result["churn_probability"] <= 1

