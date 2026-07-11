import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📉 Telco Churn Prediction")
st.write("Müşteri bilgilerini gir ve churn tahmini al.")

# Form alanları
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.5)
total_charges = st.number_input("Total Charges", min_value=0.0, value=850.2)
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

if st.button("Tahmin Yap"):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "InternetService": internet_service,
        "Contract": contract,
        "PaymentMethod": payment_method
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success("Tahmin başarılı!")
            st.write(f"**Prediction:** {result['prediction_label']}")
            st.write(f"**Churn Probability:** {result['churn_probability']}")

            if result["prediction"] == 1:
                st.error("Bu müşteri churn etme riski taşıyor.")
            else:
                st.info("Bu müşteri churn etmeyebilir.")
        else:
            st.error(f"API hatası: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error("FastAPI servisine bağlanılamadı.")
        st.write(str(e))
