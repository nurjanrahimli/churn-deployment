import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1) Veriyi oku
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2) total charges temizliği
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 3) customerID işimize yaramaz, çıkar
df = df.drop("customerID", axis=1)

# 4) hedef değişken
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 5) X-y ayır
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 6) kategorik kolonları one-hot encode et
X = pd.get_dummies(X, drop_first=True)

# Eğitimde oluşan kolon sırasını saklayacağız
feature_columns = X.columns.tolist()

# 7) train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8) modeli eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9) test doğruluğu
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model accuracy: {acc:.4f}")

# 10) model + kolonları birlikte kaydet
joblib.dump(
    {
        "model": model,
        "columns": feature_columns
    },
    "model/churn_model.pkl"
)

print("Model kaydedildi: model/churn_model.pkl")
