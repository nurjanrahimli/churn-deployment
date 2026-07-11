import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1) Dataset oku
df = pd.read_csv("telco_churn.csv")

# 2) Hedef değişkeni sayısallaştır
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 3) TotalCharges bazı datasetlerde string gelebiliyor
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 4) Kullanılmayacak kolon varsa çıkar
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# 5) X / y ayır
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 6) Numerik / kategorik kolonları bul
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 7) Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 8) Pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# 9) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 10) Train
model_pipeline.fit(X_train, y_train)

# 11) Test performansı
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 12) Kaydet
joblib.dump(model_pipeline, "app/model/churn_pipeline.pkl")
print("Model kaydedildi: app/model/churn_pipeline.pkl")
