import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===============================
# Load & Preprocess Data
# ===============================
df = pd.read_csv("kidney_disease.csv")
df.drop("id", axis=1, inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df.replace("?", np.nan, inplace=True)

df["classification"] = df["classification"].map({
    "ckd": 1,
    "ckd\t": 1,
    "notckd": 0
})

for col in ["pcv", "wc", "rc"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

binary_map = {
    "yes": 1, "no": 0,
    "present": 1, "notpresent": 0,
    "abnormal": 1, "normal": 0,
    "good": 1, "poor": 0
}

for col in cat_cols:
    df[col] = df[col].map(binary_map)

# ===============================
# Split Features / Target
# ===============================
X = df.drop("classification", axis=1)
y = df["classification"].values

MODEL_COLUMNS = X.columns.tolist()

# ===============================
# Train-Test Split + Scaling
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

MODEL_SCALER = StandardScaler()
X_train_scaled = MODEL_SCALER.fit_transform(X_train)
X_test_scaled = MODEL_SCALER.transform(X_test)

# ===============================
# Logistic Regression FROM SCRATCH
# ===============================
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.01, epochs=300):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)

        w -= lr * dw
        b -= lr * db

    return w, b

WEIGHTS, BIAS = train(X_train_scaled, y_train)

def predict_proba(X):
    return sigmoid(np.dot(X, WEIGHTS) + BIAS)

# ===============================
# Evaluation
# ===============================
def evaluate_model():
    y_pred = (predict_proba(X_test_scaled) >= 0.6).astype(int)

    accuracy = np.mean(y_pred == y_test)
    precision = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
    recall = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)

    return accuracy, precision, recall

MODEL_ACCURACY, MODEL_PRECISION, MODEL_RECALL = evaluate_model()

# ===============================
# Debug Print (اختياري)
# ===============================
print("Model Loaded Successfully")
print(f"Accuracy  : {MODEL_ACCURACY*100:.2f}%")
print(f"Precision : {MODEL_PRECISION*100:.2f}%")
print(f"Recall    : {MODEL_RECALL*100:.2f}%")
MODEL_COLUMNS = X.columns.tolist()
MODEL_SCALER = MODEL_SCALER
MODEL_ACCURACY = MODEL_ACCURACY
