# tugas2_knn.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ------------------------------------
# Training model sekali saat import
# ------------------------------------
df = pd.read_csv("data/stunting_wasting_dataset.csv")

gender_col = "Jenis Kelamin"
age_col = "Umur (bulan)"
height_col = "Tinggi Badan (cm)"
weight_col = "Berat Badan (kg)"
stunting_col = "Stunting"
wasting_col = "Wasting"

# encode gender
df[gender_col] = df[gender_col].astype(str).str.lower().map({
    "l": 0, "laki-laki": 0, "laki laki": 0, "male": 0, "m": 0,
    "p": 1, "perempuan": 1, "female": 1, "f": 1
})

df = df.dropna(subset=[
    gender_col, age_col, height_col, weight_col,
    stunting_col, wasting_col
])

X = df[[gender_col, age_col, height_col, weight_col]]

y_stunting = df[stunting_col].astype(str)
y_wasting = df[wasting_col].astype(str)

X_train, X_test, y_train_s, y_test_s = train_test_split(
    X, y_stunting, test_size=0.2, random_state=42
)

_, _, y_train_w, y_test_w = train_test_split(
    X, y_wasting, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# default K=3
knn_stunting = KNeighborsClassifier(n_neighbors=3)
knn_stunting.fit(X_train_scaled, y_train_s)

knn_wasting = KNeighborsClassifier(n_neighbors=3)
knn_wasting.fit(X_train_scaled, y_train_w)

# ------------------------------------
# ✅ Evaluasi Model (Akurasi, Precision, Recall, F1)
# ------------------------------------
y_pred_s = knn_stunting.predict(X_test_scaled)
y_pred_w = knn_wasting.predict(X_test_scaled)

evaluation = {
    "stunting": {
        "accuracy": accuracy_score(y_test_s, y_pred_s),
        "precision": precision_score(y_test_s, y_pred_s, average='macro', zero_division=0),
        "recall": recall_score(y_test_s, y_pred_s, average='macro', zero_division=0),
        "f1": f1_score(y_test_s, y_pred_s, average='macro', zero_division=0),
        "report": classification_report(y_test_s, y_pred_s, zero_division=0)
    },
    "wasting": {
        "accuracy": accuracy_score(y_test_w, y_pred_w),
        "precision": precision_score(y_test_w, y_pred_w, average='macro', zero_division=0),
        "recall": recall_score(y_test_w, y_pred_w, average='macro', zero_division=0),
        "f1": f1_score(y_test_w, y_pred_w, average='macro', zero_division=0),
        "report": classification_report(y_test_w, y_pred_w, zero_division=0)
    }
}

# ------------------------------------
# Fungsi Prediksi untuk Flask
# ------------------------------------
def predict_stunting_wasting(gender, age, height, weight, k):
    model_s = KNeighborsClassifier(n_neighbors=k)
    model_w = KNeighborsClassifier(n_neighbors=k)

    model_s.fit(X_train_scaled, y_train_s)
    model_w.fit(X_train_scaled, y_train_w)

    X_user = np.array([[gender, age, height, weight]])
    X_user_scaled = scaler.transform(X_user)

    stunting_pred = model_s.predict(X_user_scaled)[0]
    wasting_pred = model_w.predict(X_user_scaled)[0]

    return stunting_pred, wasting_pred
