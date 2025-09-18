import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_v3():
    df = pd.read_csv("data/heart_v4.csv", header=None)
    df.columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = df.replace("?", np.nan).apply(pd.to_numeric)
    df["target"] = (df["target"] > 0).astype(int)
    df = df.dropna()

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(scaler.fit_transform(X_train), y_train)
    y_pred = knn.predict(scaler.transform(X_test))

    acc_split = accuracy_score(y_test, y_pred)

    # Cross-validation evaluation
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring="accuracy")
    acc_cv = scores.mean()

    print("\n=== V3: Train-Test Split vs K-Fold CV ===")
    print(f"Train-Test Split Accuracy: {acc_split:.4f}")
    print(f"5-Fold CV Accuracy: {acc_cv:.4f}")
