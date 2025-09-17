import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

datasets = {
    "V1 (Hungary)": "data/heart_v1.csv",
    "V3 (Switzerland)": "data/heart_v3.csv",
    "V4 (Cleveland)": "data/heart_v4.csv"
}

results = []

def load_and_clean(path):
    df = pd.read_csv(path, header=None)
    df.columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric)

    # Convert target to binary
    df["target"] = (df["target"] > 0).astype(int)

    # Impute missing values with column median instead of dropping
    df = df.fillna(df.median(numeric_only=True))

    return df

for name, file in datasets.items():
    print(f"\n=== Running on {name} ===")
    df = load_and_clean(file)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-validation to find best k
    k_values = range(1, 21)
    cv_scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="accuracy")
        cv_scores.append(scores.mean())

    best_k = k_values[cv_scores.index(max(cv_scores))]

    # Retrain with best k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"Best k = {best_k}, Test Accuracy = {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results.append([name, best_k, acc, len(df)])

print("\n=== Summary Table ===")
summary = pd.DataFrame(results, columns=["Dataset", "Best k", "Test Accuracy", "Samples"])
print(summary)
