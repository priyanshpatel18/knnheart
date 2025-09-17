import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load and clean dataset
df = pd.read_csv("data/heart_v4.csv", header=None)
df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)
df["target"] = (df["target"] > 0).astype(int)
df = df.dropna()

X = df.drop("target", axis=1)
y = df["target"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Search for best k using 5-fold CV
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = k_values[cv_scores.index(max(cv_scores))]
print("Best k found via CV:", best_k)
print("Corresponding CV accuracy:", max(cv_scores))

# Plot accuracy vs k
plt.figure(figsize=(8,4))
plt.plot(k_values, cv_scores, marker="o")
plt.xlabel("k (n_neighbors)")
plt.ylabel("CV Accuracy")
plt.title("Choosing k for KNN")
plt.grid(True)
plt.show()
