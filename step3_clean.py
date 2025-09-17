import pandas as pd
import numpy as np

# Load again
df = pd.read_csv("data/heart_v4.csv", header=None)

# Assign column names (from UCI docs)
df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Replace '?' with NaN and convert to numeric
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)

# Convert target to binary: 0 = no disease, >0 = disease
df["target"] = (df["target"] > 0).astype(int)

print(df.info())
print("\nClass distribution:\n", df["target"].value_counts())
print("\nFirst 5 rows:\n", df.head())
