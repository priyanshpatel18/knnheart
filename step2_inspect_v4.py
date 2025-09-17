import pandas as pd

# Load the dataset (Cleveland processed)
df = pd.read_csv("data/heart_v4.csv", header=None)

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())
print("\nUnique values in last column (assumed target):", df.iloc[:, -1].unique())
