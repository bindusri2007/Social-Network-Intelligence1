
import os
import pandas as pd

print("Starting data preprocessing...")

# Load the original Facebook dataset
edges = pd.read_csv(
    "data/facebook_combined.txt.gz",
    sep=" ",
    compression="gzip",
    header=None,
    names=["source", "target"]
)

print("\nOriginal dataset loaded!")
print("Original records:", len(edges))

# 1. Remove missing values
edges = edges.dropna()

# 2. Convert user IDs into integers
edges["source"] = edges["source"].astype(int)
edges["target"] = edges["target"].astype(int)

# 3. Remove self-connections
edges = edges[edges["source"] != edges["target"]].copy()

# 4. Arrange each connection in a standard order
edges["user1"] = edges[["source", "target"]].min(axis=1)
edges["user2"] = edges[["source", "target"]].max(axis=1)

# 5. Remove duplicate and reverse-direction connections
edges = edges.drop_duplicates(subset=["user1", "user2"])

# 6. Keep only the cleaned columns
cleaned_data = edges[["user1", "user2"]].copy()

cleaned_data.columns = ["source", "target"]

# 7. Create results folder if necessary
os.makedirs("results", exist_ok=True)

# 8. Save cleaned dataset
cleaned_data.to_csv(
    "results/cleaned_edges.csv",
    index=False
)

# 9. Display results
print("\nDATA CLEANING COMPLETED!")
print("--------------------------------")

print("Cleaned connections:", len(cleaned_data))

total_users = pd.unique(cleaned_data.values.ravel()).size

print("Unique users:", total_users)

print("\nFirst 5 cleaned records:")
print(cleaned_data.head())

print("\nCleaned dataset saved successfully!")
print("Location: results/cleaned_edges.csv")
