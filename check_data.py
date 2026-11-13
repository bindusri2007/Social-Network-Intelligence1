
import pandas as pd

print("Loading Facebook dataset...")

# Read the compressed dataset
edges = pd.read_csv(
    "data/facebook_combined.txt.gz",
    sep=" ",
    compression="gzip",
    header=None,
    names=["source", "target"]
)

# Display first five records
print("\nFirst 5 records:")
print(edges.head())

# Count connections
print("\nTotal connections:", len(edges))

# Count unique users
total_users = pd.unique(edges.values.ravel()).size

print("Total unique users:", total_users)

# Display dataset information
print("\nDataset information:")
print(edges.info())

print("\nDataset loaded successfully!")
