
import pandas as pd
import networkx as nx
import os

print("Loading Facebook social network...")

# Load cleaned dataset
edges = pd.read_csv("results/cleaned_edges.csv")

# Construct graph
G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target"
)

print("Graph loaded successfully!")
print("Total users:", G.number_of_nodes())
print("Total connections:", G.number_of_edges())

# Apply Louvain community detection
print("\nDetecting communities using Louvain algorithm...")

communities = nx.community.louvain_communities(
    G,
    seed=42
)

print("\nCommunity detection completed!")

# Assign community IDs to users
user_communities = []

for community_id, community in enumerate(communities, start=1):

    for user in community:
        user_communities.append({
            "User_ID": user,
            "Community_ID": community_id
        })

# Convert results into DataFrame
community_df = pd.DataFrame(user_communities)

# Create community summary
summary = (
    community_df.groupby("Community_ID")
    .size()
    .reset_index(name="Number_of_Users")
)

# Calculate modularity
modularity = nx.community.modularity(G, communities)

print("\nCOMMUNITY ANALYSIS RESULTS")
print("--------------------------------")

print("Total communities:", len(communities))
print("Modularity score:", round(modularity, 4))

print("\nCommunity sizes:")
print(summary.to_string(index=False))

# Save results
os.makedirs("results", exist_ok=True)

community_df.to_csv(
    "results/user_communities.csv",
    index=False
)

summary.to_csv(
    "results/community_summary.csv",
    index=False
)

print("\nFiles saved successfully!")
print("1. results/user_communities.csv")
print("2. results/community_summary.csv")
