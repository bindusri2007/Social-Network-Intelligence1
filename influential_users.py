
import pandas as pd
import networkx as nx
import os

print("Loading social network graph...")

# Load cleaned Facebook connections
edges = pd.read_csv("results/cleaned_edges.csv")

# Create graph
G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target"
)

print("Graph loaded successfully!")
print("Total users:", G.number_of_nodes())

# 1. Degree Centrality
print("\nCalculating Degree Centrality...")
degree = nx.degree_centrality(G)

# 2. PageRank
print("Calculating PageRank...")
pagerank = nx.pagerank(G)

# 3. Closeness Centrality
print("Calculating Closeness Centrality...")
closeness = nx.closeness_centrality(G)

# 4. Betweenness Centrality
print("Calculating Betweenness Centrality...")
betweenness = nx.betweenness_centrality(
    G,
    k=min(200, G.number_of_nodes()),
    seed=42
)

# Create results table
results = []

for user in G.nodes():
    results.append({
        "User_ID": user,
        "Connections": G.degree(user),
        "Degree_Centrality": degree[user],
        "PageRank": pagerank[user],
        "Closeness_Centrality": closeness[user],
        "Betweenness_Centrality": betweenness[user]
    })

df = pd.DataFrame(results)

# Sort users by PageRank
df = df.sort_values(
    by="PageRank",
    ascending=False
)

# Create results folder
os.makedirs("results", exist_ok=True)

# Save all results
df.to_csv(
    "results/influential_users.csv",
    index=False
)

print("\nTOP 10 INFLUENTIAL USERS")
print("--------------------------------")

print(
    df.head(10).to_string(index=False)
)

print("\nAnalysis completed successfully!")
print("Results saved in results/influential_users.csv")
