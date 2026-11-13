
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

print("Loading cleaned Facebook dataset...")

# Load cleaned connections
edges = pd.read_csv("results/cleaned_edges.csv")

# Create an undirected social network graph
G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target"
)

# Display graph information
print("\nGRAPH CONSTRUCTION COMPLETED!")
print("--------------------------------")

print("Total Users (Nodes):", G.number_of_nodes())
print("Total Friendships (Edges):", G.number_of_edges())

print("Is the graph directed?", nx.is_directed(G))
print("Is the graph connected?", nx.is_connected(G))

# Calculate average degree
average_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()

print("Average Connections per User:", round(average_degree, 2))

# Save graph information
os.makedirs("results", exist_ok=True)

graph_info = pd.DataFrame({
    "Metric": [
        "Total Users",
        "Total Friendships",
        "Average Connections",
        "Network Density"
    ],
    "Value": [
        G.number_of_nodes(),
        G.number_of_edges(),
        round(average_degree, 2),
        round(nx.density(G), 5)
    ]
})

graph_info.to_csv("results/graph_statistics.csv", index=False)

# Draw a small sample of the network
print("\nGenerating network visualization...")

sample_nodes = list(G.nodes())[:100]
H = G.subgraph(sample_nodes)

plt.figure(figsize=(12, 8))

pos = nx.spring_layout(H, seed=42)

nx.draw_networkx(
    H,
    pos,
    node_size=40,
    with_labels=False,
    edge_color="gray",
    node_color="skyblue"
)

plt.title("Facebook Social Network - Sample Graph")
plt.axis("off")

plt.savefig("results/network_graph.png", dpi=300)
plt.show()

print("\nGraph statistics saved!")
print("Graph image saved in results/network_graph.png")
