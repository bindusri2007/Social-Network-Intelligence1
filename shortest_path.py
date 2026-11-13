
import pandas as pd
import networkx as nx
import os

print("Loading Facebook social network...")

# Load cleaned dataset
edges = pd.read_csv("results/cleaned_edges.csv")

# Create graph
G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target"
)

print("Graph loaded successfully!")
print("Total users:", G.number_of_nodes())

# Function to find shortest path
def find_shortest_path(user1, user2):

    if user1 not in G:
        print("First user ID does not exist!")
        return None

    if user2 not in G:
        print("Second user ID does not exist!")
        return None

    try:
        path = nx.shortest_path(
            G,
            source=user1,
            target=user2
        )

        return path

    except nx.NetworkXNoPath:
        print("No connection path exists between these users.")
        return None


# Get input from user
print("\nSHORTEST PATH ANALYSIS")
print("--------------------------------")

try:
    user1 = int(input("Enter first User ID: "))
    user2 = int(input("Enter second User ID: "))

    if user1 == user2:
        print("Please enter two different user IDs.")

    else:
        path = find_shortest_path(user1, user2)

        if path is not None:

            print("\nShortest friendship path:")
            print(" -> ".join(map(str, path)))

            print("\nNumber of users in path:", len(path))
            print("Number of friendship connections:", len(path) - 1)

            # Save path results
            os.makedirs("results", exist_ok=True)

            path_data = pd.DataFrame({
                "Step": range(1, len(path) + 1),
                "User_ID": path
            })

            path_data.to_csv(
                "results/shortest_path.csv",
                index=False
            )

            print("\nShortest path saved successfully!")
            print("File: results/shortest_path.csv")

except ValueError:
    print("Please enter valid integer User IDs.")
