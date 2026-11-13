
import pandas as pd
import networkx as nx
import os

print("Loading Facebook social network...")

# Load cleaned Facebook dataset
edges = pd.read_csv("results/cleaned_edges.csv")

# Create social network graph
G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target"
)

print("Graph loaded successfully!")
print("Total users:", G.number_of_nodes())

# Function to recommend friends
def recommend_friends(user_id, top_n=10):

    if user_id not in G:
        print("User ID does not exist!")
        return pd.DataFrame()

    friends = set(G.neighbors(user_id))

    recommendations = []

    for candidate in G.nodes():

        # Skip the same user
        if candidate == user_id:
            continue

        # Skip existing friends
        if G.has_edge(user_id, candidate):
            continue

        candidate_friends = set(G.neighbors(candidate))

        # Calculate common friends
        common_friends = friends.intersection(candidate_friends)

        common_count = len(common_friends)

        if common_count > 0:

            # Calculate Jaccard similarity
            union_count = len(friends.union(candidate_friends))

            jaccard_score = (
                common_count / union_count
                if union_count > 0 else 0
            )

            recommendations.append({
                "User_ID": candidate,
                "Common_Friends": common_count,
                "Jaccard_Score": round(jaccard_score, 4)
            })

    result = pd.DataFrame(recommendations)

    if result.empty:
        print("No recommendations found!")
        return result

    # Sort by similarity score
    result = result.sort_values(
        by="Jaccard_Score",
        ascending=False
    )

    return result.head(top_n)


# Ask user for an ID
print("\nFRIEND RECOMMENDATION SYSTEM")
print("--------------------------------")

user_id = int(input("Enter Facebook User ID (0 to 4038): "))

# Generate recommendations
recommendations = recommend_friends(user_id)

if not recommendations.empty:

    print("\nRecommended Friends:")
    print(recommendations.to_string(index=False))

    # Save recommendations
    os.makedirs("results", exist_ok=True)

    recommendations.to_csv(
        "results/friend_recommendations.csv",
        index=False
    )

    print("\nRecommendations saved successfully!")
    print("File: results/friend_recommendations.csv")
