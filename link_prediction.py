
import os
import random

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("Loading Facebook social network...", flush=True)

edges = pd.read_csv("results/cleaned_edges.csv")

G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target"
)

print("Graph loaded successfully!", flush=True)
print("Total users:", G.number_of_nodes())
print("Total connections:", G.number_of_edges())

# --------------------------------------------------
# 2. SPLIT EXISTING CONNECTIONS
# --------------------------------------------------

random.seed(42)
np.random.seed(42)

all_edges = list(G.edges())

train_edges, test_edges = train_test_split(
    all_edges,
    test_size=0.20,
    random_state=42
)

train_graph = G.copy()
train_graph.remove_edges_from(test_edges)

print("\nTraining graph created!")
print("Training connections:", train_graph.number_of_edges())
print("Testing connections:", len(test_edges))

# --------------------------------------------------
# 3. CREATE ADJACENCY SETS
# --------------------------------------------------

print("\nPreparing graph connections...", flush=True)

adjacency = {
    user: set(train_graph.neighbors(user))
    for user in train_graph.nodes()
}

# --------------------------------------------------
# 4. GENERATE NEGATIVE SAMPLES
# --------------------------------------------------

print("\nGenerating negative samples...", flush=True)

nodes = list(G.nodes())
negative_edges = set()

while len(negative_edges) < len(all_edges):

    u, v = random.sample(nodes, 2)

    if not G.has_edge(u, v):
        pair = tuple(sorted((u, v)))
        negative_edges.add(pair)

negative_edges = list(negative_edges)

print("Negative samples generated:", len(negative_edges))

# --------------------------------------------------
# 5. FEATURE EXTRACTION FUNCTION
# --------------------------------------------------

def extract_features(graph_adjacency, u, v):

    friends_u = graph_adjacency[u]
    friends_v = graph_adjacency[v]

    common = len(friends_u & friends_v)

    union = len(friends_u | friends_v)

    jaccard = common / union if union > 0 else 0

    preferential = len(friends_u) * len(friends_v)

    return [
        common,
        jaccard,
        preferential
    ]

# --------------------------------------------------
# 6. PREPARE MACHINE LEARNING DATASET
# --------------------------------------------------

print("\nPreparing machine learning dataset...", flush=True)

X = []
y = []

print("Processing positive connections...")

for u, v in all_edges:

    X.append(
        extract_features(adjacency, u, v)
    )

    y.append(1)

print("Processing negative connections...")

for u, v in negative_edges:

    X.append(
        extract_features(adjacency, u, v)
    )

    y.append(0)

X = np.array(X, dtype=float)
y = np.array(y)

print("\nMachine learning dataset prepared!")
print("Total samples:", len(X))

# --------------------------------------------------
# 7. SPLIT MACHINE LEARNING DATASET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 8. TRAIN RANDOM FOREST MODEL
# --------------------------------------------------

print("\nTraining Random Forest model...", flush=True)

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

print("Model training completed!", flush=True)

# --------------------------------------------------
# 9. EVALUATE MODEL
# --------------------------------------------------

print("\nEvaluating model...", flush=True)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(y_test, probabilities)

print("\nMODEL PERFORMANCE")
print("--------------------------------")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(auc, 4))

# --------------------------------------------------
# 10. GENERATE CANDIDATE CONNECTIONS
# --------------------------------------------------

print("\nPredicting possible new connections...", flush=True)

MAX_CANDIDATES = 200000

candidate_pairs = set()

candidate_users = list(train_graph.nodes())
random.shuffle(candidate_users)

print("Generating candidate connections...", flush=True)

for user in candidate_users:

    direct_friends = adjacency[user]

    candidates = set()

    for friend in direct_friends:

        candidates.update(adjacency[friend])

    candidates.discard(user)
    candidates -= direct_friends

    for candidate in candidates:

        pair = tuple(sorted((user, candidate)))

        candidate_pairs.add(pair)

        if len(candidate_pairs) >= MAX_CANDIDATES:
            break

    if len(candidate_pairs) >= MAX_CANDIDATES:
        break

candidate_pairs = list(candidate_pairs)

print(
    "Candidate connections generated:",
    len(candidate_pairs),
    flush=True
)

# --------------------------------------------------
# 11. CALCULATE CANDIDATE FEATURES
# --------------------------------------------------

print("\nCalculating prediction features...", flush=True)

all_features = []

for u, v in candidate_pairs:

    features = extract_features(
        adjacency,
        u,
        v
    )

    all_features.append(features)

all_features = np.array(all_features, dtype=float)

print("Feature calculation completed!")

# --------------------------------------------------
# 12. PREDICT CONNECTIONS IN BATCHES
# --------------------------------------------------

print("\nPredicting connections in batches...", flush=True)

BATCH_SIZE = 10000

predictions_list = []

for start in range(0, len(candidate_pairs), BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        len(candidate_pairs)
    )

    batch_features = all_features[start:end]

    batch_probabilities = model.predict_proba(
        batch_features
    )[:, 1]

    for i, probability in enumerate(batch_probabilities):

        u, v = candidate_pairs[start + i]

        predictions_list.append({
            "User_1": u,
            "User_2": v,
            "Connection_Probability": round(
                float(probability), 4
            )
        })

    print(
        f"Processed {end} / {len(candidate_pairs)} candidates",
        flush=True
    )

# --------------------------------------------------
# 13. SELECT TOP 20 PREDICTED CONNECTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame(predictions_list)

if not prediction_df.empty:

    prediction_df = prediction_df.sort_values(
        by="Connection_Probability",
        ascending=False
    )

    final_predictions = prediction_df.head(20).copy()

else:

    final_predictions = pd.DataFrame(
        columns=[
            "User_1",
            "User_2",
            "Connection_Probability"
        ]
    )

# --------------------------------------------------
# 14. SAVE RESULTS
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

metrics_df.to_csv(
    "results/link_prediction_metrics.csv",
    index=False
)

final_predictions.to_csv(
    "results/predicted_links.csv",
    index=False
)

# --------------------------------------------------
# 15. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\nTOP PREDICTED CONNECTIONS")
print("--------------------------------")

if not final_predictions.empty:

    print(
        final_predictions.to_string(index=False)
    )

else:

    print("No candidate connections found.")

print("\nFiles saved successfully!")

print("1. results/link_prediction_metrics.csv")
print("2. results/predicted_links.csv")

print("\nLINK PREDICTION COMPLETED!", flush=True)
