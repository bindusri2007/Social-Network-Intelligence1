
import os
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Social Network Intelligence",
    page_icon="🌐",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #17365D;
}
h2, h3 {
    color: #24527A;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e1e7ef;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

RESULTS = "results"

FILES = {
    "edges": "cleaned_edges.csv",
    "statistics": "graph_statistics.csv",
    "influential": "influential_users.csv",
    "communities": "user_communities.csv",
    "community_summary": "community_summary.csv",
    "recommendations": "friend_recommendations.csv",
    "shortest_path": "shortest_path.csv",
    "prediction_metrics": "link_prediction_metrics.csv",
    "predicted_links": "predicted_links.csv"
}

# --------------------------------------------------
# LOAD CSV FILES
# --------------------------------------------------

@st.cache_data
def load_csv(filename):

    path = os.path.join(RESULTS, filename)

    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


edges = load_csv(FILES["edges"])
statistics = load_csv(FILES["statistics"])
influential = load_csv(FILES["influential"])
communities = load_csv(FILES["communities"])
community_summary = load_csv(FILES["community_summary"])
recommendations = load_csv(FILES["recommendations"])
shortest_path = load_csv(FILES["shortest_path"])
prediction_metrics = load_csv(FILES["prediction_metrics"])
predicted_links = load_csv(FILES["predicted_links"])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🌐 Network Intelligence")

st.sidebar.markdown(
    "### Social Network Analytics System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Network Visualization",
        "Influential Users",
        "Community Detection",
        "Friend Recommendations",
        "Shortest Path Analysis",
        "Link Prediction",
        "Model Performance",
        "Download Reports"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Big Data Analytics Project\n\n"
    "Facebook Social Network Dataset\n\n"
    "Technology: Python, NetworkX, "
    "Pandas, Scikit-learn and Streamlit"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌐 Social Network Intelligence System")

st.markdown(
    "### Advanced Graph Analytics and Machine Learning Dashboard"
)

st.markdown("---")

# --------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    if edges.empty:

        st.error(
            "Dataset not found. Please check results/cleaned_edges.csv."
        )

    else:

        total_connections = len(edges)

        if {"source", "target"}.issubset(edges.columns):

            total_users = pd.unique(
                pd.concat([
                    edges["source"],
                    edges["target"]
                ])
            ).size

        else:
            total_users = 0

        total_communities = (
            communities.iloc[:, 1].nunique()
            if communities.shape[1] >= 2
            else 0
        )

        total_recommendations = len(recommendations)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Users", f"{total_users:,}")
        col2.metric("Total Connections", f"{total_connections:,}")
        col3.metric("Communities", f"{total_communities:,}")
        col4.metric(
            "Friend Recommendations",
            f"{total_recommendations:,}"
        )

    st.markdown("---")

    st.subheader("📌 Project Information")

    st.write("""
    This project analyzes a Facebook social network dataset
    using graph analytics and machine learning.

    The system identifies influential users, discovers
    communities, recommends potential friendships, analyzes
    shortest paths, and predicts possible new connections.
    """)

    st.subheader("📂 Available Analysis Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.success("✔ Network Statistics")
        st.success("✔ Influential User Identification")
        st.success("✔ Community Detection")
        st.success("✔ Friend Recommendations")

    with col2:
        st.success("✔ Shortest Path Analysis")
        st.success("✔ Link Prediction")
        st.success("✔ Random Forest Classification")
        st.success("✔ Data Visualization")

    if not statistics.empty:

        st.subheader("Network Statistics")

        st.dataframe(
            statistics,
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# PAGE 2: NETWORK VISUALIZATION
# --------------------------------------------------

elif page == "Network Visualization":

    st.header("🕸️ Social Network Visualization")

    graph_image = os.path.join(
        RESULTS,
        "network_graph.png"
    )

    if os.path.exists(graph_image):

        st.image(
            graph_image,
            caption="Facebook Social Network Graph",
            use_container_width=True
        )

    else:

        st.warning(
            "Network graph image not found in the results folder."
        )

    if not edges.empty:

        st.subheader("Connection Data")

        st.dataframe(
            edges.head(100),
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# PAGE 3: INFLUENTIAL USERS
# --------------------------------------------------

elif page == "Influential Users":

    st.header("⭐ Influential Users")

    if influential.empty:

        st.warning("Influential user data is not available.")

    else:

        st.write(
            "Users ranked by the influence measures "
            "calculated during graph analysis."
        )

        st.dataframe(
            influential.head(50),
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = influential.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            selected_column = st.selectbox(
                "Select metric for visualization",
                numeric_columns
            )

            chart_data = influential.head(20)

            fig = px.bar(
                chart_data,
                x=chart_data.columns[0],
                y=selected_column,
                title=f"Top 20 Users - {selected_column}",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# --------------------------------------------------
# PAGE 4: COMMUNITY DETECTION
# --------------------------------------------------

elif page == "Community Detection":

    st.header("👥 Community Detection")

    if not communities.empty:

        st.subheader("User Community Assignments")

        st.dataframe(
            communities.head(100),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("Community assignment data not available.")

    if not community_summary.empty:

        st.subheader("Community Summary")

        st.dataframe(
            community_summary,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = community_summary.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            fig = px.bar(
                community_summary,
                x=community_summary.columns[0],
                y=numeric_columns[-1],
                title="Community Statistics",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# --------------------------------------------------
# PAGE 5: FRIEND RECOMMENDATIONS
# --------------------------------------------------

elif page == "Friend Recommendations":

    st.header("🤝 Friend Recommendations")

    st.write("""
    The recommendation module identifies users who may
    be suitable connection suggestions based on
    their existing network relationships.
    """)

    if recommendations.empty:

        st.warning("Friend recommendation data not available.")

    else:

        st.dataframe(
            recommendations.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Total Recommendations",
            len(recommendations)
        )

# --------------------------------------------------
# PAGE 6: SHORTEST PATH
# --------------------------------------------------

elif page == "Shortest Path Analysis":

    st.header("📍 Shortest Path Analysis")

    st.write("""
    Shortest path analysis identifies the minimum number
    of network connections needed to travel between
    two users in the graph.
    """)

    if shortest_path.empty:

        st.warning("Shortest path data not available.")

    else:

        st.dataframe(
            shortest_path,
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# PAGE 7: LINK PREDICTION
# --------------------------------------------------

elif page == "Link Prediction":

    st.header("🔮 Machine Learning - Link Prediction")

    st.write("""
    The Random Forest model estimates the probability
    of potential new connections between users.
    """)

    if not prediction_metrics.empty:

        st.subheader("Model Performance")

        metric_values = {}

        for _, row in prediction_metrics.iterrows():

            metric_values[str(row.iloc[0])] = row.iloc[1]

        preferred_metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]

        available_metrics = [
            metric for metric in preferred_metrics
            if metric in metric_values
        ]

        columns = st.columns(
            max(1, len(available_metrics))
        )

        for i, metric in enumerate(available_metrics):

            try:
                value = float(metric_values[metric])

                columns[i].metric(
                    metric,
                    f"{value:.4f}"
                )

            except (ValueError, TypeError):
                columns[i].metric(
                    metric,
                    str(metric_values[metric])
                )

    st.markdown("---")

    st.subheader("Top Predicted Connections")

    if predicted_links.empty:

        st.warning(
            "Predicted links not found. Run link_prediction.py first."
        )

    else:

        st.dataframe(
            predicted_links,
            use_container_width=True,
            hide_index=True
        )

        if "Connection_Probability" in predicted_links.columns:

            fig = px.bar(
                predicted_links.head(20),
                x="User_1",
                y="Connection_Probability",
                color="Connection_Probability",
                title="Top Predicted Connection Probabilities",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# --------------------------------------------------
# PAGE 8: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("📈 Machine Learning Performance")

    if prediction_metrics.empty:

        st.warning("Model performance data not available.")

    else:

        st.dataframe(
            prediction_metrics,
            use_container_width=True,
            hide_index=True
        )

        if prediction_metrics.shape[1] >= 2:

            metric_chart = prediction_metrics.copy()

            metric_chart.columns = [
                "Metric",
                "Score"
            ] + list(metric_chart.columns[2:])

            metric_chart["Score"] = pd.to_numeric(
                metric_chart["Score"],
                errors="coerce"
            )

            fig = px.bar(
                metric_chart,
                x="Metric",
                y="Score",
                color="Metric",
                title="Random Forest Model Performance",
                range_y=[0, 1],
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# --------------------------------------------------
# PAGE 9: DOWNLOAD REPORTS
# --------------------------------------------------

elif page == "Download Reports":

    st.header("📥 Download Project Reports")

    for title, filename in FILES.items():

        path = os.path.join(RESULTS, filename)

        if os.path.exists(path):

            with open(path, "rb") as file:

                st.download_button(
                    label=f"Download {filename}",
                    data=file.read(),
                    file_name=filename,
                    mime="text/csv",
                    key=filename
                )

    graph_image = os.path.join(
        RESULTS,
        "network_graph.png"
    )

    if os.path.exists(graph_image):

        with open(graph_image, "rb") as file:

            st.download_button(
                label="Download Network Graph",
                data=file.read(),
                file_name="network_graph.png",
                mime="image/png"
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "<center>Social Network Intelligence and Graph Analytics "
    "System | Big Data Analytics Project</center>",
    unsafe_allow_html=True
)
