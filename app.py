import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import umap
import plotly.express as px
import os

st.set_page_config(page_title="Product Category Discovery", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load product names from Excel
def load_products():
    xlsx_path = os.path.join(os.path.dirname(__file__), "data", "Category_Names.xlsx")
    if os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
        df.columns = ["text", "original_label"]
        df = df[df["text"].str.upper() != "TOTAL MARKET"].reset_index(drop=True)
        return df
    return pd.DataFrame(columns=["text", "original_label"])

@st.cache_data
def compute_embeddings(_model, texts):
    return _model.encode(texts, normalize_embeddings=True)

@st.cache_data
def reduce_umap(embeddings, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)

# --- Sidebar config ---
st.sidebar.header("Clustering Settings")

method = st.sidebar.selectbox("Algorithm", ["HDBSCAN", "K-Means"])

if method == "K-Means":
    n_clusters = st.sidebar.slider("Number of clusters", 2, 15, 5)
else:
    min_cluster_size = st.sidebar.slider("Min cluster size", 2, 15, 3)
    min_samples = st.sidebar.slider("Min samples", 1, 10, 2)

umap_neighbors = st.sidebar.slider("UMAP neighbors", 5, 50, 15)
umap_min_dist = st.sidebar.slider("UMAP min distance", 0.0, 1.0, 0.1, step=0.05)

# Allow users to add extra product names
st.sidebar.markdown("---")
st.sidebar.subheader("Add Products")
extra_products = st.sidebar.text_area(
    "One product per line",
    placeholder="Laptop\nHeadphones\nPhone Case",
)

# --- Load and prepare data ---
base_df = load_products()

# Merge extra products
extra_list = [p.strip() for p in extra_products.strip().split("\n") if p.strip()] if extra_products else []
if extra_list:
    extra_df = pd.DataFrame({"text": extra_list, "original_label": "User-added"})
    df = pd.concat([base_df, extra_df], ignore_index=True)
else:
    df = base_df.copy()

if len(df) < 3:
    st.error("Need at least 3 products to cluster.")
    st.stop()

texts = df["text"].tolist()
embeddings = compute_embeddings(model, texts)

# --- Clustering ---
if method == "K-Means":
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(embeddings)
else:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(embeddings)

df["cluster"] = labels
n_found = len(set(labels) - {-1})

# Auto-name clusters using the most central product in each cluster
cluster_names = {}
for c in sorted(set(labels)):
    if c == -1:
        cluster_names[c] = "Noise / Outlier"
        continue
    mask = labels == c
    cluster_embs = embeddings[mask]
    centroid = cluster_embs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    # Find the product closest to the centroid
    dists = np.dot(cluster_embs, centroid)
    best_idx = np.argmax(dists)
    representative = df[mask].iloc[best_idx]["text"]
    members = df[mask]["text"].tolist()
    # Use top-3 products as cluster name
    top3_idx = np.argsort(dists)[-3:][::-1]
    top3 = [df[mask].iloc[i]["text"] for i in top3_idx]
    cluster_names[c] = f"Cluster {c}: {', '.join(top3)}"

df["cluster_name"] = df["cluster"].map(cluster_names)

# --- UI ---
st.title("🔍 Product Category Discovery")
st.caption("Unsupervised clustering — no predefined labels needed")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Products", len(df))
col2.metric("Clusters found", n_found)
noise_count = (labels == -1).sum()
if method == "HDBSCAN" and noise_count > 0:
    col3.metric("Noise points", noise_count)
else:
    valid = labels[labels != -1]
    if len(set(valid)) > 1:
        sil = silhouette_score(embeddings[labels != -1], valid)
        col3.metric("Silhouette score", f"{sil:.3f}")

# 2D scatter plot
st.subheader("Cluster Map")
coords_2d = reduce_umap(embeddings, n_neighbors=umap_neighbors, min_dist=umap_min_dist)
plot_df = df.copy()
plot_df["x"] = coords_2d[:, 0]
plot_df["y"] = coords_2d[:, 1]
plot_df["cluster_str"] = plot_df["cluster"].astype(str)

fig = px.scatter(
    plot_df, x="x", y="y", color="cluster_str", hover_data=["text", "original_label"],
    title="Product Embeddings (UMAP 2D)",
    labels={"cluster_str": "Cluster", "x": "", "y": ""},
    height=550,
)
fig.update_traces(marker=dict(size=8, opacity=0.8))
fig.update_layout(showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# Cluster breakdown
st.subheader("Cluster Contents")
for c in sorted(set(labels)):
    name = cluster_names[c]
    members = df[df["cluster"] == c][["text", "original_label"]].reset_index(drop=True)
    with st.expander(f"{name} ({len(members)} products)"):
        st.dataframe(members, use_container_width=True)

# Compare with original labels
if "original_label" in df.columns and df["original_label"].nunique() > 1:
    st.subheader("Cluster vs Original Labels")
    cross = pd.crosstab(df["cluster_name"], df["original_label"])
    st.dataframe(cross, use_container_width=True)

# Predict new product
st.markdown("---")
st.subheader("Assign a New Product")
new_product = st.text_input("Enter product name:")
if st.button("Find Cluster"):
    if new_product.strip():
        emb = model.encode([new_product.strip()], normalize_embeddings=True)[0]
        if method == "K-Means":
            pred = clusterer.predict(emb.reshape(1, -1))[0]
            st.success(f"**{new_product}** → **{cluster_names[pred]}**")
        else:
            # For HDBSCAN: assign to nearest cluster centroid
            best_cluster = -1
            best_sim = -1
            for c in sorted(set(labels)):
                if c == -1:
                    continue
                mask = labels == c
                centroid = embeddings[mask].mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                sim = float(np.dot(emb, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c
            if best_sim < 0.35:
                st.warning(f"**{new_product}** → **{cluster_names.get(best_cluster, 'Unknown')}** ⚠️ Low similarity ({best_sim:.2f}) — may not fit any cluster")
            else:
                st.success(f"**{new_product}** → **{cluster_names.get(best_cluster, 'Unknown')}** (similarity: {best_sim:.2f})")
    else:
        st.warning("Enter something")
        with st.expander(f"Scores for {t}"):
            for cat, score in scores:
                st.write(f"- {cat}: {score}%")