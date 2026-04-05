import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from segmentation_core import (
    DEFAULT_OTHERS_THRESHOLD,
    TARGET_CATEGORIES,
    build_category_centroids,
    classify_products,
    classify_single_product,
    compute_umap_2d,
    encode_texts,
    load_model,
    load_products,
)

st.set_page_config(page_title="Product Category Segmentation", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Category_Names.xlsx")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_products(path: str) -> pd.DataFrame:
    return load_products(path)


@st.cache_data
def get_embeddings(texts: tuple[str, ...]) -> np.ndarray:
    model = get_model()
    return encode_texts(model, list(texts))


@st.cache_resource
def get_centroids():
    model = get_model()
    return build_category_centroids(model)


@st.cache_data
def reduce_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    return compute_umap_2d(embeddings, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)


st.title("Product Category Segmentation")
st.caption("Zero-shot classification into Food, Drinks, Home Care, Personal Care, and Others.")

st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Others threshold",
    min_value=0.15,
    max_value=0.50,
    value=float(DEFAULT_OTHERS_THRESHOLD),
    step=0.01,
)
st.sidebar.caption("Products scoring below this against all categories are classified as 'Others'.")

st.sidebar.markdown("---")
st.sidebar.subheader("Visualization")
umap_neighbors = st.sidebar.slider("UMAP neighbors", 5, 50, 15)
umap_min_dist = st.sidebar.slider("UMAP min distance", 0.0, 1.0, 0.1, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Add Products")
extra_products = st.sidebar.text_area(
    "One product per line",
    placeholder="Laptop\nHeadphones\nPhone Case",
)

base_df = get_products(DATA_PATH)
extra_items = [line.strip() for line in extra_products.splitlines() if line.strip()]
if extra_items:
    extra_df = pd.DataFrame({"text": extra_items, "original_label": "User-added"})
    df = pd.concat([base_df, extra_df], ignore_index=True)
else:
    df = base_df.copy()
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

if len(df) < 1:
    st.error("No products loaded.")
    st.stop()

model = get_model()
embeddings = get_embeddings(tuple(df["text"].tolist()))
centroids = get_centroids()

labels, _ = classify_products(embeddings, centroids, threshold=threshold)
df["predicted_category"] = labels

category_counts = df["predicted_category"].value_counts()
others_count = int(category_counts.get("Others", 0))

col1, col2, col3 = st.columns(3)
col1.metric("Products", len(df))
col2.metric("Categories used", int((category_counts > 0).sum()))
col3.metric("Others / Unknown", others_count)

st.subheader("Classification Results")
category_rows = []
for cat in TARGET_CATEGORIES:
    members = df[df["predicted_category"] == cat]
    if len(members) > 0:
        examples = members["text"].head(5).tolist()
        category_rows.append({
            "Category": cat,
            "Count": len(members),
            "Examples": ", ".join(examples),
        })
if category_rows:
    st.dataframe(pd.DataFrame(category_rows), use_container_width=True, hide_index=True)

st.subheader("Category Map")
coords_2d = reduce_umap(embeddings, n_neighbors=umap_neighbors, min_dist=umap_min_dist)
plot_df = df.copy()
plot_df["x"] = coords_2d[:, 0]
plot_df["y"] = coords_2d[:, 1]
fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    color="predicted_category",
    hover_data=["text", "original_label"],
    height=560,
    labels={"x": "", "y": "", "predicted_category": "Category"},
    title="Product Category Segmentation",
)
fig.update_traces(marker={"size": 8, "opacity": 0.82})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Classify a New Product")
new_product = st.text_input("Enter product name")
if st.button("Classify", use_container_width=True):
    candidate = new_product.strip()
    if not candidate:
        st.warning("Please enter a product name.")
    else:
        result = classify_single_product(candidate, model, centroids, threshold=threshold)
        if result["is_outlier"]:
            st.warning(
                f"'{candidate}' -> Others "
                f"(best match: {result['best_category']} at {result['best_score']:.3f}, below threshold)"
            )
        else:
            st.success(
                f"'{candidate}' -> **{result['assigned_category']}** "
                f"(score: {result['best_score']:.3f}, confidence: {result['confidence']})"
            )

        score_rows = [
            {"Category": cat, "Similarity": round(score, 3)}
            for cat, score in result["all_scores"]
        ]
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

with st.expander("Inspect category members"):
    for cat in TARGET_CATEGORIES:
        members = df[df["predicted_category"] == cat][["text", "original_label"]].reset_index(drop=True)
        if len(members) > 0:
            st.markdown(f"**{cat}** ({len(members)} products)")
            st.dataframe(members, use_container_width=True, hide_index=True)
