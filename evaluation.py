import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import hdbscan
import umap
import matplotlib.pyplot as plt
import os

# Load products from Excel
xlsx_path = os.path.join(os.path.dirname(__file__), "data", "Category_Names.xlsx")
df = pd.read_excel(xlsx_path)
df.columns = ["text", "original_label"]
df = df[df["text"].str.upper() != "TOTAL MARKET"].reset_index(drop=True)

print(f"Products: {len(df)}")
print(f"Original labels: {df['original_label'].value_counts().to_dict()}\n")

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), normalize_embeddings=True)

# UMAP 2D for visualization
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
coords_2d = reducer.fit_transform(embeddings)

# --- K-Means sweep ---
print("=" * 50)
print("K-MEANS SWEEP (k=2..10)")
print("=" * 50)
best_k, best_sil = 2, -1
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels)
    ari = adjusted_rand_score(df["original_label"], labels)
    nmi = normalized_mutual_info_score(df["original_label"], labels)
    print(f"  k={k:2d}  silhouette={sil:.3f}  ARI={ari:.3f}  NMI={nmi:.3f}")
    if sil > best_sil:
        best_sil, best_k = sil, k

print(f"\n  Best k={best_k} (silhouette={best_sil:.3f})\n")

# --- HDBSCAN ---
print("=" * 50)
print("HDBSCAN")
print("=" * 50)
for mcs in [2, 3, 4, 5]:
    for ms in [1, 2, 3]:
        hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean')
        labels = hdb.fit_predict(embeddings)
        n_clusters = len(set(labels) - {-1})
        noise = (labels == -1).sum()
        valid = labels != -1
        if valid.sum() > 1 and n_clusters > 1:
            sil = silhouette_score(embeddings[valid], labels[valid])
            ari = adjusted_rand_score(df["original_label"][valid], labels[valid])
            nmi = normalized_mutual_info_score(df["original_label"][valid], labels[valid])
        else:
            sil = ari = nmi = 0.0
        print(f"  min_cluster={mcs} min_samples={ms}  clusters={n_clusters}  noise={noise}  sil={sil:.3f}  ARI={ari:.3f}  NMI={nmi:.3f}")

# --- Best K-Means detailed output ---
print(f"\n{'=' * 50}")
print(f"DETAILED RESULTS: K-Means k={best_k}")
print(f"{'=' * 50}")
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km.fit_predict(embeddings)
df["cluster"] = labels

for c in range(best_k):
    members = df[df["cluster"] == c]
    print(f"\nCluster {c} ({len(members)} products):")
    for _, row in members.iterrows():
        print(f"  {row['text']:35s} [{row['original_label']}]")

# Cross-tab
print(f"\nCluster vs Original Labels:")
print(pd.crosstab(df["cluster"], df["original_label"]))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Clusters
for c in range(best_k):
    mask = labels == c
    axes[0].scatter(coords_2d[mask, 0], coords_2d[mask, 1], label=f"Cluster {c}", s=40, alpha=0.8)
axes[0].set_title(f"K-Means (k={best_k})")
axes[0].legend(fontsize=8)

# Original labels
for lbl in df["original_label"].unique():
    mask = df["original_label"] == lbl
    axes[1].scatter(coords_2d[mask, 0], coords_2d[mask, 1], label=lbl, s=40, alpha=0.8)
axes[1].set_title("Original Labels")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("clustering_evaluation.png", dpi=120)
print("\nSaved clustering_evaluation.png")
plt.show()