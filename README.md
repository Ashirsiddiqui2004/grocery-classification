# Product Category Discovery

Unsupervised clustering app that automatically discovers product categories from short product names using sentence embeddings.

---

## 🔍 Problem

Given short product/category names (e.g., "NB Fresh Fruits", "Laundry Bleach"), automatically discover meaningful product groupings — without predefined labels.

---

## ⚙️ Approach

- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`) encode product names into 384-dim vectors
- **Clustering**: HDBSCAN (density-based) or K-Means — configurable from the sidebar
- **Visualization**: UMAP projects embeddings to 2D; Plotly renders an interactive scatter plot
- **Cluster naming**: Each cluster is auto-labeled by its 3 most central products

---

## 📊 Evaluation

Evaluated against the original hand-labeled categories using:

| Metric | HDBSCAN (best) | K-Means (k=5) |
|--------|----------------|----------------|
| NMI | 0.750 | 0.588 |
| ARI | 0.530 | 0.412 |
| Silhouette | 0.117 | 0.054 |

Run the evaluation script:

```bash
python evaluation.py
```

Outputs a K-Means sweep (k=2..10), HDBSCAN parameter grid, detailed cluster contents, and a side-by-side UMAP plot saved to `clustering_evaluation.png`.

---

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 📁 Structure

```
app.py              # Streamlit app with clustering UI
evaluation.py       # Clustering evaluation script
data/
  Category_Names.xlsx  # Product names with original labels
requirements.txt
```

---

## 🛠️ Features

- **Sidebar controls** — switch between HDBSCAN / K-Means, tune cluster size and UMAP params
- **Add products on the fly** — paste new product names in the sidebar
- **Assign new products** — type any product name to find its nearest cluster
- **Cluster vs Original Labels** — cross-tab comparison table
- **Interactive UMAP plot** — hover to inspect individual products