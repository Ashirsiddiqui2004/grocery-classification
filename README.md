# Product Category Segmentation (Unsupervised)

Unsupervised product segmentation app:
- User enters a product name
- Model assigns it to a discovered category (or `Outlier / Unknown`)
- Result is shown with `match score`, `direct similarity`, and confidence

No supervised training labels are required for prediction.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

## Streamlit Usage (Exact Steps)

1. Open the URL printed by Streamlit (usually `http://localhost:8501`).
2. In the main page, go to **Classify a New Product**.
3. Type product name in **Enter product name**.
4. Click **Classify product**.
5. Read output:
   - Success = assigned discovered category
   - Warning = `Outlier / Unknown`
6. Check the table below for top candidate categories and **Match score** values.

## Where To Add Products

- Temporary (current run only):
  - Sidebar -> **Add Products** -> **One product per line**
  - Not written back to file

- Permanent (all future runs):
  - Add rows to `data/Category_Names.xlsx` (first column = product name)
  - Restart/rerun app

## Current Pipeline

1. Encode product names with Sentence Transformers (`all-MiniLM-L6-v2`).
2. Cluster embeddings using `Auto`, `K-Means`, or `HDBSCAN`.
3. Build category profiles with:
   - cluster members
   - centroid
   - keywords
   - cohesion score
4. Classify a query with a hybrid score (stronger than centroid-only):
   - nearest member similarity
   - keyword semantic similarity
   - keyword token overlap
5. Apply outlier gate using calibrated direct-similarity threshold (leave-one-out based).

## How To Read Scores

- `Match score`: weighted category score from the hybrid matcher.
- `Direct similarity`: best similarity to an actual known member inside that category.
- `Outlier threshold`: slider in sidebar; default is auto-calibrated from your data.

## Why Some Inputs Can Still Be Weak

If your dataset is mostly food/home-care/personal-care, broad terms like electronics may be out-of-domain and should become `Outlier / Unknown`.

For short ambiguous terms (for example `apple`), results improve when:
- you add more examples for that concept in the dataset
- input is more specific (`fresh apple fruit`, `apple juice`, etc.)

## Run Evaluation

Default (includes sweeps + final run):

```bash
python evaluation.py
```

Fast run:

```bash
python evaluation.py --skip-sweeps
```

Example options:

```bash
python evaluation.py --algorithm Auto --k-min 2 --k-max 12 --plot-path clustering_evaluation.png
```

Outputs:
- clustering metrics (`silhouette`, `ARI`, `NMI`, `coverage`, `noise`)
- discovered category breakdown
- cross-tab vs original labels
- saved plot `clustering_evaluation.png`

## Project Files

```text
app.py                 # Streamlit UI
evaluation.py          # CLI evaluator
segmentation_core.py   # Shared clustering + classification logic
data/Category_Names.xlsx
requirements.txt
```
