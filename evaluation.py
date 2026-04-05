import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from segmentation_core import (
    DEFAULT_OTHERS_THRESHOLD,
    build_category_centroids,
    classify_products,
    compute_umap_2d,
    encode_texts,
    load_model,
    load_products,
)

LABEL_MAP = {
    "food": "Food",
    "drinks": "Drinks",
    "home care": "Home Care",
    "personal care": "Personal Care",
    "other": "Others",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot product category classification.")
    parser.add_argument("--data-path", default=os.path.join("data", "Category_Names.xlsx"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_OTHERS_THRESHOLD)
    parser.add_argument("--sweep-thresholds", action="store_true", help="Sweep similarity thresholds.")
    parser.add_argument("--plot-path", default="classification_evaluation.png")
    parser.add_argument("--show", action="store_true", help="Display the matplotlib figure after saving.")
    return parser.parse_args()


def normalize_label(label: str) -> str:
    return LABEL_MAP.get(label.strip().lower(), label.strip())


def plot_results(df: pd.DataFrame, embeddings: np.ndarray, plot_path: str):
    coords = compute_umap_2d(embeddings, n_neighbors=15, min_dist=0.1, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for cat in sorted(df["predicted_category"].unique()):
        mask = df["predicted_category"] == cat
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=45, alpha=0.82, label=cat)
    axes[0].set_title("Predicted Categories (Zero-Shot)")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].legend(fontsize=7, loc="best")

    for label in sorted(df["original_label"].unique()):
        mask = df["original_label"] == label
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=45, alpha=0.82, label=label)
    axes[1].set_title("Original Labels (Reference)")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].legend(fontsize=8, loc="best")

    fig.savefig(plot_path, dpi=140)
    return fig


def main():
    args = parse_args()

    df = load_products(args.data_path)
    if len(df) < 1:
        raise SystemExit("No products found.")

    print(f"Products: {len(df)}")
    print(f"Original labels: {df['original_label'].value_counts().to_dict()}")
    print()

    model = load_model()
    embeddings = encode_texts(model, df["text"].tolist())
    centroids = build_category_centroids(model)

    df["true_label"] = df["original_label"].apply(normalize_label)

    if args.sweep_thresholds:
        print("=" * 72)
        print("THRESHOLD SWEEP")
        print("=" * 72)
        print(f"{'Threshold':<12} {'Accuracy':<10} {'Others count':<15}")
        for t in np.arange(0.15, 0.51, 0.05):
            pred_labels, _ = classify_products(embeddings, centroids, threshold=t)
            acc = accuracy_score(df["true_label"], pred_labels)
            n_others = pred_labels.count("Others")
            print(f"{t:<12.2f} {acc:<10.3f} {n_others:<15d}")
        print()

    pred_labels, similarity_matrix = classify_products(embeddings, centroids, threshold=args.threshold)
    df["predicted_category"] = pred_labels

    true_labels = df["true_label"].tolist()
    all_labels = sorted(set(true_labels + pred_labels))
    acc = accuracy_score(true_labels, pred_labels)

    print("=" * 72)
    print(f"EVALUATION (threshold={args.threshold})")
    print("=" * 72)
    print(f"Accuracy: {acc:.3f}")
    print()

    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, labels=all_labels, zero_division=0))
    print()

    print("Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    print(cm_df)
    print()

    cat_names = list(centroids.keys())
    print("Per-category avg similarity:")
    for i, cat in enumerate(cat_names):
        mask = df["predicted_category"] == cat
        if mask.sum() > 0:
            avg_sim = float(similarity_matrix[mask, i].mean())
            min_sim = float(similarity_matrix[mask, i].min())
            print(f"  {cat}: avg={avg_sim:.3f}, min={min_sim:.3f}, count={mask.sum()}")
    print()

    misclassified = df[df["true_label"] != df["predicted_category"]]
    if len(misclassified) > 0:
        print(f"Misclassified products ({len(misclassified)}):")
        for _, row in misclassified.iterrows():
            print(f"  {row['text']}: {row['true_label']} -> {row['predicted_category']}")
    else:
        print("All products correctly classified!")
    print()

    fig = plot_results(df, embeddings, args.plot_path)
    print(f"Saved plot: {args.plot_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
