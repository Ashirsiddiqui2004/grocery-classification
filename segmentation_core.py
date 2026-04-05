import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Category_Names.xlsx")

# Anchor phrases per category, used to build zero-shot centroids
CATEGORY_ANCHORS = {
    "Food": [
        "food products and groceries",
        "dairy food products like milk cheese butter yogurt cream",
        "UHT milk powder milk evaporated milk condensed milk",
        "sour cream and dairy cream products",
        "meat and poultry such as chicken turkey ham beef",
        "bread bakery and dough products",
        "snacks biscuits chips crackers",
        "canned food like tuna chicken pate",
        "frozen food and frozen vegetables",
        "cooking oil and olive oil",
        "sugar as food ingredient and sweetener",
        "cereals granola oats breakfast food",
        "condiments sauces salad dressing",
        "chocolate candy sweets confectionery easter eggs",
        "ice cream and frozen dessert",
        "fresh fruits and vegetables",
        "pasta noodles macaroni",
        "dried meat cold meat hamburger",
        "pet food and pet snacks",
        "drinkable yoghurt and yogurt beverages as dairy food",
    ],
    "Drinks": [
        "beverages and drinks",
        "coffee ground coffee bean coffee",
        "tea herbal tea mate cocido yerba mate",
        "ready to drink tea and ready to drink coffee",
        "juice fruit juice vegetable juice",
        "soft drinks soda cola lemonade",
        "sports drinks and energy drinks",
        "hot chocolate and cocoa drink",
        "distilled spirits and alcoholic beverages",
        "water and mineral water",
    ],
    "Home Care": [
        "home care and household cleaning products",
        "laundry detergent and laundry bar",
        "bleach and laundry bleach",
        "cleaners disinfectants and sanitizers",
        "floor cleaner and surface cleaner",
        "drain cleaner and declogger",
        "cleaning gloves sponge and scrubber",
        "insect repellent and insecticide",
        "scented candles and air freshener",
        "plastic storage bags and trash bags",
        "batteries and household supplies",
        "laundry additives and fabric softener",
    ],
    "Personal Care": [
        "personal care and beauty products",
        "hair shampoo and hair conditioner",
        "hair styling gel and mousse",
        "hair treatment and straightening cream",
        "skin care face care and moisturizer",
        "nail treatment and nail care",
        "body wash soap and deodorant",
        "adult incontinence and hygiene products",
        "cosmetics and makeup",
        "toothpaste and oral care",
    ],
}

TARGET_CATEGORIES = list(CATEGORY_ANCHORS.keys()) + ["Others"]
DEFAULT_OTHERS_THRESHOLD = 0.3


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def load_products(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        return pd.DataFrame(columns=["text", "original_label"])

    df = pd.read_excel(data_path)
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["text", "original_label"])

    df = df.iloc[:, :2].copy()
    df.columns = ["text", "original_label"]
    df["text"] = df["text"].astype(str).str.strip()
    df["original_label"] = df["original_label"].astype(str).str.strip()
    df = df[df["text"].str.upper() != "TOTAL MARKET"]
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True).astype(np.float32)


def compute_umap_2d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def build_category_centroids(model: SentenceTransformer) -> dict[str, np.ndarray]:
    centroids = {}
    for category, anchors in CATEGORY_ANCHORS.items():
        anchor_embeddings = encode_texts(model, anchors)
        centroid = anchor_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[category] = centroid
    return centroids


def classify_products(
    embeddings: np.ndarray,
    centroids: dict[str, np.ndarray],
    threshold: float = DEFAULT_OTHERS_THRESHOLD,
) -> tuple[list[str], np.ndarray]:
    category_names = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in category_names], axis=0)
    similarity_matrix = np.dot(embeddings, centroid_matrix.T)

    best_indices = np.argmax(similarity_matrix, axis=1)
    best_scores = similarity_matrix[np.arange(len(embeddings)), best_indices]
    labels = [
        category_names[i] if score >= threshold else "Others"
        for i, score in zip(best_indices, best_scores)
    ]
    return labels, similarity_matrix


def classify_single_product(
    product_name: str,
    model: SentenceTransformer,
    centroids: dict[str, np.ndarray],
    threshold: float = DEFAULT_OTHERS_THRESHOLD,
) -> dict:
    emb = model.encode([product_name], normalize_embeddings=True)[0]

    scores = {}
    for category, centroid in centroids.items():
        scores[category] = float(np.dot(emb, centroid))

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    is_outlier = best_score < threshold
    assigned = "Others" if is_outlier else best_category

    if not is_outlier and margin >= 0.08:
        confidence = "High"
    elif not is_outlier and margin >= 0.03:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "assigned_category": assigned,
        "best_category": best_category,
        "best_score": best_score,
        "margin": margin,
        "confidence": confidence,
        "is_outlier": is_outlier,
        "all_scores": sorted_scores,
    }
