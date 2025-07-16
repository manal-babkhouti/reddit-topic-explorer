# src/cluster_hdbscan.py

import os
import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# === Configuration ===
INPUT_FILE = "data/reddit_clean.csv"
TEXT_COLUMN = "clean_text"
EMBEDDINGS_FILE = "data/reddit_hdbscan_embeddings.npy"
LABELS_FILE = "data/reddit_hdbscan_clustered.csv"
PLOT_FILE = "data/hdbscan_umap_plot.jpg"

MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 5
REDUCE_TO = 15


def run_hdbscan_pipeline() -> None:
    """Perform HDBSCAN clustering with UMAP on MiniLM embeddings."""
    print("🔁 Loading cleaned text data...")
    df = pd.read_csv(INPUT_FILE)
    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    print("🔤 Generating MiniLM embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"📉 Reducing to {REDUCE_TO} dimensions with UMAP...")
    umap_model = umap.UMAP(n_components=REDUCE_TO, random_state=42)
    X_umap = umap_model.fit_transform(embeddings)

    print("🧠 Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        prediction_data=True
    )
    labels = clusterer.fit_predict(X_umap)
    df["cluster"] = labels
    df.to_csv(LABELS_FILE, index=False, encoding="utf-8-sig")

    print("🖼️ Creating 2D UMAP cluster plot...")
    umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    df_plot = pd.DataFrame({
        "x": umap_2d[:, 0],
        "y": umap_2d[:, 1],
        "cluster": labels
    })

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot,
        x="x",
        y="y",
        hue="cluster",
        palette="tab10",
        legend="full",
        s=30
    )
    plt.title("HDBSCAN Clusters (UMAP 2D)")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()

    print("✅ Clustering complete. Files saved:")
    print(f"    - Embeddings: {EMBEDDINGS_FILE}")
    print(f"    - Clustered CSV: {LABELS_FILE}")
    print(f"    - Plot: {PLOT_FILE}")


if __name__ == "__main__":
    run_hdbscan_pipeline()
