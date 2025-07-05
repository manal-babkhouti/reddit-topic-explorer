
import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import os

# --- Config ---
INPUT_FILE = "data/reddit_clean.csv"
TEXT_COLUMN = "clean_text"  # Use clean_text for higher quality
EMBEDDINGS_FILE = "data/reddit_hdbscan_embeddings.npy"
LABELS_FILE = "data/reddit_hdbscan_clustered.csv"
PLOT_FILE = "data/hdbscan_umap_plot.jpg"
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 5
REDUCE_TO = 15

# --- Load and Embed ---
print("🔁 Loading text data...")
df = pd.read_csv(INPUT_FILE)
texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

print("🔤 Generating MiniLM embeddings...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)
np.save(EMBEDDINGS_FILE, embeddings)

# --- Dimensionality Reduction ---
print(f"📉 Reducing to {REDUCE_TO} dimensions with UMAP...")
umap_model = umap.UMAP(n_components=REDUCE_TO, random_state=42)
X_umap = umap_model.fit_transform(embeddings)

# --- Clustering ---
print("🧠 Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    prediction_data=True
)
labels = clusterer.fit_predict(X_umap)
df["cluster"] = labels
df.to_csv(LABELS_FILE, index=False, encoding="utf-8-sig")

# --- Visualization ---
print("🖼️ Saving UMAP 2D cluster plot...")
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
df_plot = pd.DataFrame({
    "x": umap_2d[:, 0],
    "y": umap_2d[:, 1],
    "cluster": labels
})
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot, x="x", y="y", hue="cluster", palette="tab10", legend="full", s=30)
plt.title("HDBSCAN Clusters (UMAP 2D)")
plt.legend(loc="best", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
print("✅ Done. Embeddings, labels, and plot saved.")
