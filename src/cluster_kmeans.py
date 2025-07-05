import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

def elbow_method(X, k_range=range(2, 16)):
    """Plot and save the Elbow Curve to determine optimal k."""
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), wcss, marker='o')
    plt.title("Elbow Curve (KMeans on MiniLM embeddings)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS (inertia)")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig("data/elbow.jpg")
    print("📸 Saved Elbow curve to data/elbow.jpg")
    plt.close()

def visualize_clusters(X, labels, k):
    """Use PCA to reduce X to 2D and save scatter plot of clusters."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(f"KMeans Clustering (k={k})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()

    plot_path = f"data/kmeans_k{k}.jpg"
    plt.savefig(plot_path)
    print(f"📸 Saved cluster plot to {plot_path}")
    plt.close()

def run_kmeans_clustering(X, df_texts, k_values=[4, 5]):
    """Run KMeans for multiple k values, visualize, and save results."""
    for k in k_values:
        print(f"\n🔹 KMeans with k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)

        # Evaluate clustering quality
        score = silhouette_score(X, labels)
        print(f"📊 Silhouette Score: {score:.3f}")

        # Create labeled DataFrame
        col_name = f"kmeans_k{k}"
        df_labeled = df_texts.copy()
        df_labeled[col_name] = labels

        # Save labeled dataset
        output_file = f"data/reddit_kmeans_k{k}_clusters.csv"
        df_labeled.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"💾 Saved to {output_file}")

        # Save visual
        visualize_clusters(X, labels, k)

if __name__ == "__main__":
    print("🔁 Loading data...")

    # Load embeddings + cleaned posts
    X = np.load("data/reddit_embeddings_clean.npy")
    df = pd.read_csv("data/reddit_clean.csv", encoding="utf-8-sig")

    # Step 1: Elbow plot
    elbow_method(X)

    # Step 2: Run KMeans + plot clusters
    run_kmeans_clustering(X, df, k_values=[4, 5])
