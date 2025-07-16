# src/cluster_kmeans.py

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def elbow_method(X: np.ndarray, k_range=range(2, 16)) -> None:
    """
    Plot and save the Elbow Curve to help choose the optimal number of clusters (k).

    Args:
        X: Embeddings or feature matrix
        k_range: Range of k values to test
    """
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), wcss, marker='o')
    plt.title("Elbow Curve (KMeans on MiniLM Embeddings)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Inertia)")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("data/elbow.jpg")
    print("📸 Saved Elbow curve to data/elbow.jpg")
    plt.close()


def visualize_clusters(X: np.ndarray, labels: np.ndarray, k: int) -> None:
    """
    Reduce to 2D with PCA and save scatterplot of KMeans clusters.

    Args:
        X: Embeddings or features
        labels: Cluster labels
        k: Number of clusters
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10)
    plt.title(f"KMeans Clustering (k={k})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()

    plot_path = f"data/kmeans_k{k}.jpg"
    plt.savefig(plot_path)
    print(f"📸 Saved cluster plot to {plot_path}")
    plt.close()


def run_kmeans_clustering(
    X: np.ndarray,
    df_texts: pd.DataFrame,
    k_values: list[int] = [4, 5]
) -> None:
    """
    Run KMeans for multiple values of k, evaluate, and save results.

    Args:
        X: Embeddings or features
        df_texts: Original post data
        k_values: List of k to try
    """
    for k in k_values:
        print(f"\n🔹 KMeans with k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)
        print(f"📊 Silhouette Score: {score:.3f}")

        col_name = f"kmeans_k{k}"
        df_labeled = df_texts.copy()
        df_labeled[col_name] = labels

        output_file = f"data/reddit_kmeans_k{k}_clusters.csv"
        df_labeled.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"💾 Saved clustered data to {output_file}")

        visualize_clusters(X, labels, k)


if __name__ == "__main__":
    print("🔁 Loading embeddings and post data...")

    X = np.load("data/reddit_embeddings_clean.npy")
    df = pd.read_csv("data/reddit_clean.csv", encoding="utf-8-sig")

    elbow_method(X)
    run_kmeans_clustering(X, df, k_values=[4, 5])
