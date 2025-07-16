# src/add_pca_coordinates.py

import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Apply PCA on numeric columns and add principal coordinates to the DataFrame.

    Args:
        df: Input DataFrame
        n_components: Number of principal components to keep

    Returns:
        DataFrame with 'pca_1', 'pca_2' columns added
    """
    numeric_cols = df.select_dtypes(include="number").dropna(axis=1)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(numeric_cols)

    df["pca_1"] = coords[:, 0]
    df["pca_2"] = coords[:, 1]
    return df


if __name__ == "__main__":
    # === Load clustered datasets ===
    paths = {
        "k4": "data/reddit_kmeans_k4_clusters.csv",
        "k5": "data/reddit_kmeans_k5_clusters.csv",
        "hdb": "data/reddit_hdbscan_clustered.csv"
    }

    for key, path in paths.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = apply_pca(df)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ PCA coordinates added and saved to {path}")
