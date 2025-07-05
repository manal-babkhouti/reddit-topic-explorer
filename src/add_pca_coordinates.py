import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(df, n_components=2):
    numeric_cols = df.select_dtypes(include="number").dropna(axis=1)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(numeric_cols)
    df["pca_1"] = coords[:, 0]
    df["pca_2"] = coords[:, 1]
    return df

# Load the files
df_k4 = pd.read_csv("data/reddit_kmeans_k4_clusters.csv", encoding="utf-8-sig")
df_k5 = pd.read_csv("data/reddit_kmeans_k5_clusters.csv", encoding="utf-8-sig")
df_hdb = pd.read_csv("data/reddit_hdbscan_clustered.csv", encoding="utf-8-sig")

# Apply PCA and save
df_k4 = apply_pca(df_k4)
df_k5 = apply_pca(df_k5)
df_hdb = apply_pca(df_hdb)

df_k4.to_csv("data/reddit_kmeans_k4_clusters.csv", index=False, encoding="utf-8-sig")
df_k5.to_csv("data/reddit_kmeans_k5_clusters.csv", index=False, encoding="utf-8-sig")
df_hdb.to_csv("data/reddit_hdbscan_clustered.csv", index=False, encoding="utf-8-sig")
