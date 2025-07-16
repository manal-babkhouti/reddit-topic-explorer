# src/extract_keywords.py

import pandas as pd
from keybert import KeyBERT
import os


def extract_keywords_per_cluster(
    input_path: str = "data/reddit_hdbscan_clustered.csv",
    output_path: str = "data/hdbscan_cluster_keywords.csv",
    top_n: int = 5,
    skip_noise: bool = True,
    text_col: str = "text",
    cluster_col: str = "cluster"
) -> None:
    """
    Extract top keywords per cluster using KeyBERT.

    Args:
        input_path: CSV file with clustered text data
        output_path: Where to save the keyword summary
        top_n: Number of top keywords per cluster
        skip_noise: Whether to ignore cluster -1 (noise)
        text_col: Column name for text
        cluster_col: Column name for cluster labels
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if cluster_col not in df or text_col not in df:
        raise ValueError(f"❌ Columns '{text_col}' or '{cluster_col}' not found in CSV.")

    kw_model = KeyBERT()
    clusters = sorted(df[cluster_col].dropna().unique())

    summary = []
    print(f"🔎 Extracting keywords for {len(clusters)} clusters from: {input_path}")

    for cluster_id in clusters:
        if skip_noise and cluster_id == -1:
            continue

        cluster_texts = df[df[cluster_col] == cluster_id][text_col].dropna().tolist()
        if not cluster_texts:
            continue

        combined_text = " ".join(cluster_texts)
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n
        )
        top_terms = ", ".join([kw for kw, _ in keywords])

        summary.append({
            cluster_col: cluster_id,
            "post_count": len(cluster_texts),
            "top_keywords": top_terms
        })

        print(f"📌 Cluster {cluster_id} ({len(cluster_texts)} posts): {top_terms}")

    pd.DataFrame(summary).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Keyword summary saved to: {output_path}")


if __name__ == "__main__":
    extract_keywords_per_cluster()
