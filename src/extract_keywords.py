import pandas as pd
from keybert import KeyBERT
import os


def extract_keywords_per_cluster(input_path="data/reddit_hdbscan_clustered.csv",
                                  output_path="data/hdbscan_cluster_keywords.csv",
                                  top_n=5,
                                  skip_noise=True,
                                  text_col="text",
                                  cluster_col="cluster"):
    """
    Extracts top keywords per cluster using KeyBERT from a CSV file.
    
    Parameters:
        input_path: Path to the clustered CSV (must contain `text_col` and `cluster_col`)
        output_path: Path to save the keyword summary CSV
        top_n: Number of top keywords to extract per cluster
        skip_noise: If True, ignores cluster -1 (HDBSCAN noise)
        text_col: Column containing the text
        cluster_col: Column containing the cluster IDs
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if cluster_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"❌ Input CSV must contain '{cluster_col}' and '{text_col}' columns.")

    # Initialize KeyBERT
    kw_model = KeyBERT()
    clusters = sorted(df[cluster_col].dropna().unique())

    summary = []
    print(f"🔎 Extracting keywords for {len(clusters)} clusters from: {input_path}")

    for cluster_id in clusters:
        if skip_noise and cluster_id == -1:
            continue

        subset = df[df[cluster_col] == cluster_id][text_col].dropna().tolist()
        if not subset:
            continue

        combined_text = " ".join(subset)
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n
        )
        top_terms = ", ".join([kw for kw, _ in keywords])

        summary.append({
            cluster_col: cluster_id,
            "post_count": len(subset),
            "top_keywords": top_terms
        })

        print(f"📌 Cluster {cluster_id} ({len(subset)} posts): {top_terms}")

    # Save results
    pd.DataFrame(summary).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Keyword summary saved to {output_path}")


if __name__ == "__main__":
    extract_keywords_per_cluster()
