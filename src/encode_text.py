# src/encode_text.py

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def encode_text_column(
    input_path: str = "data/reddit_clean.csv",
    output_path: str = "data/reddit_embeddings_clean.npy",
    text_col: str = "clean_text"
) -> None:
    """
    Encode text using a SentenceTransformer (MiniLM) and save embeddings.

    Args:
        input_path: Path to the cleaned dataset
        output_path: File to save the generated embeddings
        text_col: Name of the column to encode
    """
    print(f"🔁 Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    texts = df[text_col].fillna("").tolist()

    print("🔤 Generating embeddings with MiniLM...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    np.save(output_path, embeddings)
    print(f"✅ Embeddings saved to: {output_path}")


if __name__ == "__main__":
    encode_text_column()
