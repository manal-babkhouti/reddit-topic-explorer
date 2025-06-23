# encode_text.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load cleaned data
df = pd.read_csv("data/reddit_clean.csv")

# Use the 'clean_text' column for embedding
texts = df["clean_text"].fillna("").tolist()

# Load MiniLM model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Save embeddings with a descriptive filename
np.save("data/reddit_embeddings_clean.npy", embeddings)

print("✅ Embeddings generated and saved to data/reddit_embeddings_clean.npy")
