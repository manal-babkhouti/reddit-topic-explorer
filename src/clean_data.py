# src/clean_data.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords (only first time)
nltk.download("stopwords")

# === Custom stopword set (NLTK + Reddit filler terms) ===
CUSTOM_STOPWORDS = set(stopwords.words("english"))
FILLERS = {"im", "like", "would", "get", "one", "also", "really",
           "see", "even", "cant", "dont", "didnt", "eli", "just"}
CUSTOM_STOPWORDS.update(FILLERS)

def clean_text(text: str) -> str:
    """
    Remove links, punctuation, stopwords, and low-value tokens.
    Normalize spacing and lowercase all text.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\*+|\>+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()

    tokens = [w for w in text.split() if w not in CUSTOM_STOPWORDS]
    return " ".join(tokens)

def clean_dataframe(input_path="data/reddit_raw.csv", output_path="data/reddit_clean.csv"):
    """
    Loads Reddit posts, cleans the text, and outputs a filtered CSV.
    - Merges title + selftext
    - Cleans text
    - Converts date column
    - Adds char count
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Merge title + selftext
    df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")

    # Clean text and compute char count
    df["clean_text"] = df["text"].apply(clean_text)
    df["char_count"] = df["clean_text"].str.len()

    # Convert timestamp for future time-based analysis
    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    # Keep only meaningful posts
    df = df[df["char_count"] > 30]

    # Save cleaned dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ Cleaned data saved to {output_path} — {len(df)} posts kept.")

if __name__ == "__main__":
    clean_dataframe()
