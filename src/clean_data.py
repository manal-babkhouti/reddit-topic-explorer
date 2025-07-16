# src/clean_data.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords if not already present
nltk.download("stopwords")

# === Custom stopword list: NLTK + Reddit-specific fillers ===
CUSTOM_STOPWORDS = set(stopwords.words("english"))
REDDIT_FILLERS = {
    "im", "like", "would", "get", "one", "also", "really",
    "see", "even", "cant", "dont", "didnt", "eli", "just"
}
CUSTOM_STOPWORDS.update(REDDIT_FILLERS)


def clean_text(text: str) -> str:
    """
    Clean a single text string:
    - Remove links, punctuation, and filler words
    - Lowercase and normalize whitespace

    Args:
        text: Input text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"[\r\n]+", " ", text)              # remove line breaks
    text = re.sub(r"http\S+", "", text)               # remove links
    text = re.sub(r"\*+|\>+", "", text)               # remove markdown
    text = re.sub(r"[^a-zA-Z\s]", "", text)           # keep only letters
    text = re.sub(r"\s+", " ", text)                  # normalize spaces
    text = text.lower().strip()

    tokens = [w for w in text.split() if w not in CUSTOM_STOPWORDS]
    return " ".join(tokens)


def clean_dataframe(
    input_path: str = "data/reddit_raw.csv",
    output_path: str = "data/reddit_clean.csv"
) -> None:
    """
    Load raw Reddit data, clean it, and save a filtered version.

    - Merges title + selftext
    - Cleans text and removes low-effort posts
    - Adds character count
    - Converts timestamp

    Args:
        input_path: Path to raw CSV
        output_path: Path to cleaned CSV
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["clean_text"] = df["text"].apply(clean_text)
    df["char_count"] = df["clean_text"].str.len()
    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

    df = df[df["char_count"] > 30]

    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Cleaned data saved to {output_path} — {len(df)} posts kept.")


if __name__ == "__main__":
    clean_dataframe()
