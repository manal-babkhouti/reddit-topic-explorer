# src/fetch_data.py

import os
import praw
import pandas as pd
import datetime
from dotenv import load_dotenv

# === Load credentials from .env ===
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# === Target subreddits ===
TARGET_SUBREDDITS = [
    "MachineLearning",
    "learnmachinelearning",
    "artificial",
    "deeplearning"
]

def fetch_subreddit_posts(subreddit_name: str, limit: int = 500) -> list[dict]:
    """
    Fetch top posts from a subreddit.
    
    Args:
        subreddit_name: Name of the subreddit
        limit: Max number of posts to fetch

    Returns:
        List of post dictionaries
    """
    posts = []
    print(f"🔍 Fetching from r/{subreddit_name}...")

    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.top(time_filter="month", limit=limit):
            if post.selftext and len(post.selftext.strip()) > 50:
                posts.append({
                    "subreddit": subreddit_name,
                    "id": post.id,
                    "created_utc": datetime.datetime.fromtimestamp(post.created_utc),
                    "title": post.title.strip(),
                    "selftext": post.selftext.strip()
                })
    except Exception as e:
        print(f"⚠️ Error fetching from r/{subreddit_name}: {e}")

    print(f"✅ {len(posts)} posts collected from r/{subreddit_name}")
    return posts


def fetch_all_subreddits(subreddits: list[str], post_limit: int = 500) -> pd.DataFrame:
    """
    Fetch posts from multiple subreddits and save them to CSV.

    Args:
        subreddits: List of subreddit names
        post_limit: Max number of posts per subreddit

    Returns:
        Combined DataFrame of all posts
    """
    all_data = []
    for sub in subreddits:
        all_data.extend(fetch_subreddit_posts(sub, post_limit))

    df = pd.DataFrame(all_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/reddit_raw.csv", index=False, encoding="utf-8-sig")

    print(f"\n🧾 Saved {len(df)} total posts to data/reddit_raw.csv")
    return df


if __name__ == "__main__":
    fetch_all_subreddits(TARGET_SUBREDDITS, post_limit=500)
