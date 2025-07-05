# src/fetch_data.py

import os
import praw
import pandas as pd
import datetime
from dotenv import load_dotenv

# === Load Reddit API credentials ===
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# === List of subreddits to crawl ===
TARGET_SUBREDDITS = [
    "MachineLearning",
    "learnmachinelearning",
    "artificial",
    "deeplearning"
]

def fetch_subreddit_posts(subreddit_name, limit=500):
    """
    Fetch top posts from a given subreddit using PRAW.
    Filters out posts with short or empty selftext.
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

def fetch_all_subreddits(subreddits, post_limit=500):
    """
    Fetch posts from a list of subreddits and merge into a single DataFrame.
    """
    all_data = []
    for sub in subreddits:
        sub_data = fetch_subreddit_posts(sub, post_limit)
        all_data.extend(sub_data)

    df = pd.DataFrame(all_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/reddit_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n🧾 Saved {len(df)} total posts to data/reddit_raw.csv")
    return df

if __name__ == "__main__":
    fetch_all_subreddits(TARGET_SUBREDDITS, post_limit=500)
