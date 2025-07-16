# 📁 File: app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from wordcloud import WordCloud
import os
import re

# =======================
# 🔧 Config & Setup
# =======================
st.set_page_config(page_title="Reddit Clustering Explorer", layout="wide")

@st.cache_data
def load_data(path):
    return pd.read_csv(path, encoding="utf-8-sig")

# =======================
# 📂 File Paths
# =======================
BASIC_PATHS = {
    "KMeans (k=4)": "data/reddit_kmeans_k4_clusters.csv",
    "KMeans (k=5)": "data/reddit_kmeans_k5_clusters.csv",
    "HDBSCAN": "data/reddit_hdbscan_clustered.csv",
}
SUMMARY_PATHS = {
    "KMeans (k=4)": "data/reddit_kmeans_k4_with_summary.csv",
    "KMeans (k=5)": "data/reddit_kmeans_k5_with_summary.csv",
    "HDBSCAN": "data/reddit_hdbscan_with_summary.csv",
}

# =======================
# 🎨 Dark Theme Toggle
# =======================
theme = st.sidebar.radio("🌃 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            .stButton>button, .stDownloadButton>button {
                background-color: #4a4a4a;
                color: white;
            }
            .highlight {
                background-color: #ffea00;
                color: black;
                padding: 0 3px;
                border-radius: 4px;
            }
        </style>
    """, unsafe_allow_html=True)

# =======================
# 🔘 Dataset Selection
# =======================
st.title("🔍 Reddit Clustering Explorer")
selected_algo = st.sidebar.radio("🔢 Select Clustering Algorithm", list(BASIC_PATHS.keys()))
version_type = st.sidebar.radio("📄 Data Version", ["Basic", "With Summary"])
path = SUMMARY_PATHS[selected_algo] if version_type == "With Summary" else BASIC_PATHS[selected_algo]

try:
    df = load_data(path)
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

label_col = next((col for col in df.columns if col.startswith("kmeans") or col.startswith("hdb") or col == "cluster"), None)
if not label_col:
    st.error("❌ No clustering label column found.")
    st.stop()

df[label_col] = df[label_col].astype(int)
text_col = "text"

# =======================
# 🗂️ Tabs Setup
# =======================
tabs = ["1️⃣ Overview", "2️⃣ Visualizations", "3️⃣ Explore Cluster"]
tabs[0] += " (Noise Ratio)" if "HDBSCAN" in selected_algo else " (Silhouette Score)"
tab1, tab2, tab3 = st.tabs(tabs)

# =======================
# 📌 TAB 1: Overview
# =======================
with tab1:
    st.header("📊 Cluster Overview")
    num_clusters = df[label_col].nunique() - (1 if -1 in df[label_col].unique() else 0)
    cluster_counts = df[df[label_col] != -1][label_col].value_counts().sort_index()

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", num_clusters)

    if "kmeans" in label_col:
        X = df.select_dtypes(include="number").drop(columns=[label_col], errors="ignore")
        if num_clusters > 1:
            sil = silhouette_score(X, df[label_col])
            db = davies_bouldin_score(X, df[label_col])
            col2.metric("Silhouette Score", f"{sil:.3f}")
            col3.metric("Davies–Bouldin Index", f"{db:.3f}")
        else:
            col2.warning("Not enough clusters.")
            col3.warning("Not enough clusters.")
    else:
        noise = (df[label_col] == -1).mean()
        col2.metric("Noise Ratio", f"{noise:.2%}")
        col3.metric("Total Posts", len(df))

    st.markdown(f"""
        <div style='padding: 10px; border-radius: 10px; background-color: #2b313c; color:#fafafa'>
        <h4>🧾 Summary:</h4>
        <ul>
            <li><b>Algorithm:</b> {selected_algo}</li>
            <li><b>Version:</b> {version_type}</li>
            <li><b>Clusters:</b> {num_clusters}</li>
            <li><b>Total Posts:</b> {len(df)}</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    # 📈 Show Elbow Plot if KMeans
    if "kmeans" in label_col:
        st.subheader("📈 K Value Tuning (Elbow Method)")
        if os.path.exists("images/elbow.jpg"):
            st.image("images/elbow.jpg", caption="Elbow Plot (Inertia vs K)", width=400, use_container_width=True)
        else:
            st.info("Elbow plot image not found in /images.")

# =======================
# 📉 TAB 2: Visualizations
# =======================
with tab2:
    st.subheader("📦 Cluster Sizes")
    fig, ax = plt.subplots(figsize=(4, 2))
    cluster_counts.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Distribution of Posts per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Posts")
    st.pyplot(fig)

    st.subheader("📌 PCA Cluster Visualization")
    if {"pca_1", "pca_2"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.scatterplot(data=df, x="pca_1", y="pca_2", hue=df[label_col].astype(str), palette="tab10", ax=ax, legend=False)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("PCA Projection of Clusters")
        st.pyplot(fig)
    else:
        st.warning("❌ PCA coordinates not found in dataset.")

# =======================
# 🔍 TAB 3: Explore Cluster
# =======================
with tab3:
    st.subheader("🔍 Explore a Cluster")

    with st.expander("🛠️ Advanced Filters", expanded=True):
        keyword_filter = st.text_input("Filter by keyword in text")

    available_clusters = sorted([c for c in df[label_col].unique() if c != -1])
    selected_cluster = st.radio("Select Cluster", available_clusters, horizontal=True)

    cluster_df = df[df[label_col] == selected_cluster]

    # 🧠 Show Cluster Summary Keywords
    if "summary_keywords" in df.columns:
        cluster_keywords = cluster_df["summary_keywords"].dropna()
        if not cluster_keywords.empty:
            all_keywords = ", ".join(cluster_keywords)
            keyword_list = list(set(k.strip() for k in all_keywords.split(",") if k.strip()))
            top_keywords = sorted(keyword_list)[:10]
            st.markdown("### 🧠 Cluster Summary Keywords")
            st.markdown(f"""
                <div style='background:#f9f9f9;padding:10px;margin-top:5px;margin-bottom:10px;
                            border-radius:8px;border:1px solid #ccc;font-size:15px'>
                    {' • '.join(top_keywords)}
                </div>
            """, unsafe_allow_html=True)

    if keyword_filter:
        cluster_df = cluster_df[cluster_df[text_col].str.contains(keyword_filter, case=False, na=False)]

    examples = cluster_df[text_col].dropna().astype(str)
    example_indices = examples.index.tolist()
    examples = examples.reset_index(drop=True)

    st.markdown("**🧠 Word Cloud from Cluster Text:**")
    full_text = " ".join(cluster_df[text_col].dropna().astype(str).tolist())
    if full_text.strip():
        wc = WordCloud(width=450, height=225, background_color="white").generate(full_text)
        fig_wc, ax_wc = plt.subplots(figsize=(4, 2))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("No text found for this cluster.")

    st.markdown("**📝 Sample Posts:**")
    total = len(examples)
    st.markdown(f"**Total posts found:** {total}")

    if total == 0:
        st.info("No posts match this cluster + filter.")
    else:
        page_size = 5
        total_pages = (total - 1) // page_size + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)

        for j in range((page - 1) * page_size, min(page * page_size, total)):
            post = examples[j]
            orig_idx = example_indices[j]

            # Highlight summary keywords
            if "summary_keywords" in df.columns and pd.notna(df.loc[orig_idx, "summary_keywords"]):
                summary_words = [w.strip() for w in str(df.loc[orig_idx, "summary_keywords"]).split(",") if w.strip()]
                for word in summary_words:
                    post = re.sub(
                        rf"(?<!\w)({re.escape(word)})(?!\w)",
                        r"<span style='background-color: #ffea00; color: black; padding: 0 2px; border-radius: 4px;'>\1</span>",
                        post,
                        flags=re.IGNORECASE
                    )

            # Highlight keyword filter
            if keyword_filter:
                post = re.sub(
                    rf"(?<!\w)({re.escape(keyword_filter)})(?!\w)",
                    r"<span style='background-color: #00ffc3; color: black; padding: 0 2px; border-radius: 4px;'>\1</span>",
                    post,
                    flags=re.IGNORECASE
                )

            st.markdown(f"**Post {j + 1}**", unsafe_allow_html=True)
            st.markdown(
                f"<div style='padding:10px;background:#f2f2f2;border-radius:5px'>{post}</div>",
                unsafe_allow_html=True
            )
            st.markdown("---")

# =======================
# ⬇️ Download Button
# =======================
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="⬇️ Download Clustered Data",
    data=df.to_csv(index=False, encoding="utf-8-sig"),
    file_name=os.path.basename(path),
    mime="text/csv"
)
