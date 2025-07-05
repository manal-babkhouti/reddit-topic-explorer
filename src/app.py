# 📁 File: app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import re

# === Load Data ===
@st.cache_data
def load_data(path):
    return pd.read_csv(path, encoding="utf-8-sig")

data_options = {
    "KMeans (k=4)": "data/reddit_kmeans_k4_clusters.csv",
    "KMeans (k=5)": "data/reddit_kmeans_k5_clusters.csv",
    "HDBSCAN": "data/reddit_hdbscan_clustered.csv",
}

st.set_page_config(page_title="Reddit Clustering Explorer", layout="wide")

# === Theme Toggle ===
theme = st.sidebar.radio("🌃 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            .css-18e3th9, .css-1d391kg, .css-1cpxqw2 {
                background-color: #0e1117;
                color: #fafafa;
            }
            .markdown-text-container, .css-10trblm, .css-1v3fvcr {
                color: #fafafa !important;
            }
            .css-1avcm0n, .css-1cpxqw2 {
                background-color: #161a21;
            }
            .css-ffhzg2, .css-1d391kg, .css-1v0mbdj, .css-1x8cf1d {
                color: #fafafa;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #fafafa;
            }
            .stButton>button, .stDownloadButton>button {
                background-color: #4a4a4a;
                color: white;
            }
            .stRadio>div>label, .stCheckbox>div>label {
                color: #fafafa;
            }
            .highlight {
                background-color: #ffea00;
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("🔍 Reddit Clustering Explorer")

selected_algo = st.sidebar.radio("Select Clustering Algorithm", list(data_options.keys()))

path = data_options[selected_algo]
try:
    df = load_data(path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# === Preprocessing ===
label_col = [col for col in df.columns if col.startswith("kmeans") or col.startswith("hdb") or col == "cluster"]
if not label_col:
    st.error("No clustering label column found in the dataset.")
    st.stop()
label_col = label_col[0]

text_col = "text"
df[label_col] = df[label_col].astype(int)

# === Tabs Adjusted by Algorithm ===
tabs = ["1️⃣ Overview", "2️⃣ Visualizations", "3️⃣ Explore Cluster"]
if "HDBSCAN" in selected_algo:
    tabs[0] += " (with Noise Ratio)"
else:
    tabs[0] += " (with Silhouette Score)"

tab1, tab2, tab3 = st.tabs(tabs)

with tab1:
    st.header("📊 Cluster Overview")
    num_clusters = df[label_col].nunique() - (1 if -1 in df[label_col].unique() else 0)
    cluster_counts = df[df[label_col] != -1][label_col].value_counts().sort_index()

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", num_clusters)

    if "kmeans" in label_col:
        X = df.select_dtypes(include="number").drop(columns=[label_col], errors="ignore")
        if df[label_col].nunique() > 1:
            sil_score = silhouette_score(X, df[label_col])
            db_score = davies_bouldin_score(X, df[label_col])
            col2.metric("Silhouette Score 🧪", f"{sil_score:.3f}", help="-1 to 1 → higher is better. Shows how well-separated the clusters are.")
            col3.metric("Davies–Bouldin 🧪", f"{db_score:.3f}", help="Lower is better. Measures cluster compactness vs separation.")
            if sil_score < 0:
                st.warning("⚠️ Silhouette Score is negative — clusters may be poorly formed. Consider adjusting k or preprocessing.")
        else:
            col2.warning("Not enough clusters for silhouette score.")
            col3.warning("Not enough clusters for DB index.")
    else:
        noise_ratio = (df[label_col] == -1).mean()
        col2.metric("Noise Ratio 🧪", f"{noise_ratio:.2%}", help="Percentage of points not assigned to any cluster.")
        col3.metric("Total Posts", len(df))

    # === Summary Card ===
    st.markdown("""
        <div style='padding: 10px; border-radius: 10px; background-color: #2b313c; margin-top: 20px;'>
        <h4 style='color: #fafafa;'>📌 Summary:</h4>
        <ul style='color: #fafafa;'>
            <li>Data clustered using <b>{}</b></li>
            <li>Number of clusters detected: <b>{}</b></li>
            <li>Total posts analyzed: <b>{}</b></li>
        </ul>
        </div>
    """.format(selected_algo, num_clusters, len(df)), unsafe_allow_html=True)

    # === Optional: Comparison Table ===
    st.markdown("### 📊 Clustering Algorithm Comparison")
    comparison_data = []
    for name, path in data_options.items():
        df_cmp = load_data(path)
        lbl = [col for col in df_cmp.columns if col.startswith("kmeans") or col.startswith("hdb") or col == "cluster"]
        if not lbl:
            continue
        lbl = lbl[0]
        df_cmp[lbl] = df_cmp[lbl].astype(int)
        n_clusters = df_cmp[lbl].nunique() - (1 if -1 in df_cmp[lbl].unique() else 0)
        total = len(df_cmp)

        if "kmeans" in lbl:
            X_cmp = df_cmp.select_dtypes(include="number").drop(columns=[lbl], errors="ignore")
            if df_cmp[lbl].nunique() > 1:
                sil = silhouette_score(X_cmp, df_cmp[lbl])
                db = davies_bouldin_score(X_cmp, df_cmp[lbl])
            else:
                sil, db = None, None
            comparison_data.append([name, n_clusters, total, sil, db, None])
        else:
            noise = (df_cmp[lbl] == -1).mean()
            comparison_data.append([name, n_clusters, total, None, None, noise])

    df_compare = pd.DataFrame(comparison_data, columns=["Algorithm", "Clusters", "Posts", "Silhouette", "DB Index", "Noise Ratio"])
    st.dataframe(df_compare.style.format({"Silhouette": "{:.3f}", "DB Index": "{:.3f}", "Noise Ratio": "{:.2%}"}))

with tab2:
    st.subheader("📦 Cluster Sizes")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    cluster_counts.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Posts")
    ax.set_title("Distribution of Posts per Cluster")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("📌 Cluster Visualization (PCA)")
    if {"pca_1", "pca_2"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(5, 2.8))
        sns.scatterplot(
            x=df["pca_1"],
            y=df["pca_2"],
            hue=df[label_col].astype(str),
            palette="tab10",
            ax=ax,
            legend=False
        )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("PCA Projection of Clusters")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("PCA coordinates not found in this dataset.")

with tab3:
    st.subheader("🔍 Explore a Cluster")
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # === Filters ===
    with st.expander("🔍 Advanced Filters", expanded=True):
        keyword_filter = st.text_input("Filter by keyword in text")

    # === Cluster selection ===
    available_clusters = sorted([c for c in df[label_col].unique() if c != -1])
    selected_cluster = st.radio("Select Cluster", available_clusters, horizontal=True)

    filtered_df = df[df[label_col] == selected_cluster]

    if keyword_filter:
        filtered_df = filtered_df[filtered_df[text_col].str.contains(keyword_filter, case=True, na=False)]

    # === Word Cloud dynamically generated from cluster text ===
    st.markdown("**🧠 Word Cloud from Cluster Text:**")
    full_text = " ".join(filtered_df[text_col].dropna().astype(str).tolist())

    if len(full_text.strip()) > 0:
        wordcloud = WordCloud(width=600, height=300, background_color="white", stopwords=None).generate(full_text)
        fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("No text available for this cluster.")

    # === Show Sample Posts with Pagination ===
    st.markdown("**📝 Sample Posts:**")

    # Extract posts from the selected cluster
    examples = df[df[label_col] == selected_cluster][text_col].dropna().astype(str).reset_index(drop=True)

    # Apply keyword filter
    if keyword_filter:
        examples = examples[examples.str.contains(keyword_filter, case=True, na=False)]

    total_posts = len(examples)

    # Show count
    st.markdown(f"**Total posts found:** {total_posts}")

    # Set pagination params
    posts_per_page = 5
    total_pages = (total_posts - 1) // posts_per_page + 1

    # Avoid breaking if no posts
    if total_posts == 0:
        st.info("No posts found for this cluster with the current filter.")
    else:
        selected_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)

        start_idx = (selected_page - 1) * posts_per_page
        end_idx = min(start_idx + posts_per_page, total_posts)

        for i, post in enumerate(examples[start_idx:end_idx], start=start_idx + 1):
            if keyword_filter:
                pattern = re.compile(f"({re.escape(keyword_filter)})", re.IGNORECASE)
                post = pattern.sub(r"<mark class='highlight'>\1</mark>", post)

            st.markdown(f"**Post {i}**", unsafe_allow_html=True)
            st.text_area(label="", value=post, height=150, key=f"post_{i}")
            st.markdown("---")


    # === Optional thumbnails ===
    if "image_url" in df.columns:
        st.markdown("**🖼️ Related Thumbnails:**")
        imgs = filtered_df["image_url"].dropna().unique()[:5]
        for img_url in imgs:
            st.image(img_url, width=200)


# === Download Option ===
st.sidebar.markdown("---")
if st.sidebar.button("⬇️ Download CSV"):
    st.sidebar.download_button(
        label="Download Clustered Data",
        data=df.to_csv(index=False, encoding="utf-8-sig"),
        file_name=os.path.basename(path),
        mime="text/csv"
    )

# === Visual Mockup ===
st.sidebar.markdown("---")
st.sidebar.info("To preview the visual layout, scroll through the tabs above.")