# 📁 File: src/email_dispatcher.py

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ssl
from datetime import datetime
import os
from dotenv import load_dotenv

# ========== CONFIG ==========
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
REPORT_LINK = os.getenv("REPORT_LINK")

# ========== SUMMARY LOGIC ==========
def generate_summary():
    dfs = {
        "KMeans (k=4)": pd.read_csv("data/reddit_kmeans_k4_clusters.csv"),
        "KMeans (k=5)": pd.read_csv("data/reddit_kmeans_k5_clusters.csv"),
        "HDBSCAN": pd.read_csv("data/reddit_hdbscan_clustered.csv"),
    }

    month_title = datetime.now().strftime("%B %Y")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_lines = [
        f"<h2 style='margin-bottom:0;'>📊 Reddit Clustering Summary – {month_title}</h2>",
        f"<p style='color:gray;'>Generated on: {timestamp}</p><hr>"
    ]

    image_paths = []
    insights = []
    best_algo = None
    max_clusters = 0
    inline_image = ""

    for name, df in dfs.items():
        if "cluster" in df.columns:
            label_col = "cluster"
        else:
            label_col_candidates = [col for col in df.columns if col.startswith("kmeans") or col.startswith("hdb")]
            if not label_col_candidates:
                raise ValueError(f"❌ No clustering label column found in {name}")
            label_col = label_col_candidates[0]

        cluster_counts = df[label_col].value_counts().sort_index()
        n_clusters = df[label_col].nunique() - (1 if -1 in df[label_col].values else 0)
        if n_clusters > max_clusters:
            max_clusters = n_clusters
            best_algo = name

        summary_lines.append(f"<h3 style='margin-top:20px'>{name}</h3>")
        summary_lines.append(f"<ul>\n<li><b>Clusters:</b> {n_clusters}</li>\n<li><b>Total Posts:</b> {len(df)}</li>")

        keyword_file = f"data/cluster_top_keywords_{label_col}.csv"
        if os.path.exists(keyword_file):
            kw_df = pd.read_csv(keyword_file)
            top_keywords = kw_df['top_keywords'].tolist()
            summary_lines.append("<li><b>Top Keywords:</b><ul>")
            for i, kw in enumerate(top_keywords):
                summary_lines.append(f"<li>Cluster {i}: {kw}</li>")
            summary_lines.append("</ul></li>")
        summary_lines.append("</ul>")

        fig, ax = plt.subplots(figsize=(6, 4))
        cluster_counts.plot(kind="bar", color="#87CEEB", ax=ax)
        ax.set_title(f"{name} – Cluster Sizes")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("# of Posts")
        plt.tight_layout()
        img_path = f"summary_{name.replace(' ', '_')}.png"
        fig.savefig(img_path)
        image_paths.append(img_path)
        plt.close()

        if "inline_image" not in locals():
            inline_image = img_path

    if best_algo:
        insights.append(f"💡 <b>Insight:</b> <i>{best_algo}</i> found the most clusters ({max_clusters}).")

    return "\n".join(summary_lines), image_paths, insights, inline_image

# ========== EMAIL SENDING ==========
def send_email():
    print("📨 Sending summary email...")
    summary_html, attachments, insights, inline_img_path = generate_summary()

    msg = MIMEMultipart("related")
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"Monthly Reddit Clustering Report – {datetime.now().strftime('%B %Y')}"

    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)

    intro = """
    <p>Hello 👋,</p>
    <p>Here is your monthly Reddit clustering analysis, including number of clusters, key topics, and post distributions.</p>
    """

    insight_html = "".join([f"<p>{line}</p>" for line in insights])
    inline_img_tag = f"<img src='cid:inline_chart' style='max-width:500px; margin-top:10px;' alt='Chart'>" if inline_img_path else ""

    outro = f"""
    <p style='margin-top:20px'>You can view the full dashboard or access additional resources <a href='{REPORT_LINK}' target='_blank'>here</a>.</p>
    <p style='margin-top:20px'>Best regards,<br><b>Reddit Clustering Bot</b> 🤖</p>
    """

    html_body = f"{intro}{insight_html}{summary_html}{inline_img_tag}{outro}"
    msg_alt.attach(MIMEText(html_body, 'html'))

    all_files = attachments + [
        "data/reddit_kmeans_k4_clusters.csv",
        "data/reddit_kmeans_k5_clusters.csv",
        "data/reddit_hdbscan_clustered.csv"
    ]

    for file_path in all_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping missing file: {file_path}")
            continue
        with open(file_path, "rb") as file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {os.path.basename(file_path)}",
            )
            msg.attach(part)

    if inline_img_path and os.path.exists(inline_img_path):
        with open(inline_img_path, 'rb') as img:
            mime_img = MIMEBase('image', 'png', name=os.path.basename(inline_img_path))
            mime_img.set_payload(img.read())
            encoders.encode_base64(mime_img)
            mime_img.add_header('Content-ID', '<inline_chart>')
            mime_img.add_header('Content-Disposition', 'inline', filename=os.path.basename(inline_img_path))
            msg.attach(mime_img)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)

    print("✅ Email sent successfully.")

if __name__ == "__main__":
    send_email()