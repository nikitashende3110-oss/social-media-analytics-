import sqlite3
import os
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def get_db_data():
    empty_data = {
        "total_posts": 0,
        "total_likes": 0,
        "total_shares": 0,
        "sentiments": {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        },
        "popular_hashtags": [],
        "top_posts": []
    }

    csv_path = os.path.join(app.config["UPLOAD_FOLDER"], "dataset.csv")

    if not os.path.exists(csv_path):
        return empty_data

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return empty_data

    required = [
        "post_text", "likes", "shares", "comments",
        "sentiment", "platform", "hashtag"
    ]

    if not all(column in df.columns for column in required):
        return empty_data

    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    df["comments"] = pd.to_numeric(df["comments"], errors="coerce").fillna(0)

    total_posts = len(df)
    total_likes = int(df["likes"].sum())
    total_shares = int(df["shares"].sum())

    sentiment_values = df["sentiment"].fillna("").astype(str).str.lower()
    sentiments = {
        "positive": int((sentiment_values == "positive").sum()),
        "neutral": int((sentiment_values == "neutral").sum()),
        "negative": int((sentiment_values == "negative").sum())
    }

    hashtags_count = {}
    for tags in df["hashtag"].dropna():
        for tag in str(tags).split(","):
            tag = tag.strip()
            if tag:
                hashtags_count[tag] = hashtags_count.get(tag, 0) + 1

    popular_hashtags = sorted(
        hashtags_count.items(),
        key=lambda item: item[1],
        reverse=True
    )[:6]

    df["engagement"] = df["likes"] + df["shares"] + df["comments"]
    top_posts = []

    for _, row in df.nlargest(5, "engagement").iterrows():
        sentiment = str(row["sentiment"]).capitalize()
        top_posts.append({
            "text": str(row["post_text"]),
            "likes": int(row["likes"]),
            "shares": int(row["shares"]),
            "comments": int(row["comments"]),
            "sentiment": sentiment,
            "platform": str(row["platform"]),
            "engagement": int(row["engagement"])
        })

    return {
        "total_posts": total_posts,
        "total_likes": total_likes,
        "total_shares": total_shares,
        "sentiments": sentiments,
        "popular_hashtags": popular_hashtags,
        "top_posts": top_posts
    }


@app.route("/", methods=["GET", "POST"])
def dashboard():
    message = ""

    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or not file.filename:
            message = "Please select a CSV file."
        elif not file.filename.lower().endswith(".csv"):
            message = "Please upload a CSV file."
        else:
            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "dataset.csv"
            )
            file.save(save_path)
            message = "Dataset uploaded successfully!"

    data = get_db_data()
    return render_template("index.html", data=data, message=message)


if __name__ == "__main__":
    app.run(debug=True, port=5000)