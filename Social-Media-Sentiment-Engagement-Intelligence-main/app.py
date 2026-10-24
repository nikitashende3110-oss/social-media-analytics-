import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained sentiment model
MODEL_PATH = os.path.join("models", "sentiment_model.pkl")

with open(MODEL_PATH, "rb") as f:
    sentiment_model = pickle.load(f)

# CSV upload settings
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# Predict sentiment for a single post
def predict_sentiment(text):
    return sentiment_model.predict([text])[0]


# Live sentiment prediction route
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()

    if not text:
        return render_template(
            "index.html",
            data=get_db_data(),
            message="",
            prediction=None,
            analyzed_text=""
        )

    sentiment = predict_sentiment(text)

    return render_template(
        "index.html",
        data=get_db_data(),
        message="",
        prediction=sentiment,
        analyzed_text=text
    )


# Read and calculate dashboard data
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

    csv_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "dataset.csv"
    )

    if not os.path.exists(csv_path):
        return empty_data

    try:
        df = pd.read_csv(csv_path)
    except Exception as error:
        print("Error reading CSV:", error)
        return empty_data

    required = [
        "post_text", "likes", "shares", "comments",
        "sentiment", "platform", "hashtag"
    ]

    if not all(column in df.columns for column in required):
        print("CSV is missing required columns.")
        return empty_data

    # Clean numeric columns
    for column in ["likes", "shares", "comments"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        ).fillna(0)

    # Main metrics
    total_posts = len(df)
    total_likes = int(df["likes"].sum())
    total_shares = int(df["shares"].sum())

    # Normalize sentiment labels
    sentiment_values = (
        df["sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Labels commonly associated with positive sentiment
    positive_labels = {
        "positive", "joy", "excitement", "contentment",
        "gratitude", "serenity", "happy", "hopeful",
        "awe", "acceptance", "pride", "elation", "euphoria",
        "enthusiasm", "determination", "playful",
        "inspiration", "happiness", "hope", "empowerment",
        "inspired", "admiration", "calmness", "compassion",
        "tenderness", "fulfillment", "reverence", "love",
        "amusement", "anticipation", "kind", "empathetic",
        "free-spirited", "confident", "satisfaction",
        "accomplishment", "harmony", "creativity", "wonder",
        "adventure", "enjoyment", "affection", "adoration",
        "zest", "radiance", "coziness", "resilience",
        "tranquility", "captivation", "overjoyed",
        "motivation", "blessed", "appreciation", "confidence",
        "wonderment", "optimism", "intrigue", "playfuljoy",
        "mindfulness", "freedom", "dazzle", "spark", "marvel",
        "positivity", "kindness", "friendship", "success",
        "amazement", "romance", "grandeur", "energy",
        "celebration", "charm", "ecstasy", "connection",
        "iconic", "triumph", "heartwarming", "solace",
        "breakthrough", "relief", "thrill", "proud",
        "grateful", "enchantment", "renewed effort",
        "joy in baking", "creative inspiration", "nature's beauty",
        "winter magic", "celestial wonder", "culinary adventure",
        "festivejoy", "joyfulreunion", "melodic", "elegance",
        "artisticburst", "culinaryodyssey", "immersion",
        "engagement", "touched", "vibrancy", "mesmerizing",
        "runway creativity", "ocean's freedom", "radiance",
        "yearning"
    }

    # Labels commonly associated with negative sentiment
    negative_labels = {
        "negative", "despair", "grief", "loneliness", "sad",
        "frustration", "regret", "numbness", "melancholy",
        "hate", "bad", "disgust", "bitterness", "frustrated",
        "betrayal", "bitter", "shame", "jealousy", "resentment",
        "fearful", "jealous", "devastated", "envious",
        "dismissive", "heartbreak", "anger", "fear", "sadness",
        "disappointed", "anxiety", "intimidation", "helplessness",
        "envy", "apprehensive", "isolation", "disappointment",
        "sorrow", "loss", "suffering", "lostlove", "exhaustion",
        "darkness", "desperation", "ruins", "heartache",
        "solitude", "emotionalstorm", "overwhelmed",
        "desolation", "pressure", "obstacle", "miscalculation",
        "challenge", "embarrassed"
    }

    # Labels treated as neutral or mixed/uncertain
    neutral_labels = {
        "neutral", "curiosity", "confusion", "nostalgia",
        "surprise", "indifference", "ambivalence", "reflection",
        "arousal", "boredom", "contemplation", "exploration",
        "mischievous", "pensive", "innerjourney", "suspense",
        "emotion", "whimsy", "dreamchaser", "envisioning history",
        "imagination", "bittersweet"
    }

    # Assign labels to three groups.
    # Any unrecognized label is counted as neutral so that
    # all dataset rows are included in the distribution.
    sentiment_groups = []

    for label in sentiment_values:
        if label in positive_labels:
            sentiment_groups.append("positive")
        elif label in negative_labels:
            sentiment_groups.append("negative")
        else:
            sentiment_groups.append("neutral")

    grouped_sentiments = pd.Series(
        sentiment_groups, index=df.index
    )

    sentiments = {
        "positive": int((grouped_sentiments == "positive").sum()),
        "neutral": int((grouped_sentiments == "neutral").sum()),
        "negative": int((grouped_sentiments == "negative").sum())
    }

    # Count popular hashtags
    hashtags_count = {}

    for tags in df["hashtag"].fillna(""):
        for tag in str(tags).replace(",", " ").split():
            tag = tag.strip()

            if tag:
                hashtags_count[tag] = (
                    hashtags_count.get(tag, 0) + 1
                )

    popular_hashtags = sorted(
        hashtags_count.items(),
        key=lambda item: item[1],
        reverse=True
    )[:6]

    # Calculate engagement
    df["engagement"] = (
        df["likes"] + df["shares"] + df["comments"]
    )

    # Select five posts with the highest engagement
    top_posts = []

    for _, row in df.nlargest(5, "engagement").iterrows():
        original_sentiment = str(row["sentiment"]).strip().lower()

        if original_sentiment in positive_labels:
            display_sentiment = "Positive"
        elif original_sentiment in negative_labels:
            display_sentiment = "Negative"
        else:
            display_sentiment = "Neutral"

        top_posts.append({
            "text": str(row["post_text"]),
            "likes": int(row["likes"]),
            "shares": int(row["shares"]),
            "comments": int(row["comments"]),
            "sentiment": display_sentiment,
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


# Main dashboard and CSV upload
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
                app.config["UPLOAD_FOLDER"],
                "dataset.csv"
            )
            file.save(save_path)
            message = "Dataset uploaded successfully!"

    data = get_db_data()

    return render_template(
        "index.html",
        data=data,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)