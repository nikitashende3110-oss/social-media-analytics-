import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the new training dataset
DATA_PATH = os.path.join("training_data", "train_csv")
df = pd.read_csv(DATA_PATH)

# Keep only rows with usable text and sentiment labels
df = df.dropna(subset=["text", "sentiment"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
df = df[df["text"] != ""]

# Keep the three required sentiment classes
df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]

print("Usable examples:", len(df))
print("\nClass distribution:")
print(df["sentiment"].value_counts())

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build the text-processing and sentiment-classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=["negative", "neutral", "positive"]))

# Save the complete pipeline for the dashboard
os.makedirs("models", exist_ok=True)

with open(os.path.join("models", "sentiment_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("\nNew sentiment model saved successfully!")