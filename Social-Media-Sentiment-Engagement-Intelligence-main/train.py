import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression

# Make sure the models directory exists
os.makedirs("models", exist_ok=True)

# 1. Train and save Sentiment Model & Vectorizer
texts = ["I love this product, it is amazing!", "This is the worst experience ever.", "It is okay, nothing special."]
labels = ["positive", "negative", "neutral"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

sentiment_model = MultinomialNB()
sentiment_model.fit(X, labels)

with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# 2. Train and save Engagement Model
X_eng = [[1, 2], [3, 4], [5, 6]]
y_eng = [10, 20, 30]
engagement_model = LinearRegression()
engagement_model.fit(X_eng, y_eng)

with open("models/engagement_model.pkl", "wb") as f:
    pickle.dump(engagement_model, f)

print("All models created and saved successfully!")