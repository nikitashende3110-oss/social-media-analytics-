import pandas as pd

# Read the original dataset
df = pd.read_csv("sentimentdataset.csv")

# Convert columns to the names expected by app.py
dashboard_df = pd.DataFrame({
    "post_text": df["Text"].fillna("").astype(str).str.strip(),
    "likes": pd.to_numeric(df["Likes"], errors="coerce").fillna(0),
    "shares": pd.to_numeric(df["Retweets"], errors="coerce").fillna(0),
    "comments": 0,
    "sentiment": df["Sentiment"].fillna("").astype(str).str.strip().str.lower(),
    "platform": df["Platform"].fillna("Unknown").astype(str).str.strip(),
    "hashtag": df["Hashtags"].fillna("").astype(str).str.strip()
})

dashboard_df.to_csv("dashboard_dataset.csv", index=False)

print("Dashboard dataset created successfully!")
print("Total posts:", len(dashboard_df))
print(dashboard_df.head())