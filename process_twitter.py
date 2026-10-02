import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path("twitter_collection_results.json")
OUTPUT_DIR = Path("twitter_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 16


# ============================================================
# 1. LOAD RAW COLLECTION
# ============================================================

print("\nLoading Twitter collection...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle either a direct list or a wrapper dictionary
if isinstance(data, dict):
    if "results" in data:
        records = data["results"]
    elif "data" in data:
        records = data["data"]
    else:
        raise ValueError(
            "JSON is a dictionary but I could not find a 'results' or 'data' field."
        )
else:
    records = data

print(f"Collection windows found: {len(records)}")


# ============================================================
# 2. FLATTEN TWEETS
# ============================================================

rows = []
window_rows = []

for record in records:

    requested_date = record.get("requested_date_new_york")
    window_number = record.get("window_number")
    window_start = record.get("window_start_new_york")
    window_end = record.get("window_end_new_york")
    credential = record.get("credential_label")
    sampling = record.get("sampling")

    response = record.get("response") or {}

    tweets = response.get("tweets") or []
    has_next_page = response.get("has_next_page")
    next_cursor = response.get("next_cursor")

    window_rows.append({
        "requested_date": requested_date,
        "window_number": window_number,
        "window_start_ny": window_start,
        "window_end_ny": window_end,
        "credential": credential,
        "n_tweets": len(tweets),
        "has_next_page": has_next_page,
        "has_cursor": bool(next_cursor),
    })

    for tweet in tweets:

        author = tweet.get("author") or {}

        rows.append({
            "tweet_id": str(tweet.get("id", "")),
            "text": tweet.get("text", ""),
            "url": tweet.get("url") or tweet.get("twitterUrl"),

            "created_at": tweet.get("createdAt"),
            "lang": tweet.get("lang"),

            "requested_date": requested_date,
            "window_number": window_number,
            "window_start_ny": window_start,
            "window_end_ny": window_end,
            "credential": credential,
            "sampling": sampling,

            "is_reply": tweet.get("isReply", False),

            "likes": tweet.get("likeCount", 0) or 0,
            "retweets": tweet.get("retweetCount", 0) or 0,
            "replies": tweet.get("replyCount", 0) or 0,
            "quotes": tweet.get("quoteCount", 0) or 0,
            "views": tweet.get("viewCount", 0) or 0,
            "bookmarks": tweet.get("bookmarkCount", 0) or 0,

            "username": author.get("userName"),
            "author_name": author.get("name"),
            "followers": author.get("followers", 0) or 0,
            "verified": author.get("isVerified", False),
            "blue_verified": author.get("isBlueVerified", False),
        })


df = pd.DataFrame(rows)
window_df = pd.DataFrame(window_rows)

print(f"Raw tweets: {len(df):,}")


# ============================================================
# 3. QUALITY CONTROL
# ============================================================

if df.empty:
    raise ValueError("No tweets were found in the JSON.")

# Remove duplicates by tweet ID
before = len(df)

df = (
    df[df["tweet_id"].notna()]
    .drop_duplicates(subset=["tweet_id"])
    .copy()
)

duplicates_removed = before - len(df)

print(f"Duplicate tweets removed: {duplicates_removed:,}")
print(f"Unique tweets: {len(df):,}")


# Parse Twitter timestamps
df["created_at_utc"] = pd.to_datetime(
    df["created_at"],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce",
    utc=True,
)

# Convert UTC -> New York
df["created_at_ny"] = (
    df["created_at_utc"]
    .dt.tz_convert("America/New_York")
)

df["date_ny"] = df["created_at_ny"].dt.date.astype(str)

# Check whether returned tweet actually belongs to intended sample date
df["date_matches_window"] = df["date_ny"] == df["requested_date"]

# Keep English tweets
df = df[
    (df["lang"].fillna("").str.lower() == "en")
].copy()

# Remove empty text
df["text"] = df["text"].fillna("").astype(str)
df = df[df["text"].str.strip().ne("")].copy()


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def preprocess_tweet(text):
    """
    CardiffNLP recommended Twitter-style preprocessing:
    usernames -> @user
    URLs -> http
    """

    words = []

    for token in str(text).split():
        if token.startswith("@") and len(token) > 1:
            token = "@user"

        elif token.startswith("http"):
            token = "http"

        words.append(token)

    cleaned = " ".join(words)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


df["text_model"] = df["text"].apply(preprocess_tweet)


# ============================================================
# 5. ENGAGEMENT VARIABLES
# ============================================================

numeric_cols = [
    "likes",
    "retweets",
    "replies",
    "quotes",
    "views",
    "bookmarks",
    "followers",
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)


df["engagement"] = (
    df["likes"]
    + df["retweets"]
    + df["replies"]
    + df["quotes"]
)

# log weighting prevents giant viral tweets from dominating everything
df["engagement_weight"] = np.log1p(df["engagement"]) + 1


# ============================================================
# 6. LOAD TWITTER SENTIMENT MODEL
# ============================================================

print("\nLoading Twitter-RoBERTa sentiment model...")
print("The model downloads automatically the first time.\n")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()


# ============================================================
# 7. SCORE ALL TWEETS
# ============================================================

texts = df["text_model"].tolist()

probabilities = []

print("\nScoring tweets...")

for start in tqdm(
    range(0, len(texts), BATCH_SIZE),
    desc="Sentiment batches",
):

    batch = texts[start:start + BATCH_SIZE]

    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    with torch.no_grad():

        logits = model(**encoded).logits

        probs = torch.softmax(
            logits,
            dim=1,
        ).cpu().numpy()

    probabilities.append(probs)


probs = np.vstack(probabilities)


# Cardiff Twitter-RoBERTa:
# 0 = negative
# 1 = neutral
# 2 = positive

df["prob_negative"] = probs[:, 0]
df["prob_neutral"] = probs[:, 1]
df["prob_positive"] = probs[:, 2]


# Continuous sentiment:
# -1 approximately very negative
# +1 approximately very positive

df["sentiment_score"] = (
    df["prob_positive"]
    - df["prob_negative"]
)


labels = np.array([
    "negative",
    "neutral",
    "positive",
])

df["sentiment_label"] = labels[
    np.argmax(probs, axis=1)
]


# ============================================================
# 8. SIMPLE AI-RISK LANGUAGE MEASURE
# ============================================================

risk_terms = [
    "risk",
    "risks",
    "danger",
    "dangerous",
    "harm",
    "harmful",
    "threat",
    "unsafe",
    "safety",
    "kill",
    "killing",
    "death",
    "replace workers",
    "replace jobs",
    "job loss",
    "unemployment",
    "misinformation",
    "deepfake",
    "bias",
    "copyright",
    "lawsuit",
    "regulation",
    "regulate",
    "ban",
    "fraud",
    "scam",
    "bubble",
    "crash",
]

risk_pattern = re.compile(
    "|".join(
        re.escape(term)
        for term in risk_terms
    ),
    flags=re.IGNORECASE,
)

df["risk_flag"] = (
    df["text"]
    .str.contains(
        risk_pattern,
        na=False,
    )
    .astype(int)
)


# ============================================================
# 9. DAILY SENTIMENT AGGREGATION
# ============================================================

def weighted_average(group):

    values = group["sentiment_score"].to_numpy()
    weights = group["engagement_weight"].to_numpy()

    if weights.sum() == 0:
        return values.mean()

    return np.average(
        values,
        weights=weights,
    )


daily_basic = (
    df
    .groupby("date_ny")
    .agg(
        n_posts=("tweet_id", "count"),

        mean_sentiment=(
            "sentiment_score",
            "mean",
        ),

        median_sentiment=(
            "sentiment_score",
            "median",
        ),

        positive_share=(
            "sentiment_label",
            lambda x: (x == "positive").mean(),
        ),

        negative_share=(
            "sentiment_label",
            lambda x: (x == "negative").mean(),
        ),

        neutral_share=(
            "sentiment_label",
            lambda x: (x == "neutral").mean(),
        ),

        risk_share=(
            "risk_flag",
            "mean",
        ),

        total_likes=(
            "likes",
            "sum",
        ),

        total_engagement=(
            "engagement",
            "sum",
        ),
    )
    .reset_index()
)


weighted = (
    df
    .groupby("date_ny")
    .apply(
        weighted_average,
        include_groups=False,
    )
    .rename("engagement_weighted_sentiment")
    .reset_index()
)


daily = daily_basic.merge(
    weighted,
    on="date_ny",
    how="left",
)


# ============================================================
# 10. WINDOW QC TABLE
# ============================================================

window_df = window_df.sort_values(
    ["requested_date", "window_number"]
)

window_df.to_csv(
    OUTPUT_DIR / "twitter_window_qc.csv",
    index=False,
)


# ============================================================
# 11. SAVE TWEET-LEVEL DATA
# ============================================================

df = df.sort_values(
    ["date_ny", "created_at_utc"]
)

df.to_csv(
    OUTPUT_DIR / "twitter_posts_scored.csv",
    index=False,
)


# ============================================================
# 12. SAVE DAILY DATA
# ============================================================

daily = daily.sort_values("date_ny")

daily.to_csv(
    OUTPUT_DIR / "twitter_daily_sentiment.csv",
    index=False,
)


# ============================================================
# 13. PRINT FINAL QC
# ============================================================

print("\n" + "=" * 65)
print("TWITTER COLLECTION QC")
print("=" * 65)

print(
    f"Collection windows:       {len(window_df):,}"
)

print(
    f"Raw tweets:               {before:,}"
)

print(
    f"Unique English tweets:    {len(df):,}"
)

print(
    f"Duplicates removed:       {duplicates_removed:,}"
)

print(
    f"Dates represented:        {df['date_ny'].nunique()}"
)

print(
    f"Date range:               "
    f"{df['date_ny'].min()} to {df['date_ny'].max()}"
)

print(
    f"Date/window mismatches:   "
    f"{(~df['date_matches_window']).sum():,}"
)

print(
    f"Windows with next page:   "
    f"{window_df['has_next_page'].fillna(False).sum():,}"
)

print("\nSentiment distribution:")

print(
    df["sentiment_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .astype(str)
    + "%"
)

print("\nDaily sample:")
print(
    daily[
        [
            "date_ny",
            "n_posts",
            "mean_sentiment",
            "engagement_weighted_sentiment",
            "positive_share",
            "negative_share",
            "risk_share",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\nFILES CREATED:")

print(
    OUTPUT_DIR / "twitter_posts_scored.csv"
)

print(
    OUTPUT_DIR / "twitter_daily_sentiment.csv"
)

print(
    OUTPUT_DIR / "twitter_window_qc.csv"
)

print("\nDONE.")