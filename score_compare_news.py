from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from scipy.stats import (
    ttest_rel,
    ttest_ind,
    wilcoxon,
    chi2_contingency,
)


# ============================================================
# SETTINGS
# ============================================================

NEWS_FILE = Path(
    "news_outputs/google_news_ai_raw.csv"
)

TWITTER_DAILY_FILE = Path(
    "twitter_outputs/twitter_daily_sentiment.csv"
)

TWITTER_POST_FILE = Path(
    "twitter_outputs/twitter_posts_scored.csv"
)

NEWS_OUTPUT_DIR = Path("news_outputs")
COMPARE_OUTPUT_DIR = Path("comparison_outputs")

NEWS_OUTPUT_DIR.mkdir(exist_ok=True)
COMPARE_OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = (
    "cardiffnlp/"
    "twitter-roberta-base-sentiment-latest"
)

BATCH_SIZE = 32

RANDOM_SEED = 7871


# ============================================================
# 1. LOAD NEWS
# ============================================================

print("\nLoading Google News dataset...")

news = pd.read_csv(
    NEWS_FILE
)

print(
    f"Raw news rows: {len(news):,}"
)


# ============================================================
# 2. PRIMARY RELEVANCE FILTER
# ============================================================

#
# IMPORTANT:
#
# We are scoring the HEADLINE.
#
# Therefore, for the primary comparison, require AI to
# actually appear in the headline.
#

news["ai_in_title"] = pd.to_numeric(
    news["ai_in_title"],
    errors="coerce",
).fillna(0)


news = news[
    news["ai_in_title"] == 1
].copy()


print(
    f"AI-explicit headlines: {len(news):,}"
)


# ============================================================
# 3. CLEAN HEADLINES
# ============================================================

news["headline"] = (
    news["headline"]
    .fillna("")
    .astype(str)
    .str.strip()
)


news = news[
    news["headline"].ne("")
].copy()


news["headline_normalized"] = (
    news["headline"]
    .str.lower()
    .str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    .str.strip()
)


# ============================================================
# 4. GLOBAL EXACT-HEADLINE DEDUPLICATION
# ============================================================

#
# Same Reuters/AP/etc. story can appear through several
# different outlets.
#
# We don't want identical syndicated headlines receiving
# several votes merely because multiple outlets carried them.
#

before_global_dedup = len(news)

news = news.drop_duplicates(
    subset=["headline_normalized"],
    keep="first",
).copy()


global_duplicates_removed = (
    before_global_dedup - len(news)
)


print(
    f"Cross-source identical headlines removed: "
    f"{global_duplicates_removed:,}"
)

print(
    f"Primary unique headline sample: "
    f"{len(news):,}"
)


# ============================================================
# 5. DATE CLEANING
# ============================================================

news["date_ny"] = (
    pd.to_datetime(
        news["date_ny"],
        errors="coerce",
    )
    .dt.strftime("%Y-%m-%d")
)


news = news.dropna(
    subset=["date_ny"]
).copy()


# ============================================================
# 6. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(text)

    words = []

    for token in text.split():

        if (
            token.startswith("@")
            and len(token) > 1
        ):
            token = "@user"

        elif token.startswith("http"):
            token = "http"

        words.append(token)

    text = " ".join(words)

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


news["text_model"] = (
    news["headline"]
    .apply(preprocess_text)
)


# ============================================================
# 7. LOAD SAME SENTIMENT MODEL USED FOR TWITTER
# ============================================================

print("\nLoading sentiment model...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device,
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME
    )
)

model.to(device)

model.eval()


# ============================================================
# 8. SCORE HEADLINES
# ============================================================

texts = news[
    "text_model"
].tolist()


all_probs = []


print("\nScoring news headlines...")


for start in tqdm(
    range(
        0,
        len(texts),
        BATCH_SIZE,
    ),
    desc="News sentiment batches",
):

    batch = texts[
        start:
        start + BATCH_SIZE
    ]


    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,

        # Headlines are short, so no reason to use 512
        max_length=128,

        return_tensors="pt",
    )


    encoded = {
        key: value.to(device)
        for key, value
        in encoded.items()
    }


    with torch.no_grad():

        logits = model(
            **encoded
        ).logits


        probs = torch.softmax(
            logits,
            dim=1,
        ).cpu().numpy()


    all_probs.append(
        probs
    )


probs = np.vstack(
    all_probs
)


# Same Cardiff ordering used for Twitter:
#
# 0 negative
# 1 neutral
# 2 positive

news["prob_negative"] = (
    probs[:, 0]
)

news["prob_neutral"] = (
    probs[:, 1]
)

news["prob_positive"] = (
    probs[:, 2]
)


news["sentiment_score"] = (
    news["prob_positive"]
    - news["prob_negative"]
)


labels = np.array([
    "negative",
    "neutral",
    "positive",
])


news["sentiment_label"] = (
    labels[
        np.argmax(
            probs,
            axis=1,
        )
    ]
)


# ============================================================
# 9. SAME RISK DICTIONARY USED FOR TWITTER
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


news["risk_flag"] = (
    news["headline"]
    .str.contains(
        risk_pattern,
        na=False,
    )
    .astype(int)
)


# ============================================================
# 10. DAILY NEWS SENTIMENT
# ============================================================

daily_news = (
    news
    .groupby("date_ny")
    .agg(

        n_news=(
            "headline",
            "count",
        ),

        news_mean_sentiment=(
            "sentiment_score",
            "mean",
        ),

        news_median_sentiment=(
            "sentiment_score",
            "median",
        ),

        news_positive_share=(
            "sentiment_label",
            lambda x:
                (x == "positive").mean(),
        ),

        news_negative_share=(
            "sentiment_label",
            lambda x:
                (x == "negative").mean(),
        ),

        news_neutral_share=(
            "sentiment_label",
            lambda x:
                (x == "neutral").mean(),
        ),

        news_risk_share=(
            "risk_flag",
            "mean",
        ),

        n_sources=(
            "source",
            "nunique",
        ),
    )
    .reset_index()
)


# ============================================================
# 11. SAVE SCORED NEWS
# ============================================================

news.to_csv(
    NEWS_OUTPUT_DIR /
    "google_news_ai_scored.csv",
    index=False,
)


daily_news.to_csv(
    NEWS_OUTPUT_DIR /
    "google_news_daily_sentiment.csv",
    index=False,
)


# ============================================================
# 12. CREATE MATCHED 40-HEADLINE/DAY NEWS SAMPLE
# ============================================================

#
# Twitter contains exactly 40 posts/day.
#
# This provides a useful robustness comparison:
#
# 920 Twitter posts
# vs
# up to 920 news headlines
#
# This is NOT the primary analysis.
# The daily aggregated comparison below is primary.
#

matched_parts = []


for date, group in news.groupby(
    "date_ny"
):

    n = min(
        40,
        len(group),
    )

    sampled = group.sample(
        n=n,
        random_state=RANDOM_SEED,
    )

    matched_parts.append(
        sampled
    )


matched_news = pd.concat(
    matched_parts,
    ignore_index=True,
)


matched_news.to_csv(
    NEWS_OUTPUT_DIR /
    "google_news_matched_40_per_day.csv",
    index=False,
)


print(
    f"\nMatched news sample: "
    f"{len(matched_news):,}"
)


# ============================================================
# 13. LOAD TWITTER DAILY DATA
# ============================================================

twitter_daily = pd.read_csv(
    TWITTER_DAILY_FILE
)


twitter_daily = twitter_daily.rename(
    columns={
        "mean_sentiment":
            "twitter_mean_sentiment",

        "positive_share":
            "twitter_positive_share",

        "negative_share":
            "twitter_negative_share",

        "neutral_share":
            "twitter_neutral_share",

        "risk_share":
            "twitter_risk_share",

        "n_posts":
            "n_twitter",
    }
)


twitter_daily["date_ny"] = (
    pd.to_datetime(
        twitter_daily["date_ny"]
    )
    .dt.strftime("%Y-%m-%d")
)


# ============================================================
# 14. MERGE DAILY NEWS + TWITTER
# ============================================================

comparison = twitter_daily.merge(
    daily_news,
    on="date_ny",
    how="inner",
)


comparison = comparison.sort_values(
    "date_ny"
).reset_index(
    drop=True
)


# Differences:
# positive = news is MORE of that characteristic

comparison[
    "sentiment_difference"
] = (
    comparison[
        "news_mean_sentiment"
    ]
    -
    comparison[
        "twitter_mean_sentiment"
    ]
)


comparison[
    "negative_share_difference"
] = (
    comparison[
        "news_negative_share"
    ]
    -
    comparison[
        "twitter_negative_share"
    ]
)


comparison[
    "risk_share_difference"
] = (
    comparison[
        "news_risk_share"
    ]
    -
    comparison[
        "twitter_risk_share"
    ]
)


comparison.to_csv(
    COMPARE_OUTPUT_DIR /
    "news_vs_twitter_daily.csv",
    index=False,
)


# ============================================================
# 15. PAIRED DAILY STATISTICAL TESTS
# ============================================================

print("\n" + "=" * 75)
print("DAILY NEWS VS TWITTER COMPARISON")
print("=" * 75)


test_results = []


def paired_test(
    news_col,
    twitter_col,
    metric_name,
):

    temp = comparison[
        [
            news_col,
            twitter_col,
        ]
    ].dropna()


    news_values = (
        temp[news_col]
        .astype(float)
        .to_numpy()
    )

    twitter_values = (
        temp[twitter_col]
        .astype(float)
        .to_numpy()
    )


    difference = (
        news_values
        - twitter_values
    )


    # Paired t-test
    t_result = ttest_rel(
        news_values,
        twitter_values,
    )


    # Non-parametric robustness test
    try:

        w_result = wilcoxon(
            difference
        )

        wilcoxon_stat = (
            w_result.statistic
        )

        wilcoxon_p = (
            w_result.pvalue
        )

    except Exception:

        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan


    print(
        f"\n{metric_name}"
    )

    print(
        f"Mean news:     "
        f"{news_values.mean():.4f}"
    )

    print(
        f"Mean Twitter:  "
        f"{twitter_values.mean():.4f}"
    )

    print(
        f"Difference:    "
        f"{difference.mean():.4f}"
    )

    print(
        f"Paired t-test p: "
        f"{t_result.pvalue:.4f}"
    )

    print(
        f"Wilcoxon p:      "
        f"{wilcoxon_p:.4f}"
    )


    test_results.append({

        "metric":
            metric_name,

        "n_days":
            len(temp),

        "news_mean":
            news_values.mean(),

        "twitter_mean":
            twitter_values.mean(),

        "news_minus_twitter":
            difference.mean(),

        "paired_t_stat":
            t_result.statistic,

        "paired_t_p":
            t_result.pvalue,

        "wilcoxon_stat":
            wilcoxon_stat,

        "wilcoxon_p":
            wilcoxon_p,
    })


paired_test(
    "news_mean_sentiment",
    "twitter_mean_sentiment",
    "Mean sentiment",
)


paired_test(
    "news_negative_share",
    "twitter_negative_share",
    "Negative share",
)


paired_test(
    "news_positive_share",
    "twitter_positive_share",
    "Positive share",
)


paired_test(
    "news_risk_share",
    "twitter_risk_share",
    "Risk-language share",
)


# ============================================================
# 16. MATCHED ARTICLE-LEVEL ROBUSTNESS
# ============================================================

twitter_posts = pd.read_csv(
    TWITTER_POST_FILE
)


twitter_scores = (
    pd.to_numeric(
        twitter_posts[
            "sentiment_score"
        ],
        errors="coerce",
    )
    .dropna()
)


news_scores = (
    matched_news[
        "sentiment_score"
    ]
    .dropna()
)


welch = ttest_ind(
    news_scores,
    twitter_scores,
    equal_var=False,
)


print("\n" + "=" * 75)
print("MATCHED 40/DAY ROBUSTNESS CHECK")
print("=" * 75)


print(
    f"News headlines: "
    f"{len(news_scores):,}"
)

print(
    f"Twitter posts:   "
    f"{len(twitter_scores):,}"
)

print(
    f"News sentiment mean: "
    f"{news_scores.mean():.4f}"
)

print(
    f"Twitter sentiment mean: "
    f"{twitter_scores.mean():.4f}"
)

print(
    f"Welch t-test p-value: "
    f"{welch.pvalue:.4f}"
)


# ============================================================
# 17. SENTIMENT-LABEL CHI-SQUARE
# ============================================================

news_counts = (
    matched_news[
        "sentiment_label"
    ]
    .value_counts()
    .reindex(
        [
            "negative",
            "neutral",
            "positive",
        ],
        fill_value=0,
    )
)


twitter_counts = (
    twitter_posts[
        "sentiment_label"
    ]
    .value_counts()
    .reindex(
        [
            "negative",
            "neutral",
            "positive",
        ],
        fill_value=0,
    )
)


table = np.array([
    news_counts.values,
    twitter_counts.values,
])


chi2, chi_p, dof, expected = (
    chi2_contingency(
        table
    )
)


print("\nSentiment distribution:")

print(
    "\nNews:"
)

print(
    (
        news_counts
        / news_counts.sum()
        * 100
    ).round(2)
)


print(
    "\nTwitter:"
)

print(
    (
        twitter_counts
        / twitter_counts.sum()
        * 100
    ).round(2)
)


print(
    f"\nChi-square statistic: "
    f"{chi2:.4f}"
)

print(
    f"Chi-square p-value: "
    f"{chi_p:.4f}"
)


# ============================================================
# 18. SAVE TEST RESULTS
# ============================================================

tests_df = pd.DataFrame(
    test_results
)


tests_df.to_csv(
    COMPARE_OUTPUT_DIR /
    "news_vs_twitter_tests.csv",
    index=False,
)


# ============================================================
# 19. NEWS QC
# ============================================================

print("\n" + "=" * 75)
print("NEWS SENTIMENT QC")
print("=" * 75)


print(
    f"Primary unique AI headlines: "
    f"{len(news):,}"
)


print(
    f"Dates represented: "
    f"{news['date_ny'].nunique()}"
)


print(
    "\nOverall news sentiment distribution:"
)


print(
    news[
        "sentiment_label"
    ]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
    .astype(str)
    + "%"
)


print(
    "\nOverall mean sentiment:"
)

print(
    round(
        news[
            "sentiment_score"
        ].mean(),
        4,
    )
)


print(
    "\nOverall risk-language share:"
)

print(
    round(
        news[
            "risk_flag"
        ].mean(),
        4,
    )
)


# ============================================================
# 20. DAILY OUTPUT
# ============================================================

print("\nDaily comparison:")


print(
    comparison[
        [
            "date_ny",

            "n_twitter",
            "n_news",

            "twitter_mean_sentiment",
            "news_mean_sentiment",

            "twitter_negative_share",
            "news_negative_share",

            "twitter_risk_share",
            "news_risk_share",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL FILE LIST
# ============================================================

print("\n" + "=" * 75)
print("FILES CREATED")
print("=" * 75)


print(
    NEWS_OUTPUT_DIR /
    "google_news_ai_scored.csv"
)

print(
    NEWS_OUTPUT_DIR /
    "google_news_daily_sentiment.csv"
)

print(
    NEWS_OUTPUT_DIR /
    "google_news_matched_40_per_day.csv"
)

print(
    COMPARE_OUTPUT_DIR /
    "news_vs_twitter_daily.csv"
)

print(
    COMPARE_OUTPUT_DIR /
    "news_vs_twitter_tests.csv"
)

print("\nDONE.")