from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

TWITTER_POSTS = Path(
    "twitter_outputs/twitter_posts_scored.csv"
)

TWITTER_DAILY = Path(
    "twitter_outputs/twitter_daily_sentiment.csv"
)

NEWS_SCORED = Path(
    "news_outputs/google_news_ai_scored.csv"
)

NEWS_DAILY = Path(
    "news_outputs/google_news_daily_sentiment.csv"
)

DAILY_COMPARISON = Path(
    "comparison_outputs/news_vs_twitter_daily.csv"
)

COMPARISON_TESTS = Path(
    "comparison_outputs/news_vs_twitter_tests.csv"
)

GRANGER_RESULTS = Path(
    "market_outputs/formal_granger_results.csv"
)

PREDICTIVE_RESULTS = Path(
    "market_outputs/aligned_predictive_results.csv"
)

TWITTER_MARKET = Path(
    "market_outputs/twitter_market_merged.csv"
)


OUTPUT_DIR = Path("final_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

twitter_posts = pd.read_csv(
    TWITTER_POSTS
)

twitter_daily = pd.read_csv(
    TWITTER_DAILY
)

news = pd.read_csv(
    NEWS_SCORED
)

news_daily = pd.read_csv(
    NEWS_DAILY
)

comparison = pd.read_csv(
    DAILY_COMPARISON
)

comparison_tests = pd.read_csv(
    COMPARISON_TESTS
)

granger = pd.read_csv(
    GRANGER_RESULTS
)

predictive = pd.read_csv(
    PREDICTIVE_RESULTS
)

market = pd.read_csv(
    TWITTER_MARKET
)


comparison["date_ny"] = pd.to_datetime(
    comparison["date_ny"]
)

market["target_trade_date"] = pd.to_datetime(
    market["target_trade_date"]
)


# ============================================================
# FIGURE 1
# NEWS VS TWITTER SENTIMENT DISTRIBUTION
# ============================================================

labels = [
    "Negative",
    "Neutral",
    "Positive",
]


twitter_distribution = [
    (
        twitter_posts["sentiment_label"]
        == "negative"
    ).mean() * 100,

    (
        twitter_posts["sentiment_label"]
        == "neutral"
    ).mean() * 100,

    (
        twitter_posts["sentiment_label"]
        == "positive"
    ).mean() * 100,
]


news_distribution = [
    (
        news["sentiment_label"]
        == "negative"
    ).mean() * 100,

    (
        news["sentiment_label"]
        == "neutral"
    ).mean() * 100,

    (
        news["sentiment_label"]
        == "positive"
    ).mean() * 100,
]


x = np.arange(
    len(labels)
)

width = 0.36


fig, ax = plt.subplots(
    figsize=(9, 6)
)


bars1 = ax.bar(
    x - width / 2,
    twitter_distribution,
    width,
    label="X / Twitter",
)


bars2 = ax.bar(
    x + width / 2,
    news_distribution,
    width,
    label="Google News",
)


ax.set_ylabel(
    "Share of observations (%)"
)

ax.set_title(
    "AI Sentiment Distribution: "
    "News vs. X"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels
)

ax.legend()


for bars in [
    bars1,
    bars2,
]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            height + 1,

            f"{height:.1f}%",

            ha="center",
            va="bottom",
            fontsize=9,
        )


ax.set_ylim(
    0,
    max(
        twitter_distribution
        + news_distribution
    ) + 10
)

fig.tight_layout()


fig.savefig(
    OUTPUT_DIR /
    "figure1_sentiment_distribution.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# FIGURE 2
# DAILY MEAN SENTIMENT
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)


ax.plot(
    comparison["date_ny"],
    comparison[
        "twitter_mean_sentiment"
    ],
    marker="o",
    label="X / Twitter",
)


ax.plot(
    comparison["date_ny"],
    comparison[
        "news_mean_sentiment"
    ],
    marker="o",
    label="Google News",
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_title(
    "Daily AI Sentiment: "
    "News vs. X"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Mean sentiment score"
)

ax.legend()


plt.xticks(
    rotation=45
)


fig.tight_layout()


fig.savefig(
    OUTPUT_DIR /
    "figure2_daily_sentiment.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# FIGURE 3
# RISK-LANGUAGE SHARE
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)


ax.plot(
    comparison["date_ny"],
    comparison[
        "twitter_risk_share"
    ] * 100,
    marker="o",
    label="X / Twitter",
)


ax.plot(
    comparison["date_ny"],
    comparison[
        "news_risk_share"
    ] * 100,
    marker="o",
    label="Google News",
)


ax.set_title(
    "Daily AI Risk-Language Share: "
    "News vs. X"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Risk-language share (%)"
)

ax.legend()


plt.xticks(
    rotation=45
)


fig.tight_layout()


fig.savefig(
    OUTPUT_DIR /
    "figure3_daily_risk_share.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# FIGURE 4
# TWITTER SENTIMENT VS NEXT AI EXCESS RETURN
# ============================================================

x_sentiment = market[
    "twitter_sentiment"
].to_numpy()


y_return = (
    market[
        "ai_excess_return"
    ].to_numpy()
    * 100
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.scatter(
    x_sentiment,
    y_return,
)


# Regression line
if len(x_sentiment) > 1:

    slope, intercept = np.polyfit(
        x_sentiment,
        y_return,
        1,
    )

    x_line = np.linspace(
        x_sentiment.min(),
        x_sentiment.max(),
        100,
    )

    y_line = (
        intercept
        + slope * x_line
    )

    ax.plot(
        x_line,
        y_line,
    )


ax.axhline(
    0,
    linewidth=1,
)

ax.axvline(
    0,
    linewidth=1,
)


ax.set_title(
    "Prior X Sentiment vs. "
    "Next-Trading-Day AI Excess Return"
)

ax.set_xlabel(
    "Prior X sentiment score"
)

ax.set_ylabel(
    "AI basket excess return (%)"
)


fig.tight_layout()


fig.savefig(
    OUTPUT_DIR /
    "figure4_sentiment_vs_ai_return.png",
    dpi=300,
)

plt.close(fig)


# ============================================================
# TABLE 1
# DATA SAMPLE SUMMARY
# ============================================================

sample_summary = pd.DataFrame({

    "Dataset": [
        "X / Twitter",
        "Google News",
        "Market",
    ],

    "Observations": [
        len(twitter_posts),
        len(news),
        len(market),
    ],

    "Dates": [
        twitter_daily["date_ny"].nunique(),
        news_daily["date_ny"].nunique(),
        market[
            "target_trade_date"
        ].nunique(),
    ],

    "Description": [
        "English AI-related posts",
        "Unique AI-explicit headlines",
        "Trading-day observations",
    ],
})


sample_summary.to_csv(
    OUTPUT_DIR /
    "table1_sample_summary.csv",
    index=False,
)


# ============================================================
# TABLE 2
# OVERALL SENTIMENT SUMMARY
# ============================================================

sentiment_summary = pd.DataFrame({

    "Metric": [
        "Mean sentiment",
        "Negative share",
        "Neutral share",
        "Positive share",
        "Risk-language share",
    ],

    "X_Twitter": [

        twitter_posts[
            "sentiment_score"
        ].mean(),

        (
            twitter_posts[
                "sentiment_label"
            ] == "negative"
        ).mean(),

        (
            twitter_posts[
                "sentiment_label"
            ] == "neutral"
        ).mean(),

        (
            twitter_posts[
                "sentiment_label"
            ] == "positive"
        ).mean(),

        twitter_posts[
            "risk_flag"
        ].mean(),
    ],

    "Google_News": [

        news[
            "sentiment_score"
        ].mean(),

        (
            news[
                "sentiment_label"
            ] == "negative"
        ).mean(),

        (
            news[
                "sentiment_label"
            ] == "neutral"
        ).mean(),

        (
            news[
                "sentiment_label"
            ] == "positive"
        ).mean(),

        news[
            "risk_flag"
        ].mean(),
    ],
})


sentiment_summary[
    "News_minus_Twitter"
] = (
    sentiment_summary[
        "Google_News"
    ]
    -
    sentiment_summary[
        "X_Twitter"
    ]
)


sentiment_summary.to_csv(
    OUTPUT_DIR /
    "table2_sentiment_summary.csv",
    index=False,
)


# ============================================================
# TABLE 3
# NEWS VS TWITTER STATISTICAL TESTS
# ============================================================

tests_clean = (
    comparison_tests.copy()
)


for column in [
    "news_mean",
    "twitter_mean",
    "news_minus_twitter",
    "paired_t_p",
    "wilcoxon_p",
]:

    if column in tests_clean.columns:

        tests_clean[column] = (
            pd.to_numeric(
                tests_clean[column],
                errors="coerce",
            )
            .round(4)
        )


tests_clean.to_csv(
    OUTPUT_DIR /
    "table3_news_vs_twitter_tests.csv",
    index=False,
)


# ============================================================
# TABLE 4
# GRANGER RESULTS
# ============================================================

granger_clean = granger.copy()


for column in [
    "f_stat",
    "p_value",
]:

    if column in granger_clean.columns:

        granger_clean[column] = (
            pd.to_numeric(
                granger_clean[column],
                errors="coerce",
            )
            .round(4)
        )


granger_clean.to_csv(
    OUTPUT_DIR /
    "table4_granger_results.csv",
    index=False,
)


# ============================================================
# TABLE 5
# PREDICTIVE REGRESSION RESULTS
# ============================================================

predictive_clean = (
    predictive.copy()
)


for column in [
    "f_stat",
    "f_p_value",
    "standardized_beta",
    "beta_percentage_points",
    "hc1_p_value",
    "r_squared",
    "adj_r_squared",
]:

    if column in predictive_clean.columns:

        predictive_clean[column] = (
            pd.to_numeric(
                predictive_clean[column],
                errors="coerce",
            )
            .round(4)
        )


predictive_clean.to_csv(
    OUTPUT_DIR /
    "table5_predictive_results.csv",
    index=False,
)


# ============================================================
# TABLE 6
# MAIN RESULTS SUMMARY
# ============================================================

main_results = pd.DataFrame({

    "Research Question": [

        "Does X sentiment predict "
        "AI basket returns?",

        "Does X sentiment predict "
        "AI excess returns?",

        "Do AI returns predict "
        "subsequent X sentiment?",

        "Is news mean sentiment "
        "more negative than X?",

        "Does news contain a larger "
        "negative share than X?",

        "Does news contain more "
        "risk language than X?",
    ],

    "Finding": [

        "No statistically significant "
        "predictive evidence",

        "No statistically significant "
        "predictive evidence",

        "No statistically significant "
        "reverse relationship",

        "No significant difference "
        "in average sentiment",

        "No; X has substantially "
        "more negative content",

        "No; X has significantly "
        "more risk-oriented language",
    ],
})


main_results.to_csv(
    OUTPUT_DIR /
    "table6_main_findings.csv",
    index=False,
)


# ============================================================
# PRINT IMPORTANT RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL ANALYSIS SUMMARY"
)

print(
    "=" * 70
)


print(
    "\nTwitter sentiment distribution:"
)

print(
    f"Negative: "
    f"{twitter_distribution[0]:.2f}%"
)

print(
    f"Neutral:  "
    f"{twitter_distribution[1]:.2f}%"
)

print(
    f"Positive: "
    f"{twitter_distribution[2]:.2f}%"
)


print(
    "\nNews sentiment distribution:"
)

print(
    f"Negative: "
    f"{news_distribution[0]:.2f}%"
)

print(
    f"Neutral:  "
    f"{news_distribution[1]:.2f}%"
)

print(
    f"Positive: "
    f"{news_distribution[2]:.2f}%"
)


print(
    "\nOverall mean sentiment:"
)

print(
    "Twitter:",
    round(
        twitter_posts[
            "sentiment_score"
        ].mean(),
        4,
    ),
)

print(
    "News:",
    round(
        news[
            "sentiment_score"
        ].mean(),
        4,
    ),
)


print(
    "\nOverall risk-language share:"
)

print(
    "Twitter:",
    round(
        twitter_posts[
            "risk_flag"
        ].mean(),
        4,
    ),
)

print(
    "News:",
    round(
        news[
            "risk_flag"
        ].mean(),
        4,
    ),
)


print(
    "\n"
    + "=" * 70
)

print(
    "FILES CREATED"
)

print(
    "=" * 70
)


for file in sorted(
    OUTPUT_DIR.glob("*")
):

    print(file)


print("\nDONE.")