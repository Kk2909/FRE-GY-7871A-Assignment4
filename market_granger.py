from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.stattools import grangercausalitytests


# ============================================================
# SETTINGS
# ============================================================

TWITTER_FILE = Path("twitter_outputs/twitter_posts_scored.csv")

OUTPUT_DIR = Path("market_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Equal-weight AI / AI-exposed equity basket
AI_TICKERS = [
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AMD",
    "AVGO",
]

BENCHMARK = "SPY"

ALL_TICKERS = AI_TICKERS + [BENCHMARK]

# Start before Twitter sample so Sep 8 return can be calculated
PRICE_START = "2026-09-04"

# yfinance end date is EXCLUSIVE.
# 2026-10-02 therefore includes 2026-10-01 if available.
PRICE_END = "2026-10-02"


# ============================================================
# 1. LOAD SCORED TWITTER POSTS
# ============================================================

print("\nLoading Twitter sentiment data...")

posts = pd.read_csv(TWITTER_FILE)

print(f"Tweets loaded: {len(posts):,}")


# Parse timestamp
posts["created_at_ny"] = (
    pd.to_datetime(
        posts["created_at_ny"],
        utc=True,
        errors="coerce",
    )
    .dt.tz_convert("America/New_York")
)

posts = posts.dropna(
    subset=["created_at_ny", "sentiment_score"]
).copy()


# Calendar date in New York
posts["tweet_date"] = (
    posts["created_at_ny"]
    .dt.tz_localize(None)
    .dt.normalize()
)


# Make numeric
numeric_cols = [
    "sentiment_score",
    "engagement_weight",
    "risk_flag",
    "prob_positive",
    "prob_negative",
    "prob_neutral",
]

for col in numeric_cols:
    if col in posts.columns:
        posts[col] = pd.to_numeric(
            posts[col],
            errors="coerce",
        )


print(
    "Twitter date range:",
    posts["tweet_date"].min().date(),
    "to",
    posts["tweet_date"].max().date(),
)


# ============================================================
# 2. DOWNLOAD MARKET DATA
# ============================================================

print("\nDownloading market prices...")

prices_raw = yf.download(
    ALL_TICKERS,
    start=PRICE_START,
    end=PRICE_END,
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=False,
    group_by="column",
)


if prices_raw.empty:
    raise RuntimeError(
        "yfinance returned no market data."
    )


# Handle yfinance MultiIndex format
if isinstance(prices_raw.columns, pd.MultiIndex):

    level0 = prices_raw.columns.get_level_values(0)

    if "Close" in level0:
        close = prices_raw["Close"].copy()

    else:
        # fallback in case ticker and price levels are reversed
        close = prices_raw.xs(
            "Close",
            level=1,
            axis=1,
        ).copy()

else:

    # This branch is mainly for single-ticker downloads,
    # but included for robustness.
    close = prices_raw[["Close"]].copy()


# Make sure tickers are ordered correctly
missing_tickers = [
    ticker
    for ticker in ALL_TICKERS
    if ticker not in close.columns
]

if missing_tickers:
    raise RuntimeError(
        f"Missing market data for: {missing_tickers}"
    )


close = close[ALL_TICKERS]

close.index = pd.to_datetime(
    close.index
).tz_localize(None)

close = close.sort_index()


print("\nAdjusted close prices retrieved:")
print(close.round(2).tail())


# ============================================================
# 3. CALCULATE RETURNS
# ============================================================

returns = close.pct_change(
    fill_method=None
)

market = pd.DataFrame(
    index=returns.index
)


# Individual stock returns
for ticker in AI_TICKERS:
    market[f"{ticker}_return"] = returns[ticker]


# Equal-weight AI basket
market["ai_basket_return"] = (
    returns[AI_TICKERS]
    .mean(axis=1)
)


# S&P 500 benchmark
market["spy_return"] = returns[BENCHMARK]


# AI-specific market-adjusted return
market["ai_excess_return"] = (
    market["ai_basket_return"]
    - market["spy_return"]
)


# Keep dates where return can actually be computed
market = market.dropna(
    subset=[
        "ai_basket_return",
        "spy_return",
        "ai_excess_return",
    ]
).copy()


market.index.name = "trade_date"


print("\nMarket return dates:")
print(
    market.index.min().date(),
    "to",
    market.index.max().date(),
)

print(
    f"Trading observations downloaded: {len(market)}"
)


# ============================================================
# 4. MAP EACH TWEET TO THE NEXT TRADING DAY
# ============================================================

#
# IMPORTANT:
#
# Tweet on Tuesday -> predicts Wednesday return
# Tweet on Friday  -> predicts Monday return
# Tweet on Saturday -> predicts Monday return
# Tweet on Sunday -> predicts Monday return
#
# This avoids using same-day information to explain a return
# that may already have partially occurred.
#

trade_dates = pd.DatetimeIndex(
    market.index
).normalize()


trade_array = trade_dates.values.astype(
    "datetime64[ns]"
)

tweet_array = posts[
    "tweet_date"
].values.astype(
    "datetime64[ns]"
)


# side="right" = STRICTLY next trading date
positions = np.searchsorted(
    trade_array,
    tweet_array,
    side="right",
)


posts["target_trade_date"] = pd.NaT

valid = positions < len(
    trade_dates
)


posts.loc[
    valid,
    "target_trade_date"
] = trade_dates[
    positions[valid]
].values


posts = posts.dropna(
    subset=["target_trade_date"]
).copy()


posts["target_trade_date"] = pd.to_datetime(
    posts["target_trade_date"]
)


print(
    f"\nTweets with future trading-day assignment: "
    f"{len(posts):,}"
)


# ============================================================
# 5. CREATE TRADING-DAY TWITTER SIGNAL
# ============================================================

# Engagement-weighted numerator
posts["sentiment_x_weight"] = (
    posts["sentiment_score"]
    * posts["engagement_weight"]
)


signal = (
    posts
    .groupby("target_trade_date")
    .agg(

        twitter_posts=(
            "tweet_id",
            "count",
        ),

        twitter_sentiment=(
            "sentiment_score",
            "mean",
        ),

        median_sentiment=(
            "sentiment_score",
            "median",
        ),

        positive_share=(
            "sentiment_label",
            lambda x: (
                x == "positive"
            ).mean(),
        ),

        negative_share=(
            "sentiment_label",
            lambda x: (
                x == "negative"
            ).mean(),
        ),

        neutral_share=(
            "sentiment_label",
            lambda x: (
                x == "neutral"
            ).mean(),
        ),

        risk_share=(
            "risk_flag",
            "mean",
        ),

        weighted_sentiment_num=(
            "sentiment_x_weight",
            "sum",
        ),

        weighted_sentiment_den=(
            "engagement_weight",
            "sum",
        ),
    )
    .reset_index()
)


signal["engagement_weighted_sentiment"] = (
    signal["weighted_sentiment_num"]
    / signal["weighted_sentiment_den"]
)


signal = signal.drop(
    columns=[
        "weighted_sentiment_num",
        "weighted_sentiment_den",
    ]
)


# ============================================================
# 6. MERGE SENTIMENT WITH RETURNS
# ============================================================

market_reset = (
    market
    .reset_index()
    .rename(
        columns={
            "trade_date":
            "target_trade_date"
        }
    )
)


merged = signal.merge(
    market_reset,
    on="target_trade_date",
    how="inner",
)


merged = merged.sort_values(
    "target_trade_date"
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MERGED TWITTER + MARKET DATA")
print("=" * 70)

print(
    merged[
        [
            "target_trade_date",
            "twitter_posts",
            "twitter_sentiment",
            "engagement_weighted_sentiment",
            "risk_share",
            "ai_basket_return",
            "spy_return",
            "ai_excess_return",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


print(
    f"\nMerged trading observations: "
    f"{len(merged)}"
)


# ============================================================
# 7. CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATIONS")
print("=" * 70)


corr_cols = [
    "twitter_sentiment",
    "engagement_weighted_sentiment",
    "risk_share",
    "ai_basket_return",
    "spy_return",
    "ai_excess_return",
]


correlations = (
    merged[corr_cols]
    .corr()
)


print(
    correlations.round(3)
)


correlations.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)


# ============================================================
# 8. GRANGER CAUSALITY — LAG 1
# ============================================================

#
# statsmodels expects:
#
# column 1 = variable being predicted
# column 2 = potential Granger-causal variable
#
# Therefore:
#
# [AI RETURN, SENTIMENT]
#
# tests whether sentiment Granger-causes AI returns.
#

print("\n" + "=" * 70)
print("GRANGER CAUSALITY TESTS — LAG 1")
print("=" * 70)


def run_granger(
    dataframe,
    target,
    predictor,
    description,
):

    test_data = (
        dataframe[
            [target, predictor]
        ]
        .dropna()
        .astype(float)
    )

    try:

        result = grangercausalitytests(
            test_data,
            maxlag=[1],
            addconst=True,
        )

        f_stat = result[1][0][
            "ssr_ftest"
        ][0]

        p_value = result[1][0][
            "ssr_ftest"
        ][1]

        n_obs = len(
            test_data
        )

        print(
            f"\n{description}"
        )

        print(
            f"N = {n_obs}"
        )

        print(
            f"F-stat = {f_stat:.4f}"
        )

        print(
            f"p-value = {p_value:.4f}"
        )

        return {
            "test": description,
            "target": target,
            "predictor": predictor,
            "lag": 1,
            "n_obs": n_obs,
            "f_stat": f_stat,
            "p_value": p_value,
        }

    except Exception as exc:

        print(
            f"\nCould not run "
            f"{description}: {exc}"
        )

        return {
            "test": description,
            "target": target,
            "predictor": predictor,
            "lag": 1,
            "n_obs": len(test_data),
            "f_stat": np.nan,
            "p_value": np.nan,
        }


granger_results = []


# ------------------------------------------------------------
# MAIN TEST:
# Twitter sentiment -> AI basket return
# ------------------------------------------------------------

granger_results.append(
    run_granger(
        merged,
        target="ai_basket_return",
        predictor="twitter_sentiment",
        description=(
            "Twitter sentiment -> "
            "AI basket return"
        ),
    )
)


# ------------------------------------------------------------
# IMPORTANT TEST:
# Twitter sentiment -> AI return ABOVE SPY
# ------------------------------------------------------------

granger_results.append(
    run_granger(
        merged,
        target="ai_excess_return",
        predictor="twitter_sentiment",
        description=(
            "Twitter sentiment -> "
            "AI excess return"
        ),
    )
)


# ------------------------------------------------------------
# Is this simply predicting the whole market?
# ------------------------------------------------------------

granger_results.append(
    run_granger(
        merged,
        target="spy_return",
        predictor="twitter_sentiment",
        description=(
            "Twitter sentiment -> "
            "SPY return"
        ),
    )
)


# ------------------------------------------------------------
# ROBUSTNESS:
# engagement-weighted sentiment
# ------------------------------------------------------------

granger_results.append(
    run_granger(
        merged,
        target="ai_excess_return",
        predictor=(
            "engagement_weighted_sentiment"
        ),
        description=(
            "Engagement-weighted Twitter "
            "sentiment -> AI excess return"
        ),
    )
)


# ------------------------------------------------------------
# REVERSE DIRECTION
#
# Did AI returns predict later Twitter sentiment?
# ------------------------------------------------------------

granger_results.append(
    run_granger(
        merged,
        target="twitter_sentiment",
        predictor="ai_basket_return",
        description=(
            "AI basket return -> "
            "Twitter sentiment"
        ),
    )
)


granger_df = pd.DataFrame(
    granger_results
)


granger_df.to_csv(
    OUTPUT_DIR / "granger_results.csv",
    index=False,
)


# ============================================================
# 9. PREDICTIVE REGRESSION
# ============================================================

#
# Because our Twitter signal has ALREADY been mapped
# to the following trading day:
#
# Return_t = alpha
#          + beta * SentimentSignal_t
#          + gamma * Return_(t-1)
#
# beta therefore measures the predictive relationship.
#

reg = merged.copy()


# Standardize sentiment
reg["sentiment_z"] = (
    reg["twitter_sentiment"]
    - reg["twitter_sentiment"].mean()
) / reg["twitter_sentiment"].std(
    ddof=1
)


# Lagged excess return control
reg["lag_ai_excess_return"] = (
    reg["ai_excess_return"]
    .shift(1)
)


reg_data = reg[
    [
        "ai_excess_return",
        "sentiment_z",
        "lag_ai_excess_return",
    ]
].dropna()


X = reg_data[
    [
        "sentiment_z",
        "lag_ai_excess_return",
    ]
]

X = sm.add_constant(X)

y = reg_data[
    "ai_excess_return"
]


model = sm.OLS(
    y,
    X,
).fit(
    cov_type="HC1"
)


print("\n" + "=" * 70)
print("PREDICTIVE OLS REGRESSION")
print("=" * 70)

print(model.summary())


beta = model.params[
    "sentiment_z"
]

beta_p = model.pvalues[
    "sentiment_z"
]


print(
    "\nInterpretation:"
)

print(
    f"A 1-standard-deviation increase "
    f"in Twitter sentiment is associated "
    f"with {beta * 100:.3f} percentage points "
    f"of next-trading-day AI excess return."
)

print(
    f"Sentiment coefficient p-value: "
    f"{beta_p:.4f}"
)


# Save regression coefficients
reg_results = pd.DataFrame({
    "variable": model.params.index,
    "coefficient": model.params.values,
    "std_error": model.bse.values,
    "t_stat": model.tvalues.values,
    "p_value": model.pvalues.values,
})


reg_results.to_csv(
    OUTPUT_DIR /
    "predictive_regression_results.csv",
    index=False,
)


# ============================================================
# 10. SAVE MARKET + MERGED DATA
# ============================================================

close.to_csv(
    OUTPUT_DIR /
    "adjusted_close_prices.csv"
)


market.to_csv(
    OUTPUT_DIR /
    "market_returns.csv"
)


merged.to_csv(
    OUTPUT_DIR /
    "twitter_market_merged.csv",
    index=False,
)


# ============================================================
# 11. FIGURE 1:
# STANDARDIZED SENTIMENT VS AI EXCESS RETURN
# ============================================================

plot_df = merged.copy()


plot_df["sentiment_z"] = (
    plot_df["twitter_sentiment"]
    - plot_df["twitter_sentiment"].mean()
) / plot_df["twitter_sentiment"].std(
    ddof=1
)


plot_df["ai_excess_z"] = (
    plot_df["ai_excess_return"]
    - plot_df["ai_excess_return"].mean()
) / plot_df["ai_excess_return"].std(
    ddof=1
)


plt.figure(figsize=(11, 6))

plt.plot(
    plot_df["target_trade_date"],
    plot_df["sentiment_z"],
    marker="o",
    label="Twitter sentiment",
)

plt.plot(
    plot_df["target_trade_date"],
    plot_df["ai_excess_z"],
    marker="o",
    label="AI basket excess return",
)

plt.axhline(
    0,
    linewidth=1,
)

plt.title(
    "Twitter AI Sentiment vs. "
    "Next-Trading-Day AI Excess Returns"
)

plt.xlabel(
    "Trading Date"
)

plt.ylabel(
    "Standardized Value"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "sentiment_vs_ai_excess.png",
    dpi=300,
)

plt.close()


# ============================================================
# 12. FIGURE 2:
# SENTIMENT VS NEXT-DAY EXCESS RETURN SCATTER
# ============================================================

x = merged[
    "twitter_sentiment"
].to_numpy()

y_scatter = (
    merged[
        "ai_excess_return"
    ].to_numpy()
    * 100
)


plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    x,
    y_scatter,
)


if len(x) >= 2:

    slope, intercept = np.polyfit(
        x,
        y_scatter,
        1,
    )

    x_line = np.linspace(
        x.min(),
        x.max(),
        100,
    )

    y_line = (
        intercept
        + slope * x_line
    )

    plt.plot(
        x_line,
        y_line,
    )


plt.axhline(
    0,
    linewidth=1,
)

plt.axvline(
    0,
    linewidth=1,
)

plt.title(
    "Twitter Sentiment and "
    "Next-Trading-Day AI Excess Return"
)

plt.xlabel(
    "Twitter Sentiment Score"
)

plt.ylabel(
    "AI Excess Return (%)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "sentiment_return_scatter.png",
    dpi=300,
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    OUTPUT_DIR /
    "adjusted_close_prices.csv"
)

print(
    OUTPUT_DIR /
    "market_returns.csv"
)

print(
    OUTPUT_DIR /
    "twitter_market_merged.csv"
)

print(
    OUTPUT_DIR /
    "correlation_matrix.csv"
)

print(
    OUTPUT_DIR /
    "granger_results.csv"
)

print(
    OUTPUT_DIR /
    "predictive_regression_results.csv"
)

print(
    OUTPUT_DIR /
    "sentiment_vs_ai_excess.png"
)

print(
    OUTPUT_DIR /
    "sentiment_return_scatter.png"
)

print("\nDONE.")