from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.tsa.stattools import grangercausalitytests


# ============================================================
# FILES
# ============================================================

DAILY_SENTIMENT_FILE = Path(
    "twitter_outputs/twitter_daily_sentiment.csv"
)

MARKET_FILE = Path(
    "market_outputs/market_returns.csv"
)

ALIGNED_FILE = Path(
    "market_outputs/twitter_market_merged.csv"
)

OUTPUT_DIR = Path("market_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

daily = pd.read_csv(
    DAILY_SENTIMENT_FILE
)

market = pd.read_csv(
    MARKET_FILE
)

aligned = pd.read_csv(
    ALIGNED_FILE
)


# ============================================================
# 2. CLEAN DATES
# ============================================================

daily["date"] = pd.to_datetime(
    daily["date_ny"]
).dt.normalize()


# market_returns.csv should already contain trade_date
if "trade_date" not in market.columns:

    first_col = market.columns[0]

    market = market.rename(
        columns={
            first_col: "trade_date"
        }
    )


market["trade_date"] = pd.to_datetime(
    market["trade_date"]
).dt.normalize()


aligned["target_trade_date"] = pd.to_datetime(
    aligned["target_trade_date"]
).dt.normalize()


# ============================================================
# 3. FORMAL GRANGER DATASET
# ============================================================

#
# This is the conventional Granger setup.
#
# Sentiment observed on trading day t
# is allowed to predict return on trading day t+1.
#
# Weekend observations cannot be represented naturally
# in a standard trading-day Granger series, so weekend
# sentiment is excluded from THIS test.
#
# We retain weekend information separately in the aligned
# predictive regression below.
#

formal = daily.merge(
    market,
    left_on="date",
    right_on="trade_date",
    how="inner",
)


formal = formal.sort_values(
    "trade_date"
).reset_index(drop=True)


print("\n" + "=" * 75)
print("FORMAL TRADING-DAY DATASET")
print("=" * 75)

print(
    formal[
        [
            "trade_date",
            "mean_sentiment",
            "engagement_weighted_sentiment",
            "ai_basket_return",
            "spy_return",
            "ai_excess_return",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print(
    f"\nTrading-day observations: {len(formal)}"
)


# ============================================================
# 4. FORMAL GRANGER TEST
# ============================================================

def formal_granger(
    dataframe,
    target,
    predictor,
    description,
):

    d = (
        dataframe[
            [target, predictor]
        ]
        .dropna()
        .astype(float)
    )

    result = grangercausalitytests(
        d,
        maxlag=[1],
        verbose=False,
    )

    test = result[1][0]["ssr_ftest"]

    f_stat = test[0]
    p_value = test[1]

    print(
        f"\n{description}"
    )

    print(
        f"N = {len(d)}"
    )

    print(
        f"F-stat = {f_stat:.4f}"
    )

    print(
        f"p-value = {p_value:.4f}"
    )

    return {
        "test": description,
        "direction": (
            f"{predictor} -> {target}"
        ),
        "lag": 1,
        "n": len(d),
        "f_stat": f_stat,
        "p_value": p_value,
    }


print("\n" + "=" * 75)
print("FORMAL GRANGER CAUSALITY — LAG 1")
print("=" * 75)


formal_results = []


formal_results.append(
    formal_granger(
        formal,
        target="ai_basket_return",
        predictor="mean_sentiment",
        description=(
            "Twitter sentiment -> "
            "AI basket return"
        ),
    )
)


formal_results.append(
    formal_granger(
        formal,
        target="ai_excess_return",
        predictor="mean_sentiment",
        description=(
            "Twitter sentiment -> "
            "AI excess return"
        ),
    )
)


formal_results.append(
    formal_granger(
        formal,
        target="spy_return",
        predictor="mean_sentiment",
        description=(
            "Twitter sentiment -> "
            "SPY return"
        ),
    )
)


formal_results.append(
    formal_granger(
        formal,
        target="ai_excess_return",
        predictor=(
            "engagement_weighted_sentiment"
        ),
        description=(
            "Engagement-weighted sentiment -> "
            "AI excess return"
        ),
    )
)


# Reverse direction
formal_results.append(
    formal_granger(
        formal,
        target="mean_sentiment",
        predictor="ai_basket_return",
        description=(
            "AI basket return -> "
            "Twitter sentiment"
        ),
    )
)


formal_results_df = pd.DataFrame(
    formal_results
)


formal_results_df.to_csv(
    OUTPUT_DIR /
    "formal_granger_results.csv",
    index=False,
)


# ============================================================
# 5. ALIGNED NEXT-TRADING-DAY PREDICTIVE TEST
# ============================================================

#
# IMPORTANT:
#
# In twitter_market_merged.csv:
#
# twitter_sentiment for Sep 9
# already consists of tweets from Sep 8.
#
# Therefore DO NOT shift sentiment again.
#
#
# Regression:
#
# Return_t =
# alpha
# + phi Return_(t-1)
# + beta Sentiment_before_t
# + error_t
#
#
# This directly tests whether sentiment available BEFORE
# trading day t contains incremental predictive information.
#

def aligned_predictive_test(
    dataframe,
    target,
    predictor,
    description,
):

    d = dataframe[
        [
            "target_trade_date",
            target,
            predictor,
        ]
    ].copy()


    d["lag_return"] = (
        d[target].shift(1)
    )


    d = d.dropna().copy()


    # Standardize predictor
    d["predictor_z"] = (
        d[predictor]
        - d[predictor].mean()
    ) / d[predictor].std(ddof=1)


    # -------------------------
    # Restricted model
    # -------------------------

    X_restricted = sm.add_constant(
        d[
            [
                "lag_return"
            ]
        ]
    )


    restricted = sm.OLS(
        d[target],
        X_restricted,
    ).fit()


    # -------------------------
    # Unrestricted model
    # -------------------------

    X_unrestricted = sm.add_constant(
        d[
            [
                "lag_return",
                "predictor_z",
            ]
        ]
    )


    unrestricted = sm.OLS(
        d[target],
        X_unrestricted,
    ).fit()


    # Standard nested-model F test
    f_stat, f_pvalue, df_diff = (
        unrestricted.compare_f_test(
            restricted
        )
    )


    # HC1 robust version for coefficient inference
    robust = sm.OLS(
        d[target],
        X_unrestricted,
    ).fit(
        cov_type="HC1"
    )


    beta = robust.params[
        "predictor_z"
    ]

    robust_p = robust.pvalues[
        "predictor_z"
    ]


    print(
        f"\n{description}"
    )

    print(
        f"N = {len(d)}"
    )

    print(
        f"Incremental F-stat = "
        f"{f_stat:.4f}"
    )

    print(
        f"Incremental F-test p = "
        f"{f_pvalue:.4f}"
    )

    print(
        f"Standardized beta = "
        f"{beta:.6f}"
    )

    print(
        f"Beta in percentage points = "
        f"{beta * 100:.3f}"
    )

    print(
        f"HC1 robust p-value = "
        f"{robust_p:.4f}"
    )


    return {
        "test": description,
        "target": target,
        "predictor": predictor,
        "n": len(d),
        "f_stat": f_stat,
        "f_p_value": f_pvalue,
        "standardized_beta": beta,
        "beta_percentage_points": (
            beta * 100
        ),
        "hc1_p_value": robust_p,
        "r_squared": robust.rsquared,
        "adj_r_squared": (
            robust.rsquared_adj
        ),
    }


print("\n" + "=" * 75)
print("ALIGNED NEXT-TRADING-DAY PREDICTIVE TEST")
print("=" * 75)


aligned_results = []


aligned_results.append(
    aligned_predictive_test(
        aligned,
        target="ai_basket_return",
        predictor="twitter_sentiment",
        description=(
            "Prior Twitter sentiment -> "
            "next AI basket return"
        ),
    )
)


aligned_results.append(
    aligned_predictive_test(
        aligned,
        target="ai_excess_return",
        predictor="twitter_sentiment",
        description=(
            "Prior Twitter sentiment -> "
            "next AI excess return"
        ),
    )
)


aligned_results.append(
    aligned_predictive_test(
        aligned,
        target="spy_return",
        predictor="twitter_sentiment",
        description=(
            "Prior Twitter sentiment -> "
            "next SPY return"
        ),
    )
)


aligned_results.append(
    aligned_predictive_test(
        aligned,
        target="ai_excess_return",
        predictor=(
            "engagement_weighted_sentiment"
        ),
        description=(
            "Prior engagement-weighted "
            "sentiment -> next AI excess return"
        ),
    )
)


aligned_results_df = pd.DataFrame(
    aligned_results
)


aligned_results_df.to_csv(
    OUTPUT_DIR /
    "aligned_predictive_results.csv",
    index=False,
)


# ============================================================
# 6. REVERSE-DIRECTION REACTION TEST
# ============================================================

#
# Test:
#
# Does today's AI return predict the NEXT Twitter
# sentiment interval?
#
# S_(t+1) =
# alpha + phi S_t + beta Return_t
#

reverse = aligned[
    [
        "target_trade_date",
        "twitter_sentiment",
        "ai_basket_return",
    ]
].copy()


reverse["future_twitter_sentiment"] = (
    reverse[
        "twitter_sentiment"
    ].shift(-1)
)


reverse = reverse.dropna()


X_reverse_restricted = sm.add_constant(
    reverse[
        [
            "twitter_sentiment"
        ]
    ]
)


reverse_restricted = sm.OLS(
    reverse[
        "future_twitter_sentiment"
    ],
    X_reverse_restricted,
).fit()


X_reverse = sm.add_constant(
    reverse[
        [
            "twitter_sentiment",
            "ai_basket_return",
        ]
    ]
)


reverse_model = sm.OLS(
    reverse[
        "future_twitter_sentiment"
    ],
    X_reverse,
).fit()


reverse_f, reverse_p, _ = (
    reverse_model.compare_f_test(
        reverse_restricted
    )
)


reverse_robust = sm.OLS(
    reverse[
        "future_twitter_sentiment"
    ],
    X_reverse,
).fit(
    cov_type="HC1"
)


print("\n" + "=" * 75)
print("REVERSE-DIRECTION REACTION TEST")
print("=" * 75)


print(
    "\nAI basket return -> "
    "subsequent Twitter sentiment"
)

print(
    f"N = {len(reverse)}"
)

print(
    f"F-stat = {reverse_f:.4f}"
)

print(
    f"F-test p-value = "
    f"{reverse_p:.4f}"
)

print(
    f"HC1 return coefficient = "
    f"{reverse_robust.params['ai_basket_return']:.4f}"
)

print(
    f"HC1 p-value = "
    f"{reverse_robust.pvalues['ai_basket_return']:.4f}"
)


# ============================================================
# 7. SAVE FORMAL DATA
# ============================================================

formal.to_csv(
    OUTPUT_DIR /
    "formal_granger_dataset.csv",
    index=False,
)


print("\n" + "=" * 75)
print("FILES CREATED")
print("=" * 75)

print(
    OUTPUT_DIR /
    "formal_granger_dataset.csv"
)

print(
    OUTPUT_DIR /
    "formal_granger_results.csv"
)

print(
    OUTPUT_DIR /
    "aligned_predictive_results.csv"
)

print("\nDONE.")