# Assignment 4: AI Risk Sentiment and Equity Returns

**Khushi Khanna — FRE-GY-7871A**  
Study window: September 8–30, 2026.

[Read the six-page report](AI_Risk_Sentiment_Assignment4.pdf).

This repository contains the Assignment 4 submission.

## Included evidence

- `data/twitter_posts_scored.csv`: 920 X posts, 40 per day, including original text, URLs, timestamps, classifier probabilities, labels and risk flags.
- `data/google_news_ai_raw.csv`: 3,247 collected records.
- `data/google_news_ai_scored.csv`: 2,764 unique AI-explicit headlines and their sentiment scores.
- `data/google_news_matched_40_per_day.csv`: matched 920-headline sample.
- `data/google_news_daily_qc.csv` and `data/google_news_daily_sentiment.csv`: supplied daily summaries.
- `data/market_raw.json`: retained Yahoo Finance price responses for seven AI equities and SPY.
- `src/audit_data.py`: validates the supplied data and reproduces news comparisons.
- `src/build_report.py`: computes market returns, sentiment trends, Granger tests and report figures, then regenerates the PDF.
- `outputs/`: reproduced statistics and daily market returns.

## Run

Python 3.10 or later:

```bash
cd FRE-GY-7871A-Assignment4
python -m pip install -r requirements.txt
python src/audit_data.py
python src/build_report.py
```

Commands also work from another directory because paths are resolved relative to the scripts. Generated files are written into `outputs/`. The regenerated PDF is `outputs/AI_Risk_Sentiment_Assignment4.pdf`; the reviewed submission copy is at this folder's root. Fonts use Matplotlib's bundled DejaVu Sans, including on Windows. No API credentials or live downloads are required.

## Methods

Sentiment is positive probability minus negative probability. Labels use the largest class probability. The scoring model recorded for supplied inputs is CardiffNLP `twitter-roberta-base-sentiment-latest`; these scripts use existing probabilities and do not rerun the transformer or recollect historical searches. Risk indicators use the supplied dictionary flags; audit checks test consistency, not independent model accuracy.

The AI basket equally weights NVDA, MSFT, GOOGL, AMZN, META, AMD and AVGO with daily rebalancing. Adjusted-close returns dated September 8–30 use September 4 as the initial price base. Excess return subtracts SPY. Posts from the previous session date up to the day before a session form its available pre-session sentiment. Differences between successive pre-session values define changes. Nested OLS models implement lag-one Granger F tests; coefficient uncertainty uses HC1 covariance. The first aligned session and its differencing/lag losses are documented in the PDF.

News comparisons include equal-weighted daily paired tests, a supplied 40-headline-per-day matched sample, and a category-distribution chi-square test. Publication dates in New York time govern grouping. Search request dates may differ from publication dates.

## Interpretation and limitations

X has larger negative and positive shares than news headlines, but similar average sentiment. This is a measured coverage difference, not proof of media bias. Topic mix, text length, source/author dependence, sarcasm, classifier calibration and result caps limit interpretation. The small number of trading sessions limits Granger test power and does not identify structural causality. Class readings and source references are included in the PDF.
