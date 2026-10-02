import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = Path("news_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "google_news_ai_raw.csv"
DAILY_QC_FILE = OUTPUT_DIR / "google_news_daily_qc.csv"

START_DATE = "2026-09-08"
END_DATE = "2026-09-30"

BASE_URL = "https://news.google.com/rss/search"

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 NYU Academic Research"
}

REQUEST_SLEEP = 2


# Separate searches make the OR query more robust
SEARCH_TERMS = [
    '"artificial intelligence"',
    '"generative AI"',
]


# ============================================================
# FETCH ONE RSS QUERY
# ============================================================

def fetch_rss(query):

    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }

    response = requests.get(
        BASE_URL,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    return response.text


# ============================================================
# PARSE RSS
# ============================================================

def parse_rss(xml_text, search_term, requested_date):

    root = ET.fromstring(xml_text)

    rows = []

    for item in root.findall(".//item"):

        title_node = item.find("title")
        link_node = item.find("link")
        pubdate_node = item.find("pubDate")
        source_node = item.find("source")

        title = (
            title_node.text.strip()
            if title_node is not None
            and title_node.text
            else ""
        )

        url = (
            link_node.text.strip()
            if link_node is not None
            and link_node.text
            else ""
        )

        pub_date = (
            pubdate_node.text.strip()
            if pubdate_node is not None
            and pubdate_node.text
            else None
        )

        source = (
            source_node.text.strip()
            if source_node is not None
            and source_node.text
            else None
        )

        rows.append({
            "requested_date": requested_date,
            "search_term": search_term,
            "title": title,
            "google_news_url": url,
            "published_raw": pub_date,
            "source": source,
        })

    return rows


# ============================================================
# MAIN LOOP
# ============================================================

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)

all_rows = []

print("\n" + "=" * 70)
print("GOOGLE NEWS AI COLLECTION")
print("=" * 70)

print(
    f"Period: {START_DATE} to {END_DATE}"
)

print(
    f"Search terms: {SEARCH_TERMS}"
)


for day in dates:

    day_string = day.strftime(
        "%Y-%m-%d"
    )

    next_day = (
        day + timedelta(days=1)
    ).strftime(
        "%Y-%m-%d"
    )

    print(
        f"\n{day_string}"
    )

    day_rows = []

    for term in SEARCH_TERMS:

        query = (
            f'{term} '
            f'after:{day_string} '
            f'before:{next_day}'
        )

        try:

            xml_text = fetch_rss(
                query
            )

            rows = parse_rss(
                xml_text,
                search_term=term,
                requested_date=day_string,
            )

            print(
                f"  {term}: "
                f"{len(rows)} headlines"
            )

            day_rows.extend(
                rows
            )

        except Exception as exc:

            print(
                f"  ERROR for {term}: "
                f"{exc}"
            )

        time.sleep(
            REQUEST_SLEEP
        )

    all_rows.extend(
        day_rows
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_rows
)

if df.empty:
    raise RuntimeError(
        "No Google News results retrieved."
    )


print("\n" + "=" * 70)
print("RAW COLLECTION")
print("=" * 70)

print(
    f"Raw rows: {len(df):,}"
)


# ============================================================
# CLEAN TITLES
# ============================================================

df["title"] = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["source"] = (
    df["source"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df = df[
    df["title"].ne("")
].copy()


# ============================================================
# REMOVE GOOGLE'S "- SOURCE" SUFFIX
# ============================================================

#
# Google News titles often look like:
#
# "OpenAI launches new model - Reuters"
#
# For sentiment, we want:
#
# "OpenAI launches new model"
#

def clean_google_title(row):

    title = row["title"]
    source = row["source"]

    if source:

        suffix = f" - {source}"

        if title.endswith(suffix):

            title = title[
                :-len(suffix)
            ]

    return title.strip()


df["headline"] = df.apply(
    clean_google_title,
    axis=1,
)


# ============================================================
# PARSE DATE
# ============================================================

df["published_utc"] = pd.to_datetime(
    df["published_raw"],
    errors="coerce",
    utc=True,
)


df["published_ny"] = (
    df["published_utc"]
    .dt.tz_convert(
        "America/New_York"
    )
)


df["date_ny"] = (
    df["published_ny"]
    .dt.strftime(
        "%Y-%m-%d"
    )
)


# ============================================================
# NORMALIZE TITLES
# ============================================================

df["headline_normalized"] = (
    df["headline"]
    .str.lower()
    .str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    .str.strip()
)


# ============================================================
# DEDUPLICATE
# ============================================================

before = len(df)


# Same story can appear in both search queries
df = df.drop_duplicates(
    subset=[
        "headline_normalized",
        "source",
    ],
    keep="first",
).copy()


duplicates_removed = (
    before - len(df)
)


# ============================================================
# AI TITLE FLAG
# ============================================================

headline_lower = (
    df["headline"]
    .str.lower()
)


df["ai_in_title"] = (

    headline_lower.str.contains(
        "artificial intelligence",
        regex=False,
    )

    |

    headline_lower.str.contains(
        "generative ai",
        regex=False,
    )

    |

    headline_lower.str.contains(
        r"\bai\b",
        regex=True,
    )

).astype(int)


# ============================================================
# IMPORTANT DATE QC
# ============================================================

#
# Google date filters can occasionally include boundary
# records depending on timestamp/time zone.
#
# Keep intended NY period.
#

df = df[
    (df["date_ny"] >= START_DATE)
    &
    (df["date_ny"] <= END_DATE)
].copy()


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "date_ny",
        "published_utc",
        "source",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DAILY QC
# ============================================================

daily = (
    df
    .groupby("date_ny")
    .agg(

        n_articles=(
            "headline",
            "count",
        ),

        n_sources=(
            "source",
            "nunique",
        ),

        ai_title_share=(
            "ai_in_title",
            "mean",
        ),

    )
    .reset_index()
)


daily.to_csv(
    DAILY_QC_FILE,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("NEWS COLLECTION QC")
print("=" * 70)

print(
    f"Unique headlines:      "
    f"{len(df):,}"
)

print(
    f"Duplicates removed:    "
    f"{duplicates_removed:,}"
)

print(
    f"Dates represented:     "
    f"{df['date_ny'].nunique()}"
)

print(
    f"Date range:            "
    f"{df['date_ny'].min()} "
    f"to "
    f"{df['date_ny'].max()}"
)

print(
    f"Unique sources:        "
    f"{df['source'].nunique():,}"
)

print(
    f"AI explicitly in title:"
    f" {df['ai_in_title'].mean():.1%}"
)


print("\nArticles by day:")

print(
    daily.to_string(
        index=False
    )
)


print("\nTop 20 sources:")

print(
    df["source"]
    .value_counts()
    .head(20)
    .to_string()
)


print("\nFILES CREATED:")

print(
    OUTPUT_FILE
)

print(
    DAILY_QC_FILE
)

print("\nDONE.")