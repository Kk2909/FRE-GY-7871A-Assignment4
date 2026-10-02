import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = Path("news_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "gdelt_ai_news_raw.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "gdelt_ai_news_checkpoint.csv"
DAILY_QC_FILE = OUTPUT_DIR / "gdelt_news_daily_qc.csv"

START_DATE = "2026-09-08"
END_DATE = "2026-09-30"

# Keep concept consistent with Twitter collection
QUERY = '("artificial intelligence" OR "generative AI") sourcelang:english'

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

MAX_RECORDS = 250

NY = ZoneInfo("America/New_York")

# Start with two 12-hour windows/day.
# If one returns 250 results, it gets split automatically.
BASE_WINDOW_HOURS = 12

# Don't recursively split below this window size.
MIN_WINDOW_MINUTES = 30

# Delay between successful API calls.
REQUEST_SLEEP = 12

# Increasing delays after 429 rate limits.
RATE_LIMIT_WAITS = [20, 40, 60, 90, 120]


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent":
        "NYU-FRE-GY-7871A-Academic-Research/1.0"
})


# ============================================================
# GDELT REQUEST
# ============================================================

def gdelt_request(start_utc, end_utc):
    """
    Request one GDELT article-list window.

    Returns
    -------
    articles : list
    success : bool
    """

    params = {
        "query": QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "sort": "datedesc",
        "startdatetime":
            start_utc.strftime("%Y%m%d%H%M%S"),
        "enddatetime":
            end_utc.strftime("%Y%m%d%H%M%S"),
    }

    for attempt, default_wait in enumerate(
        RATE_LIMIT_WAITS,
        start=1
    ):

        try:

            response = session.get(
                API_URL,
                params=params,
                timeout=90,
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                try:
                    data = response.json()

                except Exception:

                    print(
                        "Invalid JSON response. "
                        f"Waiting {default_wait}s..."
                    )

                    time.sleep(default_wait)
                    continue


                articles = data.get(
                    "articles",
                    []
                )

                return articles, True


            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                wait_time = default_wait

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:

                    try:

                        wait_time = max(
                            wait_time,
                            int(retry_after)
                        )

                    except ValueError:
                        pass


                print(
                    f"Rate limited (429). "
                    f"Waiting {wait_time}s "
                    f"before retry "
                    f"{attempt}/{len(RATE_LIMIT_WAITS)}..."
                )

                time.sleep(wait_time)

                continue


            # ------------------------------------------------
            # OTHER HTTP ERROR
            # ------------------------------------------------

            print(
                f"HTTP {response.status_code}. "
                f"Waiting {default_wait}s "
                f"before retry "
                f"{attempt}/{len(RATE_LIMIT_WAITS)}..."
            )

            time.sleep(default_wait)


        except requests.RequestException as exc:

            print(
                f"Request error: {exc}"
            )

            print(
                f"Waiting {default_wait}s..."
            )

            time.sleep(default_wait)


    print(
        "\nFAILED WINDOW AFTER ALL RETRIES:"
    )

    print(
        start_utc,
        "to",
        end_utc
    )

    return [], False


# ============================================================
# FORMAT ARTICLE
# ============================================================

def format_article(
    article,
    requested_date,
    start_utc,
    end_utc,
):

    return {

        "requested_date_ny":
            requested_date,

        "window_start_utc":
            start_utc.isoformat(),

        "window_end_utc":
            end_utc.isoformat(),

        "title":
            article.get("title"),

        "url":
            article.get("url"),

        "url_mobile":
            article.get("url_mobile"),

        "domain":
            article.get("domain"),

        "language":
            article.get("language"),

        "sourcecountry":
            article.get("sourcecountry"),

        "seendate":
            article.get("seendate"),

        "socialimage":
            article.get("socialimage"),
    }


# ============================================================
# RECURSIVE WINDOW COLLECTION
# ============================================================

def collect_window(
    start_utc,
    end_utc,
    requested_date,
    depth=0,
):

    articles, success = gdelt_request(
        start_utc,
        end_utc
    )


    indent = "  " * depth

    duration_minutes = (
        end_utc - start_utc
    ).total_seconds() / 60


    if not success:

        print(
            indent
            + "WINDOW FAILED"
        )

        return []


    print(
        indent
        + f"{start_utc.strftime('%Y-%m-%d %H:%M')} "
        + "to "
        + f"{end_utc.strftime('%Y-%m-%d %H:%M')} UTC"
        + f" -> {len(articles)} articles"
    )


    # ========================================================
    # HIT 250-ARTICLE LIMIT:
    # split into smaller windows
    # ========================================================

    if (
        len(articles) >= MAX_RECORDS
        and duration_minutes > MIN_WINDOW_MINUTES
    ):

        print(
            indent
            + "Hit 250-record cap. Splitting window..."
        )

        midpoint = (
            start_utc
            + (end_utc - start_utc) / 2
        )


        left = collect_window(
            start_utc,
            midpoint,
            requested_date,
            depth + 1,
        )


        time.sleep(
            REQUEST_SLEEP
        )


        right = collect_window(
            midpoint,
            end_utc,
            requested_date,
            depth + 1,
        )


        return left + right


    # ========================================================
    # CONVERT ARTICLES TO ROWS
    # ========================================================

    rows = []

    for article in articles:

        rows.append(
            format_article(
                article,
                requested_date,
                start_utc,
                end_utc,
            )
        )


    return rows


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(rows):

    if not rows:
        return

    checkpoint_df = pd.DataFrame(
        rows
    )

    checkpoint_df.to_csv(
        CHECKPOINT_FILE,
        index=False
    )


# ============================================================
# LOAD EXISTING CHECKPOINT IF PRESENT
# ============================================================

all_rows = []

completed_dates = set()


if CHECKPOINT_FILE.exists():

    try:

        old_checkpoint = pd.read_csv(
            CHECKPOINT_FILE
        )

        if not old_checkpoint.empty:

            all_rows = old_checkpoint.to_dict(
                orient="records"
            )

            if (
                "requested_date_ny"
                in old_checkpoint.columns
            ):

                completed_dates = set(
                    old_checkpoint[
                        "requested_date_ny"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                )


            print(
                "\nExisting checkpoint found."
            )

            print(
                f"Rows already collected: "
                f"{len(old_checkpoint):,}"
            )


    except Exception as exc:

        print(
            f"Could not load checkpoint: "
            f"{exc}"
        )


# ============================================================
# BUILD DATE RANGE
# ============================================================

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)


print("\n" + "=" * 70)
print("GDELT AI NEWS COLLECTION")
print("=" * 70)

print(
    f"Period: {START_DATE} to {END_DATE}"
)

print(
    f"Query: {QUERY}"
)

print(
    f"Base window: {BASE_WINDOW_HOURS} hours"
)

print(
    f"Delay between requests: "
    f"{REQUEST_SLEEP} seconds"
)


# ============================================================
# MAIN COLLECTION LOOP
# ============================================================

for day in dates:

    requested_date = (
        day.strftime("%Y-%m-%d")
    )


    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if requested_date in completed_dates:

        print(
            f"\n{requested_date}: "
            f"already in checkpoint -> skipping"
        )

        continue


    print(
        "\n"
        + "=" * 50
    )

    print(
        requested_date
    )

    print(
        "=" * 50
    )


    # --------------------------------------------------------
    # Define NY calendar day
    # --------------------------------------------------------

    local_start = datetime(
        day.year,
        day.month,
        day.day,
        0,
        0,
        0,
        tzinfo=NY,
    )


    local_end = (
        local_start
        + timedelta(days=1)
    )


    # Convert New York day to UTC
    day_start_utc = (
        local_start.astimezone(
            timezone.utc
        )
    )

    day_end_utc = (
        local_end.astimezone(
            timezone.utc
        )
    )


    # --------------------------------------------------------
    # Collect this date
    # --------------------------------------------------------

    day_rows = []

    current = day_start_utc


    while current < day_end_utc:

        window_end = min(
            current
            + timedelta(
                hours=BASE_WINDOW_HOURS
            ),
            day_end_utc,
        )


        rows = collect_window(
            current,
            window_end,
            requested_date,
        )


        day_rows.extend(
            rows
        )


        current = window_end


        # Sleep before next base window
        if current < day_end_utc:

            time.sleep(
                REQUEST_SLEEP
            )


    # --------------------------------------------------------
    # Add completed day to master data
    # --------------------------------------------------------

    all_rows.extend(
        day_rows
    )


    # --------------------------------------------------------
    # Save checkpoint AFTER every complete date
    # --------------------------------------------------------

    save_checkpoint(
        all_rows
    )


    print(
        f"\n{requested_date} complete."
    )

    print(
        f"Rows collected for day: "
        f"{len(day_rows):,}"
    )

    print(
        f"Total checkpoint rows: "
        f"{len(all_rows):,}"
    )


    # Pause before starting next day
    time.sleep(
        REQUEST_SLEEP
    )


# ============================================================
# RAW DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_rows
)


print("\n" + "=" * 70)
print("RAW COLLECTION")
print("=" * 70)

print(
    f"Rows retrieved: {len(df):,}"
)


if df.empty:

    raise RuntimeError(
        "No GDELT articles were retrieved."
    )


# ============================================================
# CLEAN BASIC FIELDS
# ============================================================

df["title"] = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df["url"] = (
    df["url"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df["domain"] = (
    df["domain"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove empty titles
df = df[
    df["title"].ne("")
].copy()


before_dedup = len(df)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

# First: exact URLs
df = df.drop_duplicates(
    subset=["url"],
    keep="first",
).copy()


# Normalize titles
df["title_normalized"] = (
    df["title"]
    .str.lower()
    .str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    .str.strip()
)


# Second: same title from same domain
df = df.drop_duplicates(
    subset=[
        "title_normalized",
        "domain",
    ],
    keep="first",
).copy()


duplicates_removed = (
    before_dedup - len(df)
)


# ============================================================
# PARSE GDELT TIMESTAMPS
# ============================================================

df["published_utc"] = pd.to_datetime(
    df["seendate"],
    format="%Y%m%dT%H%M%SZ",
    errors="coerce",
    utc=True,
)


# Fallback parser
missing_time = (
    df["published_utc"].isna()
)


if missing_time.any():

    df.loc[
        missing_time,
        "published_utc"
    ] = pd.to_datetime(
        df.loc[
            missing_time,
            "seendate"
        ],
        errors="coerce",
        utc=True,
    )


# ============================================================
# CONVERT TO NEW YORK DATE
# ============================================================

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
# KEEP ONLY INTENDED SAMPLE PERIOD
# ============================================================

df = df[
    (df["date_ny"] >= START_DATE)
    &
    (df["date_ny"] <= END_DATE)
].copy()


# ============================================================
# FLAG WHETHER AI IS EXPLICITLY IN TITLE
# ============================================================

title_lower = (
    df["title"]
    .str.lower()
)


df["ai_in_title"] = (

    title_lower.str.contains(
        "artificial intelligence",
        regex=False,
    )

    |

    title_lower.str.contains(
        "generative ai",
        regex=False,
    )

    |

    title_lower.str.contains(
        r"\bai\b",
        regex=True,
    )

).astype(int)


# ============================================================
# DOMAIN NORMALIZATION
# ============================================================

df["domain_clean"] = (
    df["domain"]
    .str.lower()
    .str.replace(
        r"^www\.",
        "",
        regex=True,
    )
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "date_ny",
        "published_utc",
        "domain_clean",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE CLEAN RAW ARTICLE DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DAILY QC TABLE
# ============================================================

daily = (
    df
    .groupby("date_ny")
    .agg(

        n_articles=(
            "url",
            "count",
        ),

        n_domains=(
            "domain_clean",
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
# FINAL QC
# ============================================================

print("\n" + "=" * 70)
print("NEWS COLLECTION QC")
print("=" * 70)


print(
    f"Raw non-empty articles: "
    f"{before_dedup:,}"
)

print(
    f"Unique articles:        "
    f"{len(df):,}"
)

print(
    f"Duplicates removed:     "
    f"{duplicates_removed:,}"
)

print(
    f"Dates represented:      "
    f"{df['date_ny'].nunique()}"
)


if not df.empty:

    print(
        f"Date range:             "
        f"{df['date_ny'].min()} "
        f"to "
        f"{df['date_ny'].max()}"
    )


print(
    f"Unique domains:         "
    f"{df['domain_clean'].nunique():,}"
)


print(
    f"AI explicitly in title:"
    f" {df['ai_in_title'].mean():.1%}"
)


# ============================================================
# DAILY COUNTS
# ============================================================

print("\nArticles by day:")

print(
    daily.to_string(
        index=False
    )
)


# ============================================================
# TOP DOMAINS
# ============================================================

print("\nTop 20 domains:")

print(
    df["domain_clean"]
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    OUTPUT_FILE
)

print(
    DAILY_QC_FILE
)

print(
    CHECKPOINT_FILE
)

print("\nDONE.")