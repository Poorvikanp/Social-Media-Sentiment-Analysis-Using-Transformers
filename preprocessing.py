import re
import pandas as pd


COLS = ["tweetid", "entity", "sentiment", "content"]


def load_twitter(path: str) -> pd.DataFrame:
    """
    Load Twitter Entity Sentiment dataset.

    The CSV files do not contain a header row.
    """

    df = pd.read_csv(
        path,
        header=None,
        names=COLS
    )

    # Remove missing text
    df = df.dropna(
        subset=["content"]
    )

    # Remove duplicate tweets
    df = df.drop_duplicates(
        subset="content"
    ).copy()

    # Normalize labels
    # Irrelevant is mapped to Neutral
    df["label"] = (
        df["sentiment"]
        .str.lower()
        .replace({"irrelevant": "neutral"})
    )

    return df.reset_index(drop=True)


def clean_baseline(t: str) -> str:
    """
    Aggressive preprocessing for TF-IDF baseline.

    - lowercase
    - remove URLs
    - remove @mentions
    """

    t = re.sub(
        r"http\S+|@\w+",
        " ",
        str(t).lower()
    )

    return re.sub(
        r"\s+",
        " ",
        t
    ).strip()


def clean_transformer(t: str) -> str:
    """
    Minimal preprocessing for Transformer.

    Preserve most of the original social-media text.
    """

    t = re.sub(
        r"@\w+",
        "@user",
        str(t)
    )

    t = re.sub(
        r"http\S+",
        "http",
        t
    )

    return t.strip()