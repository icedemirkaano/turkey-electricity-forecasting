"""Load raw EPİAŞ tables from data/raw/<table>/<YYYY-MM>.csv into one DataFrame per table."""

import pandas as pd

from src.config import RAW_DIR, TIMEZONE

# Columns that only repeat the timestamp in another format (dropped on load).
REDUNDANT_TIME_COLUMNS = ("hour", "time")


def load_raw(key: str) -> pd.DataFrame:
    """Concatenate all month files of a table, indexed by a tz-aware local timestamp.

    The `date` column (ISO 8601 with +03:00 offset) becomes the index `timestamp`.
    Rows are sorted by time; duplicates are NOT removed here, so quality checks can see them.
    """
    files = sorted((RAW_DIR / key).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw files for '{key}' in {RAW_DIR / key}. Run scripts.fetch_data first.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(TIMEZONE)
    df = df.drop(columns=["date", *[c for c in REDUNDANT_TIME_COLUMNS if c in df.columns]])
    return df.set_index("timestamp").sort_index()
