"""Download EPİAŞ tables month by month into data/raw/<table>/<YYYY-MM>.csv.

A month file is skipped only when it is complete, i.e. its last timestamp reaches the
end of that month (or yesterday, for the current month). Incomplete months — like the
current one — are downloaded again, so re-running always brings the data up to date.
Temporary errors (429 throttling, timeouts) are retried with an increasing wait.
"""

import time
from pathlib import Path

import pandas as pd
from eptr2 import EPTR2

from src.config import RAW_DIR, TIMEZONE

TIME_COLUMNS = ("date", "period", "hour")
MAX_ATTEMPTS = 4
PAUSE_SECONDS = 1.0


def month_ranges(start_month: str, end_month: str, today: pd.Timestamp | None = None):
    """Yield (label, first_day, last_day) per month; the last day never goes past yesterday."""
    today = today if today is not None else pd.Timestamp.now(tz=TIMEZONE)
    if today.tzinfo is not None:
        today = today.tz_localize(None)
    today = today.normalize()
    yesterday = today - pd.Timedelta(days=1)
    for month in pd.period_range(start_month, end_month, freq="M"):
        first = month.start_time
        last = min(month.end_time.normalize(), yesterday)
        if first <= last:
            yield str(month), first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")


def last_timestamp(path: Path):
    """Latest timestamp in a saved month file (local time, tz-naive), or None."""
    header = pd.read_csv(path, nrows=0).columns
    col = next((c for c in TIME_COLUMNS if c in header), None)
    if col is None:
        return None
    times = pd.to_datetime(pd.read_csv(path, usecols=[col])[col], utc=True, errors="coerce")
    times = times.dt.tz_convert(TIMEZONE).dt.tz_localize(None).dropna()
    return times.max() if not times.empty else None


def is_complete(path: Path, last_day: str) -> bool:
    """True if the file exists and covers the month up to its last day."""
    if not path.exists():
        return False
    latest = last_timestamp(path)
    return latest is not None and latest.normalize() >= pd.Timestamp(last_day)


def fetch_month(eptr: EPTR2, key: str, first_day: str, last_day: str) -> pd.DataFrame:
    """Call one table for one month, retrying temporary errors."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return eptr.call(key, start_date=first_day, end_date=last_day)
        except Exception as err:
            message = str(err).replace("\n", " ")
            temporary = "429" in message or "timed out" in message
            if not temporary or attempt == MAX_ATTEMPTS:
                raise
            wait = 30 * attempt
            print(f"    temporary error ({message[:40]}...), waiting {wait}s")
            time.sleep(wait)


def fetch_table(eptr: EPTR2, key: str, start_month: str, end_month: str) -> list[str]:
    """Download all incomplete months of one table; return the months that failed."""
    folder = RAW_DIR / key
    folder.mkdir(parents=True, exist_ok=True)
    failed = []
    for label, first_day, last_day in month_ranges(start_month, end_month):
        path = folder / f"{label}.csv"
        if is_complete(path, last_day):
            continue
        try:
            df = fetch_month(eptr, key, first_day, last_day)
            df.to_csv(path, index=False)
            print(f"  {key:13s} {label}  {len(df):6d} rows")
        except Exception as err:
            failed.append(label)
            print(f"  {key:13s} {label}  ERROR: {str(err).replace(chr(10), ' ')[:100]}")
        time.sleep(PAUSE_SECONDS)
    return failed
