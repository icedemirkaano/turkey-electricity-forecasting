"""Data quality checks for time-indexed tables."""

import pandas as pd


def infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Most common gap between consecutive unique timestamps (e.g. 1h or 15min)."""
    return pd.Series(index.unique().sort_values()).diff().mode().iloc[0]


def missing_timestamps(index: pd.DatetimeIndex, step: pd.Timedelta) -> pd.DatetimeIndex:
    """Timestamps expected on a regular grid between first and last, but absent."""
    expected = pd.date_range(index.min(), index.max(), freq=step)
    return expected.difference(index.unique())


def profile_table(df: pd.DataFrame) -> dict:
    """Time coverage and duplicate/missing timestamp counts for one table."""
    step = infer_step(df.index)
    missing = missing_timestamps(df.index, step)
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "step": str(step),
        "first": df.index.min(),
        "last": df.index.max(),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "missing_timestamps": len(missing),
        "missing_days": sorted({t.date() for t in missing}),
    }


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per numeric column: nulls, negatives, zeros, min, median, max."""
    num = df.select_dtypes("number")
    return pd.DataFrame({
        "nulls": num.isna().sum(),
        "negatives": (num < 0).sum(),
        "zeros": (num == 0).sum(),
        "min": num.min(),
        "median": num.median(),
        "max": num.max(),
    })
