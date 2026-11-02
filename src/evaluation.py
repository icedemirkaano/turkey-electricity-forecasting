"""Shared evaluation setup: walk-forward folds, test period, metrics and simple baselines.

Every model is evaluated on exactly the same periods with exactly the same metrics:
  - validation: quarterly walk-forward over 2025 (train on everything before each quarter)
  - test: January-September 2026, used once at the end
  - demo: October 2026 onwards (never used for training or model choice)
"""

import numpy as np
import pandas as pd

from src.config import TIMEZONE
from src.features import MODEL_START

# (name, first day, last day) of each validation quarter; training always runs from MODEL_START to the day before
WALK_FORWARD_FOLDS = [
    ("2025Q1", "2025-01-01", "2025-03-31"),
    ("2025Q2", "2025-04-01", "2025-06-30"),
    ("2025Q3", "2025-07-01", "2025-09-30"),
    ("2025Q4", "2025-10-01", "2025-12-31"),
]
TRAIN_END = "2025-12-31"          # final model is trained on MODEL_START .. TRAIN_END
TEST_PERIOD = ("2026-01-01", "2026-09-30")
DEMO_START = "2026-10-01"


def _day(text: str, end_of_day=False) -> pd.Timestamp:
    t = pd.Timestamp(text, tz=TIMEZONE)
    return t + pd.Timedelta(hours=23) if end_of_day else t


def period(frame, first: str, last: str):
    """Rows of `frame` (indexed by hourly timestamps) between two dates, both days included."""
    return frame.loc[_day(first):_day(last, end_of_day=True)]


def walk_forward_splits(X: pd.DataFrame, y: pd.Series):
    """Yield (fold name, X_train, y_train, X_val, y_val). Training ends the day before validation starts."""
    for name, first, last in WALK_FORWARD_FOLDS:
        train_last = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        yield (name, period(X, MODEL_START, train_last), period(y, MODEL_START, train_last),
               period(X, first, last), period(y, first, last))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def scores(y_true: pd.Series, y_pred: pd.Series, flags: pd.DataFrame | None = None) -> dict:
    """MAE and RMSE overall, plus MAE on price-floor and price-cap hours when `flags` is given
    (a frame with boolean columns is_price_floor and is_price_cap, aligned by timestamp)."""
    out = {"hours": len(y_true), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}
    if flags is not None:
        for flag in ("is_price_floor", "is_price_cap"):
            mask = flags[flag].reindex(y_true.index).fillna(False).astype(bool)
            out[f"mae_{flag.removeprefix('is_price_')}_hours"] = mae(y_true[mask], y_pred[mask]) if mask.any() else np.nan
            out[f"{flag.removeprefix('is_price_')}_hours"] = int(mask.sum())
    return out


def scores_by_month(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """MAE and RMSE for each calendar month of the period."""
    months = y_true.index.strftime("%Y-%m")
    rows = [{"month": m, "mae": mae(y_true[months == m], y_pred[months == m]),
             "rmse": rmse(y_true[months == m], y_pred[months == m])} for m in sorted(set(months))]
    return pd.DataFrame(rows).set_index("month")


# Baselines need no training: they reuse a lagged price feature as the forecast
BASELINES = {"same hour yesterday": "ptf_usd_lag24", "same hour last week": "ptf_usd_lag168"}


def baseline_forecast(X: pd.DataFrame, name: str) -> pd.Series:
    return X[BASELINES[name]].rename(name)


# Retraining windows (post-hoc experiments): pandas frequency of the first day of each window
RETRAIN_WINDOW_STARTS = {"month": "MS", "week": "W-MON"}


def retrain_windows(first: str, last: str, every: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(first day, last day) of each retraining window between two dates (weeks start on Monday)."""
    start, end = pd.Timestamp(first), pd.Timestamp(last)
    starts = sorted({start, *pd.date_range(start, end, freq=RETRAIN_WINDOW_STARTS[every])})
    ends = [s - pd.Timedelta(days=1) for s in starts[1:]] + [end]
    return list(zip(starts, ends))
