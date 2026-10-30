"""Model inputs (features) for the next-day PTF forecast, built only from what is known at forecast time.

Forecast time: day D, before the 12:30 day-ahead offer deadline; targets are the 24 hours of day D+1.
What is known then, relative to a target hour on D+1 (see COLUMN_AVAILABILITY in src/dataset.py):
  - known_ahead columns (load and wind forecasts, price cap): their value FOR the target hour -> lag 0
  - PTF: today's 24 prices were published yesterday at 14:00 -> any lag >= 24 hours
  - other past_only columns (realized generation, consumption, other prices): known only up to about
    the morning of D, so the latest full day is D-1 -> lag >= 48 hours
  - evaluation_only columns: never used
Every feature is declared in FEATURE_SPEC and checked against these rules before it is built.
"""

import pandas as pd

from src.dataset import COLUMN_AVAILABILITY
from src.holidays_tr import holiday_flags

TARGET = "ptf_usd"
MODEL_START = "2024-01-01"  # instructor's decision: 2023 left out of modelling (lags may still reach into 2023)

# Minimum lag (hours) a column needs before it may be used for a target hour, by availability label
MIN_LAG = {"known_ahead": 0, "past_only": 48, "target": 24}
PRICE_COLUMNS_KNOWN_FOR_TODAY = {"ptf_usd", "ptf_try", "ptf_eur", "is_price_floor", "is_price_cap"}

# (source column, lag in hours) -> feature named "<column>" for lag 0, "<column>_lag<lag>" otherwise
FEATURE_SPEC = [
    ("load_forecast", 0),
    ("wind_forecast", 0),
    ("net_load_forecast", 0),   # derived: load_forecast - wind_forecast (both known ahead)
    ("price_cap_try", 0),
    ("ptf_usd", 24),            # same hour yesterday
    ("ptf_usd", 48),
    ("ptf_usd", 168),           # same hour last week
    ("gen_solar", 48),          # decision A1: realized solar of the latest known full day
    ("gen_solar", 168),
]
DERIVED_KNOWN_AHEAD = {"net_load_forecast"}


def min_lag(column: str) -> int:
    """Smallest lag (hours) at which `column` is known at forecast time. Raises for columns never allowed."""
    if column in DERIVED_KNOWN_AHEAD:
        return 0
    label = COLUMN_AVAILABILITY.get(column)
    if label is None or label == "evaluation_only":
        raise ValueError(f"'{column}' may not be used as a model input (label: {label})")
    if column in PRICE_COLUMNS_KNOWN_FOR_TODAY:
        return 24
    return MIN_LAG[label]


def check_spec(spec=FEATURE_SPEC) -> None:
    """Raise if any feature would use information that is not available at forecast time."""
    for column, lag in spec:
        needed = min_lag(column)
        if lag < needed:
            raise ValueError(f"leakage: '{column}' used with lag {lag} h, needs at least {needed} h")


def feature_name(column: str, lag: int) -> str:
    return column if lag == 0 else f"{column}_lag{lag}"


def build_features(table: pd.DataFrame, spec=FEATURE_SPEC, start=MODEL_START) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for hours from `start` on. `table` is the hourly table from src/dataset.py."""
    check_spec(spec)
    source = table.copy()
    source["net_load_forecast"] = source["load_forecast"] - source["wind_forecast"]

    features = {feature_name(c, lag): source[c].shift(lag) for c, lag in spec}  # hourly index -> shift = hours

    # Mean price of the previous day (day D for a target on D+1): known since D-1 14:00
    daily_mean = source[TARGET].groupby(source.index.normalize()).transform("mean")
    features["ptf_usd_prev_day_mean"] = daily_mean.shift(24)

    X = pd.DataFrame(features, index=source.index)
    X = X.loc[start:]

    # Calendar of the target hour (always known in advance)
    X["hour"] = X.index.hour
    X["day_of_week"] = X.index.dayofweek
    X["is_weekend"] = X["day_of_week"] >= 5
    X["month"] = X.index.month
    X = X.join(holiday_flags(X.index))
    X = X.astype({c: int for c in X.columns if X[c].dtype == bool})

    y = source.loc[start:, TARGET]
    return X, y
