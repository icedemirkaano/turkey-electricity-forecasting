"""Tests for src/features.py: leakage rules on a small table, plus checks on the real data if available."""

import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR
from src.features import FEATURE_SPEC, build_features, check_spec, min_lag

TZ = "Europe/Istanbul"
DATASET = PROCESSED_DIR / "ptf_hourly.parquet"


def small_table(days=20, start="2023-12-20"):
    index = pd.date_range(start, periods=24 * days, freq="h", tz=TZ)
    n = len(index)
    return pd.DataFrame({
        "ptf_usd": np.arange(n, dtype=float),          # value = position, so lags are easy to check
        "load_forecast": 30_000.0, "wind_forecast": 5_000.0,
        "price_cap_try": 3_000.0, "gen_solar": np.arange(n, dtype=float) * 10,
    }, index=index)


# --- leakage rules -------------------------------------------------------------------------------

def test_declared_features_respect_the_rules():
    check_spec()  # must not raise


def test_realized_generation_needs_48_hours():
    with pytest.raises(ValueError, match="leakage"):
        check_spec([("gen_solar", 24)])  # tomorrow's 20:00 would use today's 20:00, not yet realized


def test_price_needs_24_hours_and_cannot_be_used_same_hour():
    with pytest.raises(ValueError, match="leakage"):
        check_spec([("ptf_usd", 0)])
    check_spec([("ptf_usd", 24)])  # today's prices were published yesterday


def test_evaluation_only_column_is_never_allowed():
    with pytest.raises(ValueError):
        min_lag("wind_generation_monitored")


# --- feature values ------------------------------------------------------------------------------

def test_lags_point_to_the_right_hours():
    X, y = build_features(small_table(), start="2024-01-01")
    t = pd.Timestamp("2024-01-05 20:00", tz=TZ)
    assert X.loc[t, "ptf_usd_lag24"] == y.loc[t] - 24
    assert X.loc[t, "ptf_usd_lag168"] == y.loc[t] - 168
    assert X.loc[t, "gen_solar_lag48"] == (y.loc[t] - 48) * 10
    assert X.loc[t, "net_load_forecast"] == 25_000


def test_previous_day_mean_uses_only_the_previous_day():
    X, y = build_features(small_table(), start="2024-01-01")
    day = y.loc["2024-01-04"]
    assert (X.loc["2024-01-05", "ptf_usd_prev_day_mean"] == day.mean()).all()


def test_calendar_and_holidays():
    X, _ = build_features(small_table(), start="2024-01-01")
    assert X.loc["2024-01-01 10:00+03:00", "is_holiday"] == 1  # New Year's Day
    assert X.loc["2024-01-06 10:00+03:00", "is_weekend"] == 1  # Saturday
    assert X.index.min() == pd.Timestamp("2024-01-01", tz=TZ)


# --- real data -----------------------------------------------------------------------------------

@pytest.mark.skipif(not DATASET.exists(), reason="processed dataset not built")
def test_real_features_have_no_missing_values():
    X, y = build_features(pd.read_parquet(DATASET))
    assert X.notna().all().all() and y.notna().all()
    assert len(X) == len(y)
    assert len(X.columns) == len(FEATURE_SPEC) + 1 + 4 + 2  # spec + previous-day mean + calendar + holidays
