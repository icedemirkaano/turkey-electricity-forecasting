"""Tests for src/clean.py on small hand-made tables (no real data needed)."""

import numpy as np
import pandas as pd

from src.clean import (LOAD_PLAN_FILL_LAGS_DAYS, WIND_FILL_LAGS_DAYS, clip_negative_generation,
                       fill_from_past_same_hour, to_hourly_wind)

TZ = "Europe/Istanbul"


def hourly_series(days, value_fn):
    index = pd.date_range("2026-04-01", periods=24 * days, freq="h", tz=TZ)
    return pd.Series([float(value_fn(t)) for t in index], index=index)


def test_fill_uses_same_weekday_previous_four_weeks():
    s = hourly_series(35, lambda t: t.day)  # value = day of month
    target = pd.Timestamp("2026-04-29 14:00", tz=TZ)  # a Wednesday
    s[target] = np.nan
    filled, mask = fill_from_past_same_hour(s, LOAD_PLAN_FILL_LAGS_DAYS)
    assert filled[target] == np.mean([22, 15, 8, 1])  # the four previous Wednesdays
    assert mask[target] and mask.sum() == 1


def test_fill_uses_previous_seven_days():
    s = hourly_series(10, lambda t: t.day)
    target = pd.Timestamp("2026-04-09 03:00", tz=TZ)
    s[target] = np.nan
    filled, _ = fill_from_past_same_hour(s, WIND_FILL_LAGS_DAYS)
    assert filled[target] == np.mean([8, 7, 6, 5, 4, 3, 2])


def test_fill_never_uses_future_values():
    # leakage check: a huge value AFTER the gap must not change the filled value
    s = hourly_series(10, lambda t: 1.0)
    target = pd.Timestamp("2026-04-09 03:00", tz=TZ)
    s[target] = np.nan
    s[target + pd.Timedelta(days=1)] = 1_000_000
    filled, _ = fill_from_past_same_hour(s, WIND_FILL_LAGS_DAYS)
    assert filled[target] == 1.0


def test_clip_negative_generation_sets_negatives_to_zero_but_keeps_import_export():
    df = pd.DataFrame({"sun": [-2.98, 0.0, 100.0], "wind": [5.0, -0.01, 7.0], "importExport": [-900.0, 50.0, -1.0]})
    out, counts = clip_negative_generation(df)
    assert out["sun"].tolist() == [0.0, 0.0, 100.0]
    assert out["wind"].tolist() == [5.0, 0.0, 7.0]
    assert out["importExport"].tolist() == [-900.0, 50.0, -1.0]  # sign carries meaning, untouched
    assert counts.to_dict() == {"sun": 1, "wind": 1}
    assert df["sun"].iloc[0] == -2.98  # the input table is not modified


def test_fill_leaves_hour_empty_when_no_past_value_exists():
    s = hourly_series(3, lambda t: 1.0)
    first = s.index[0]
    s[first] = np.nan  # nothing before the first day
    filled, mask = fill_from_past_same_hour(s, WIND_FILL_LAGS_DAYS)
    assert np.isnan(filled[first]) and not mask[first]


def make_wind(timestamps, forecast, generation=None):
    index = pd.DatetimeIndex(pd.to_datetime(timestamps)).tz_localize(TZ)
    n = len(index)
    return pd.DataFrame({
        "quarter1": 1.0, "quarter2": 2.0, "quarter3": 3.0, "quarter4": 4.0,
        "generation": generation if generation is not None else [0.0] * n,
        "forecast": forecast,
    }, index=index)


def test_ten_minute_values_are_averaged_per_hour():
    times = pd.date_range("2025-01-01 00:00", periods=6, freq="10min")
    hourly = to_hourly_wind(make_wind(times, forecast=[10, 20, 30, 40, 50, 60]))
    assert hourly.loc["2025-01-01 00:00", "wind_forecast"].item() == 35
    assert hourly.loc["2025-01-01 00:00", "wind_n_obs"].item() == 6


def test_single_value_hour_is_flagged_by_n_obs():
    # 15-min data where only 23:00 exists in the last hour (like the real data since June 2026)
    times = ["2026-06-01 22:00", "2026-06-01 22:15", "2026-06-01 22:30", "2026-06-01 22:45", "2026-06-01 23:00"]
    hourly = to_hourly_wind(make_wind(times, forecast=[1, 1, 1, 1, 5]))
    assert hourly.loc["2026-06-01 22:00", "wind_n_obs"].item() == 4
    assert hourly.loc["2026-06-01 23:00", "wind_n_obs"].item() == 1
    assert hourly.loc["2026-06-01 23:00", "wind_forecast"].item() == 5


def test_band_columns_are_dropped_and_names_are_clear():
    times = pd.date_range("2025-01-01", periods=4, freq="15min")
    hourly = to_hourly_wind(make_wind(times, forecast=[1, 2, 3, 4]))
    assert list(hourly.columns) == ["wind_forecast", "wind_generation_monitored", "wind_n_obs"]


def test_missing_hour_stays_visible_as_nan():
    times = ["2025-01-01 00:00", "2025-01-01 02:00"]  # nothing in the 01:00 hour
    hourly = to_hourly_wind(make_wind(times, forecast=[1.0, 2.0]))
    assert len(hourly) == 3
    assert np.isnan(hourly.loc["2025-01-01 01:00", "wind_forecast"].item())
    assert hourly.loc["2025-01-01 01:00", "wind_n_obs"].item() == 0
