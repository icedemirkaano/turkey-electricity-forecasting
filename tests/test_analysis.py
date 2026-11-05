import numpy as np
import pandas as pd
import pytest

from src import analysis as an


def _hours(days: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-10-01", periods=24 * days, freq="h", tz="Europe/Istanbul")


def test_daily_profile_shape_and_missing_hours():
    series = pd.Series(np.arange(48.0), index=_hours(2))
    profile = an.daily_profile(series)
    assert profile.shape == (2, 24) and profile.iloc[1, 0] == 24
    with pytest.raises(ValueError):
        an.daily_profile(series.iloc[:-1])


def test_day_summary_tariff_periods():
    row = pd.DataFrame([np.arange(24.0)])
    summary = an.day_summary(row).iloc[0]
    assert summary["peak"] == np.mean([17, 18, 19, 20, 21])
    assert summary["night"] == np.mean([22, 23, 0, 1, 2, 3, 4, 5])
    assert summary["cheapest_hour"] == 0 and summary["priciest_hour"] == 23 and summary["spread"] == 23


def test_hit_rate_counts_ties_as_hits():
    actual = pd.DataFrame([[0.0] * 8 + [50.0] * 16])          # eight zero-price hours
    forecast = pd.DataFrame([[50.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [60.0] * 16])
    assert an.hour_hit_rate(actual, forecast, 4, cheapest=True).iloc[0] == 1.0
    wrong = pd.DataFrame([[60.0] * 20 + [1.0, 2.0, 3.0, 4.0]])
    assert an.hour_hit_rate(actual, wrong, 4, cheapest=True).iloc[0] == 0.0


def test_band_covers_and_clips_at_zero():
    index = _hours(10)
    forecast = pd.Series(5.0, index=index)
    actual = forecast + np.tile(np.linspace(-10, 10, 10), 24)[np.argsort(np.repeat(np.arange(24), 10), kind="stable")]
    band = an.error_band(actual, forecast)
    banded = an.apply_band(forecast, band)
    assert (banded["low"] >= 0).all()
    assert 0.7 <= an.band_coverage(actual, banded) <= 0.9


def test_day_type_thresholds():
    evening = np.zeros(24); evening[17:23] = 100
    deep = evening.copy(); deep[10:16] = 10
    dip = evening.copy(); dip[10:16] = 50
    flat = evening.copy(); flat[10:16] = 90
    types = an.day_type(pd.DataFrame([deep, dip, flat]))
    assert types.tolist() == ["deep midday dip", "midday dip", "flat day"]
