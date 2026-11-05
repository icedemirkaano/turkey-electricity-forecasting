"""Tests for src/demo.py on a small synthetic demo table."""

import numpy as np
import pandas as pd
import pytest

from src.demo import (BASELINE, FORECAST, day_check, days, hour_ranges, input_change, one_day, outcome, outlook,
                      period_accuracy, period_shifting, validation_shifting)

TZ = "Europe/Istanbul"
NO_BAND = pd.DataFrame({"low": 0.0, "high": 0.0}, index=pd.RangeIndex(24, name="hour"))


def demo_table():
    index = pd.date_range("2026-10-01", periods=48, freq="h", tz=TZ)
    actual = np.tile(np.arange(24, dtype=float), 2) * 4   # price rises through the day: 0, 4, ..., 92
    return pd.DataFrame({"actual": actual, FORECAST: actual + 2, BASELINE: actual[::-1]}, index=index)


def first_day():
    demo = demo_table()
    return one_day(demo, days(demo)[0])


def test_one_day_has_hours_0_to_23():
    day = first_day()
    assert list(day.index) == list(range(24)) and day["actual"].iloc[5] == 20


def test_incomplete_day_is_rejected():
    demo = demo_table().iloc[:-1]
    with pytest.raises(ValueError):
        one_day(demo, days(demo)[1])


def test_hour_ranges():
    assert hour_ranges([13, 10, 11, 12, 18]) == [(10, 14), (18, 19)]
    assert hour_ranges([]) == []


def test_outlook_uses_only_the_forecast():
    day = first_day()
    day["actual"] = 0.0                       # the outlook must not change when the actual prices change
    band = pd.DataFrame({"low": -10.0, "high": 5.0}, index=pd.RangeIndex(24, name="hour"))
    view = outlook(day, band, n_hours=4)
    assert view["cheapest"] == [0, 1, 2, 3] and view["priciest"] == [20, 21, 22, 23]
    assert view["peak"] == pytest.approx(np.mean([4 * h + 2 for h in range(17, 22)]))
    assert view["low"].iloc[0] == 0 and view["high"].iloc[0] == 7   # low edge clipped at the price floor
    assert view["type"] == "midday dip"   # midday (10-15) mean 52 / evening (17-22) mean 80 = 0.65


def test_outcome_pays_actual_prices():
    result = outcome(first_day(), n_hours=4)
    assert result["model"]["hours"] == [0, 1, 2, 3] and result["model"]["paid"] == pytest.approx(0 + 4 + 8 + 12)
    assert result["perfect foresight"]["paid"] == result["model"]["paid"]
    assert result["same hour yesterday"]["hours"] == [20, 21, 22, 23]   # reversed day: picks the priciest hours
    assert result["no shifting"]["paid"] == pytest.approx(4 * np.mean(4 * np.arange(24)))


def test_day_check():
    check = day_check(first_day(), NO_BAND)
    assert check["error"] == pytest.approx(2) and check["inside"] == 0


def test_input_change():
    index = pd.date_range("2026-10-01", periods=48, freq="h", tz=TZ)
    inputs = pd.DataFrame({"load_forecast": [1.0] * 24 + [3.0] * 24}, index=index)
    change = input_change(inputs, days(inputs)[1])
    assert change.loc["load_forecast", "today"] == 1 and change.loc["load_forecast", "tomorrow"] == 3


def test_period_summaries():
    demo = demo_table()
    accuracy = period_accuracy(demo, NO_BAND, n_hours=4)
    assert accuracy["days"] == 2 and accuracy["mae"][FORECAST] == pytest.approx(2)
    assert accuracy["hits"]["model"] == 1 and accuracy["hits"]["baseline"] == 0
    shifting = period_shifting(demo, n_hours=4)
    assert shifting["summary"]["model"]["captured_%"] == pytest.approx(100)
    assert shifting["model_cheaper"] == 2 and shifting["baseline_cheaper"] == 0


def test_validation_shifting_reads_saved_rows():
    rows = pd.DataFrame({"period": "validation", "forecast": ["model", "same hour yesterday"], "n_hours": 4,
                         "price_no_shifting": 56.0, "price_perfect": 33.0, "price_forecast": [37.0, 39.5],
                         "captured_%": [84.0, 73.0], "days": 638, "model cheaper": 355, "equal": 119,
                         "baseline cheaper": 164, "mean gap per day": 10.8})
    result = validation_shifting(rows, 4)
    assert result["captured"] == {"model": 84.0, "same hour yesterday": 73.0}
    assert result["model_cheaper"] == 355 and result["paid"]["same hour yesterday"] == 39.5
