"""Tests for src/demo.py on a small synthetic demo table."""

import numpy as np
import pandas as pd
import pytest

from src.demo import BASELINE, FORECAST, buying_cost, day_errors, daily_errors, days, extreme_hours, one_day, \
    shifting_result

TZ = "Europe/Istanbul"


def demo_table():
    index = pd.date_range("2026-10-01", periods=48, freq="h", tz=TZ)
    actual = np.tile(np.arange(24, dtype=float), 2)   # price = hour of day
    return pd.DataFrame({"actual": actual, FORECAST: actual + 2, BASELINE: actual[::-1]}, index=index)


def test_one_day_has_hours_0_to_23():
    demo = demo_table()
    day = one_day(demo, days(demo)[1])
    assert list(day.index) == list(range(24)) and day["actual"].iloc[5] == 5


def test_incomplete_day_is_rejected():
    demo = demo_table().iloc[:-1]
    with pytest.raises(ValueError):
        one_day(demo, days(demo)[1])


def test_errors_and_costs():
    day = one_day(demo_table(), days(demo_table())[0])
    assert day_errors(day)[FORECAST] == pytest.approx(2)
    cost = buying_cost(day, mwh_per_hour=10)
    assert cost["actual"] == pytest.approx(10 * sum(range(24))) and cost["difference"] == pytest.approx(10 * 2 * 24)


def test_cheapest_and_most_expensive_hours():
    day = one_day(demo_table(), days(demo_table())[0])
    assert list(extreme_hours(day, 3).index) == [0, 1, 2]
    assert list(extreme_hours(day, 3, cheapest=False).index) == [21, 22, 23]


def test_shifting_with_a_correctly_ordered_forecast_is_perfect():
    day = one_day(demo_table(), days(demo_table())[0])
    result = shifting_result(day, 4)
    assert result["following the forecast"] == result["perfect foresight"] == pytest.approx(1.5)


def test_daily_errors_one_row_per_day():
    assert len(daily_errors(demo_table())) == 2
