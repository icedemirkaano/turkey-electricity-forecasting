"""Tests for src/demo.py on a small synthetic demo table."""

import numpy as np
import pandas as pd
import pytest

from src.demo import BASELINE, FORECAST, days, extreme_hours, one_day, outcome, period_accuracy, period_shifting, \
    plan

TZ = "Europe/Istanbul"


def demo_table():
    index = pd.date_range("2026-10-01", periods=48, freq="h", tz=TZ)
    actual = np.tile(np.arange(24, dtype=float), 2)   # price = hour of day
    return pd.DataFrame({"actual": actual, FORECAST: actual + 2, BASELINE: actual[::-1]}, index=index)


def first_day():
    demo = demo_table()
    return one_day(demo, days(demo)[0])


def test_one_day_has_hours_0_to_23():
    day = first_day()
    assert list(day.index) == list(range(24)) and day["actual"].iloc[5] == 5


def test_incomplete_day_is_rejected():
    demo = demo_table().iloc[:-1]
    with pytest.raises(ValueError):
        one_day(demo, days(demo)[1])


def test_cheapest_and_most_expensive_hours():
    prices = first_day()["actual"]
    assert extreme_hours(prices, 3) == [0, 1, 2]
    assert extreme_hours(prices, 3, cheapest=False) == [21, 22, 23]


def test_plan_uses_only_the_forecast():
    day = first_day()
    day["actual"] = 0.0                      # the plan must not change when the actual prices change
    result = plan(day, n_hours=4, power_mw=2)
    assert result["run"] == [0, 1, 2, 3] and result["avoid"] == [20, 21, 22, 23]
    assert result["expected_price"] == pytest.approx(3.5)                     # forecast = hour + 2
    assert result["expected_saving"] == pytest.approx((13.5 - 3.5) * 4 * 2)  # (day mean - plan) x hours x MW


def test_outcome_with_a_correctly_ordered_forecast_is_perfect():
    result = outcome(first_day(), n_hours=4, power_mw=1)
    assert result["paid"] == result["perfect"] == pytest.approx(1.5)
    assert result["saving"] == result["possible_saving"] and result["mae"] == pytest.approx(2)


def test_period_summaries():
    demo = demo_table()
    accuracy = period_accuracy(demo)
    assert accuracy["days"] == 2 and accuracy["mae"][FORECAST] == pytest.approx(2)
    assert period_shifting(demo, 4).loc[FORECAST, "captured_%"] == pytest.approx(100)
