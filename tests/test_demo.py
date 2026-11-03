"""Tests for src/demo.py on a small synthetic demo table."""

import numpy as np
import pandas as pd
import pytest

from src.demo import BASELINE, FORECAST, days, one_day, outcome, period_accuracy, period_plant, plan, profit_by_cost

TZ = "Europe/Istanbul"


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


def test_plan_uses_only_the_forecast():
    day = first_day()
    day["actual"] = 0.0                      # the plan must not change when the actual prices change
    result = plan(day, cost=60, min_hours=4, start_cost=10)
    assert result["hours"] == list(range(15, 24))   # forecast 62, 66, ... > 60 from hour 15 on
    assert result["expected_profit"] == pytest.approx(sum(4 * h + 2 - 60 for h in range(15, 24)) - 10)


def test_plan_stays_off_when_no_block_pays():
    result = plan(first_day(), cost=200, min_hours=4, start_cost=10)
    assert result["block"] is None and result["hours"] == [] and result["expected_profit"] == 0


def test_outcome_uses_actual_prices():
    result = outcome(first_day(), cost=60, min_hours=4, start_cost=10)
    perfect = sum(4 * h - 60 for h in range(16, 24)) - 10
    assert result["perfect foresight"]["profit"] == pytest.approx(perfect)
    assert result["model"]["profit"] == pytest.approx(perfect - 0)   # hour 15 earns exactly 0 at actual prices
    assert result["same hour yesterday"]["block"] == (0, 7)           # reversed day: runs the cheap morning
    assert result["same hour yesterday"]["profit"] < 0


def test_period_summaries():
    demo = demo_table()
    accuracy = period_accuracy(demo)
    assert accuracy["days"] == 2 and accuracy["mae"][FORECAST] == pytest.approx(2)
    plant = period_plant(demo, cost=60, min_hours=4, start_cost=10)
    assert plant.loc["perfect foresight", "captured_%"] == pytest.approx(100)
    by_cost = profit_by_cost(demo, [40, 60], min_hours=4, start_cost=10)
    assert list(by_cost.index) == [40, 60] and (by_cost["perfect foresight"] >= by_cost["model"]).all()
