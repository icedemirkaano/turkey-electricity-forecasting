"""Tests for src/plant.py (run / stop decision of a power plant)."""

import numpy as np
import pandas as pd
import pytest

from src.plant import best_block, daily_profits, plant_summary

TZ = "Europe/Istanbul"


def test_best_block_takes_the_profitable_hours():
    margins = np.array([-5, -5, 10, 10, 10, -5, -5, -5])
    assert best_block(margins, min_hours=2, start_cost=0) == (2, 4)


def test_best_block_respects_minimum_run_time():
    margins = np.array([-10, 20, -10, -10, -10, -10])
    assert best_block(margins, min_hours=1, start_cost=0) == (1, 1)
    assert best_block(margins, min_hours=3, start_cost=0) is None   # 20 - 10 - 10 = 0 does not beat 0


def test_best_block_stays_off_when_start_cost_is_not_covered():
    margins = np.array([-1, 5, 5, -1])
    assert best_block(margins, min_hours=2, start_cost=9) == (1, 2)
    assert best_block(margins, min_hours=2, start_cost=10) is None


def two_days(values):
    index = pd.date_range("2026-10-05", periods=48, freq="h", tz=TZ)
    return pd.Series(values, index=index, dtype=float)


def test_profit_uses_actual_prices_and_forecast_decisions():
    actual = two_days([50] * 8 + [80] * 8 + [50] * 8 + [40] * 24)
    too_high = two_days([90] * 48)                     # says "run all day" on both days
    days = daily_profits(actual, too_high, cost=60, min_hours=4, start_cost=10)
    assert days["hours_run"].tolist() == [24, 24]
    assert days["profit"].tolist() == [8 * 20 - 16 * 10 - 10, 24 * -20 - 10]
    assert days["loss_hours"].tolist() == [16, 24]


def test_perfect_foresight_is_the_upper_bound():
    rng = np.random.default_rng(0)
    actual = two_days(rng.uniform(0, 100, 48))
    noisy = actual + rng.normal(0, 20, 48)
    table = plant_summary(actual, {"noisy": noisy}, cost=50, min_hours=4, start_cost=20)
    assert table.loc["perfect foresight", "captured_%"] == pytest.approx(100)
    assert table["profit"].max() == table.loc["perfect foresight", "profit"]


def test_day_without_24_hours_is_rejected():
    actual = two_days(range(48)).iloc[:47]
    with pytest.raises(ValueError):
        daily_profits(actual, actual, cost=10, min_hours=1, start_cost=0)
