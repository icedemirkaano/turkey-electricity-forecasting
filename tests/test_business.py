"""Tests for src/business.py: hour choice, costs and the captured share."""

import numpy as np
import pandas as pd
import pytest

from src.business import chosen_hours, daily_costs, shifting_summary

TZ = "Europe/Istanbul"


def prices(days=3, seed=0):
    index = pd.date_range("2026-01-01", periods=24 * days, freq="h", tz=TZ)
    return pd.Series(np.random.default_rng(seed).uniform(0, 100, len(index)), index=index)


def test_picks_n_hours_every_day():
    pick = chosen_hours(prices(), 4)
    assert (pick.groupby(pick.index.normalize()).sum() == 4).all()


def test_perfect_forecast_captures_everything():
    actual = prices()
    summary = shifting_summary(daily_costs(actual, actual, 4), 4)
    assert summary["captured_%"] == pytest.approx(100)
    assert summary["price_forecast"] == pytest.approx(summary["price_perfect"])


def test_only_the_order_matters():
    actual = prices()
    biased = daily_costs(actual, actual + 50, 4)   # too high by the same amount in every hour
    assert biased["forecast"].equals(biased["perfect"])


def test_reversed_forecast_is_worse_than_not_shifting():
    actual = prices()
    summary = shifting_summary(daily_costs(actual, -actual, 4), 4)   # cheapest forecast = most expensive hours
    assert summary["captured_%"] < 0 and summary["days_worse_than_no_shifting"] == 3


def test_flat_forecast_picks_the_first_hours():
    actual = prices(days=1)
    pick = chosen_hours(pd.Series(1.0, index=actual.index), 3)
    assert list(pick[pick].index.hour) == [0, 1, 2]


def test_incomplete_day_is_rejected():
    actual = prices().iloc[:-1]
    with pytest.raises(ValueError):
        daily_costs(actual, actual, 4)
