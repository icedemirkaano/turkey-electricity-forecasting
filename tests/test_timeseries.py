"""Tests for src/timeseries.py: forecasts land on the right hours, never use the forecast day's own prices,
and pick up a simple daily pattern."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import mae
from src.timeseries import day_ahead_forecasts, fit, hourly_inputs

TZ = "Europe/Istanbul"
SMALL = {  # small seasonal periods keep the tests fast
    "ets": {"kind": "ets", "seasonal": 24},
    "sarimax": {"kind": "sarimax", "order": (1, 0, 0), "seasonal_order": (0, 1, 1, 24)},
    "prophet": {"kind": "prophet", "yearly": False},
}


def table(seed=0):
    index = pd.date_range("2024-01-01", "2024-03-10 23:00", freq="h", tz=TZ)
    rng = np.random.default_rng(seed)
    daily = 50 + 20 * np.sin(2 * np.pi * index.hour / 24)            # a clear daily pattern
    return pd.DataFrame({"ptf_usd": daily + rng.normal(0, 2, len(index)),
                         "load_forecast": 30000 + rng.normal(0, 100, len(index)),
                         "wind_forecast": np.full(len(index), 3000.0)}, index=index)


@pytest.mark.parametrize("kind", list(SMALL))
def test_forecasts_cover_the_right_hours_and_learn_the_daily_pattern(kind):
    y, exog = hourly_inputs(table())
    results, _ = fit(SMALL[kind], y, exog, "2024-02-29")
    pred = day_ahead_forecasts(results, SMALL[kind], y, exog, "2024-03-01", "2024-03-10")
    assert pred.index.equals(y.loc["2024-03-01":"2024-03-10"].index)
    flat = np.full(len(pred), y.loc[:"2024-02-29"].mean())
    assert mae(y.loc[pred.index], pred) < 0.5 * mae(y.loc[pred.index], flat)   # far better than a flat line


def test_holt_winters_daily_rerun_matches_the_fitted_model():
    # Re-running the recursion with the estimated parameters must reproduce statsmodels' own forecast
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y, exog = hourly_inputs(table())
    state, _ = fit(SMALL["ets"], y, exog, "2024-02-29")
    direct = ExponentialSmoothing(state.history, seasonal="add", seasonal_periods=24,
                                  initialization_method="heuristic").fit().forecast(24)
    assert np.allclose(state.forecast(24), direct, atol=1e-6)


def test_sarimax_is_estimated_on_the_last_weeks_only():
    from src.timeseries import SARIMAX_WINDOW_DAYS
    y, exog = hourly_inputs(table())
    state, _ = fit(SMALL["sarimax"], y, exog, "2024-02-29")      # 60 days of training data
    assert state.nobs == SARIMAX_WINDOW_DAYS * 24


def test_fast_sarimax_update_matches_the_full_one():
    # extend (only the new day) must give the same forecasts as append (whole history again)
    y, exog = hourly_inputs(table())
    results, _ = fit(SMALL["sarimax"], y, exog, "2024-02-29")
    fast = day_ahead_forecasts(results, SMALL["sarimax"], y, exog, "2024-03-01", "2024-03-05", update="extend")
    full = day_ahead_forecasts(results, SMALL["sarimax"], y, exog, "2024-03-01", "2024-03-05", update="append")
    assert np.allclose(fast, full, atol=1e-6)


@pytest.mark.parametrize("kind", list(SMALL))
def test_a_day_is_forecast_without_its_own_prices(kind):
    t = table()
    y, exog = hourly_inputs(t)
    results, _ = fit(SMALL[kind], y, exog, "2024-02-29")
    before = day_ahead_forecasts(results, SMALL[kind], y, exog, "2024-03-01", "2024-03-05")
    changed = y.copy()
    changed.loc["2024-03-05"] += 1000                                    # change the last day's actual prices
    after = day_ahead_forecasts(results, SMALL[kind], changed, exog, "2024-03-01", "2024-03-05")
    assert np.allclose(before, after)                                    # its forecast must not move


@pytest.mark.parametrize("columns", [[], ["load_forecast", "wind_forecast"]])
def test_sarimax_with_its_own_inputs(columns):
    # "exog" in the spec chooses the inputs: none, or several columns of the exog frame
    t = table()
    y, _ = hourly_inputs(t)
    exog = t[["load_forecast", "wind_forecast"]] / 1000
    spec = {**SMALL["sarimax"], "exog": columns}
    state, _ = fit(spec, y, exog, "2024-02-29")
    assert state.model.k_exog == len(columns)
    pred = day_ahead_forecasts(state, spec, y, exog, "2024-03-01", "2024-03-03")
    assert pred.index.equals(y.loc["2024-03-01":"2024-03-03"].index) and np.isfinite(pred).all()
