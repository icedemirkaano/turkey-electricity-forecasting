"""Tests for src/retraining.py: right hours, and no data from a window (or later) in its training."""

import numpy as np
import pandas as pd

from src.retraining import retrained_forecasts

TZ = "Europe/Istanbul"
SMALL = {"n_estimators": 20, "max_depth": 3}


def data():
    index = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h", tz=TZ)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=len(index)), "hour": index.hour}, index=index)
    y = pd.Series(X["a"] * 10 + X["hour"], index=index)
    return X, y


def test_forecasts_cover_exactly_the_period():
    X, y = data()
    forecasts = retrained_forecasts(X, y, "2024-03-01", "2024-03-20", "week", **SMALL)
    assert forecasts.index.equals(X.loc["2024-03-01":"2024-03-20 23:00"].index)


def test_a_window_never_sees_its_own_targets():
    X, y = data()
    changed = y.copy()
    changed.loc["2024-03-11":] = 999.0   # from Monday 11 March on: only the windows from that day may change
    before = retrained_forecasts(X, y, "2024-03-01", "2024-03-20", "week", **SMALL)
    after = retrained_forecasts(X, changed, "2024-03-01", "2024-03-20", "week", **SMALL)
    assert before.loc[:"2024-03-17 23:00"].equals(after.loc[:"2024-03-17 23:00"])   # windows up to 17 March
    assert not before.loc["2024-03-18":].equals(after.loc["2024-03-18":])            # trained after the change
