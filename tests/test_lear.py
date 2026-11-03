"""Tests for src/lear.py: 247 inputs per day, no information from the target day's prices, scaling round trip."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import mae
from src.lear import LEAR, N_FEATURES, InvariantScaler, day_inputs

TZ = "Europe/Istanbul"


def table(seed=0):
    index = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz=TZ)
    rng = np.random.default_rng(seed)
    load = 30000 + 3000 * np.sin(2 * np.pi * index.hour / 24) + rng.normal(0, 300, len(index))
    price = 50 + (load - 30000) / 100 + rng.normal(0, 2, len(index))   # price follows the load forecast
    return pd.DataFrame({"ptf_usd": price, "load_forecast": load,
                         "wind_forecast": rng.uniform(1000, 5000, len(index))}, index=index)


def test_247_inputs_per_day():
    X, Y = day_inputs(table())
    assert X.shape[1] == N_FEATURES == 247 and Y.shape[1] == 24
    assert X.index[0] == pd.Timestamp("2024-01-08", tz=TZ)   # first day with a T-7 price


def test_inputs_never_contain_the_target_day_prices():
    t = table()
    changed = t.copy()
    changed.loc["2024-03-10", "ptf_usd"] = 999.0
    X, _ = day_inputs(t)
    X2, _ = day_inputs(changed)
    day = pd.Timestamp("2024-03-10", tz=TZ)
    assert X.loc[day].equals(X2.loc[day])                                     # its own prices: not an input
    assert not X.loc[day + pd.Timedelta(days=1)].equals(X2.loc[day + pd.Timedelta(days=1)])   # next day: T-1


def test_scaler_round_trip():
    values = np.random.default_rng(1).normal(50, 20, size=(100, 3))
    scaler = InvariantScaler().fit(values)
    assert np.allclose(scaler.inverse(scaler.transform(values)), values)


def test_learns_the_load_effect():
    X, Y = day_inputs(table())
    train, test = X.index < pd.Timestamp("2024-12-01", tz=TZ), X.index >= pd.Timestamp("2024-12-01", tz=TZ)
    model = LEAR().fit(X[train], Y[train])
    pred = model.predict(X[test], TZ)
    actual = pd.Series(Y[test].to_numpy().ravel(), index=pred.index)
    assert pred.index[0] == pd.Timestamp("2024-12-01 00:00", tz=TZ) and len(pred) == 24 * test.sum()
    assert mae(actual, pred) < 0.3 * mae(actual, np.full(len(actual), Y[train].to_numpy().mean()))
