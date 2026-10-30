"""Tests for src/models.py on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import mae
from src.models import MODELS, feature_importance, fit, make_model, predict

TZ = "Europe/Istanbul"


def synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz=TZ)
    X = pd.DataFrame({"hour": index.hour, "day_of_week": index.dayofweek, "month": index.month,
                      "signal": rng.normal(size=n), "noise": rng.normal(size=n)}, index=index)
    y = pd.Series(50 + 10 * X["signal"] + 5 * (X["hour"] >= 17), index=index)  # depends on signal and hour only
    return X, y


@pytest.mark.parametrize("name", [m for m in MODELS if m != "dummy (median)"])
def test_every_model_beats_the_dummy(name):
    X, y = synthetic()
    train, val = slice(0, 1500), slice(1500, None)
    dummy = predict(fit("dummy (median)", X.iloc[train], y.iloc[train]), X.iloc[val])
    pred = predict(fit(name, X.iloc[train], y.iloc[train]), X.iloc[val])
    assert pred.index.equals(X.index[val])
    assert mae(y.iloc[val], pred) < mae(y.iloc[val], dummy)


def test_params_override_starting_settings():
    assert make_model("xgboost", max_depth=3).get_params()["max_depth"] == 3
    assert make_model("xgboost").get_params()["max_depth"] == 6   # catalogue entry itself is unchanged


def test_importance_points_to_the_real_signal():
    X, y = synthetic()
    importance = feature_importance(fit("xgboost", X, y))
    assert importance.index[0] == "signal" and abs(importance.sum() - 1) < 1e-9
    assert importance.get("noise", 0) < importance["signal"]
