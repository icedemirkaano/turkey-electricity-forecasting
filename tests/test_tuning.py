"""Tests for src/tuning.py: every search space gives settings its model accepts, and a short search runs."""

import numpy as np
import pandas as pd
import pytest

from src.tuning import SEARCH_SPACES, tune

TZ = "Europe/Istanbul"


def tiny_folds(n=600, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz=TZ)
    X = pd.DataFrame({"hour": index.hour, "day_of_week": index.dayofweek, "month": index.month,
                      "signal": rng.normal(size=n)}, index=index)
    y = pd.Series(50 + 10 * X["signal"], index=index)
    # two expanding folds: train on the past, validate on the next 100 hours
    return [(f"f{i}", X.iloc[:end], y.iloc[:end], X.iloc[end:end + 100], y.iloc[end:end + 100])
            for i, end in enumerate([400, 500])]


@pytest.mark.parametrize("name", list(SEARCH_SPACES))
def test_short_search_runs_and_records_fold_scores(name):
    study = tune(name, tiny_folds(), n_trials=2)
    assert np.isfinite(study.best_value)
    assert set(study.best_trial.user_attrs["fold_mae"]) == {"f0", "f1"}
