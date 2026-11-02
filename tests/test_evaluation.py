"""Tests for src/evaluation.py: fold boundaries, no overlap with the test period, metric values."""

import numpy as np
import pandas as pd

from src.evaluation import (DEMO_START, TEST_PERIOD, TRAIN_END, WALK_FORWARD_FOLDS, mae, rmse, scores,
                            scores_by_month, walk_forward_splits)

TZ = "Europe/Istanbul"


def hourly(start="2024-01-01", end="2026-10-31 23:00"):
    index = pd.date_range(start, end, freq="h", tz=TZ)
    return pd.DataFrame({"a": np.arange(len(index))}, index=index), pd.Series(np.arange(len(index), dtype=float), index=index)


def test_folds_train_only_on_the_past_and_cover_2025():
    X, y = hourly()
    seen = []
    for name, X_tr, y_tr, X_val, y_val in walk_forward_splits(X, y):
        assert X_tr.index.max() < X_val.index.min()                     # no future in training
        assert X_tr.index.min() == pd.Timestamp("2024-01-01", tz=TZ)    # expanding window from the model start
        assert X_val.index.max() <= pd.Timestamp(f"{TRAIN_END} 23:00", tz=TZ)  # validation never reaches the test
        seen.append(X_val.index)
    covered = seen[0].append(seen[1:])
    assert covered.is_unique
    assert len(covered) == len(pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", tz=TZ))


def test_periods_are_in_order():
    assert WALK_FORWARD_FOLDS[-1][2] == TRAIN_END
    assert pd.Timestamp(TRAIN_END) < pd.Timestamp(TEST_PERIOD[0]) <= pd.Timestamp(TEST_PERIOD[1]) < pd.Timestamp(DEMO_START)


def test_metrics_on_a_known_example():
    y_true, y_pred = [2000, 1500, 3000], [2300, 1400, 3000]   # the example from the project glossary
    assert round(mae(y_true, y_pred), 1) == 133.3
    assert round(rmse(y_true, y_pred), 1) == 182.6


def test_scores_split_floor_and_cap_hours():
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz=TZ)
    y_true = pd.Series([0.0, 50.0, 100.0, 60.0], index=index)
    y_pred = pd.Series([10.0, 50.0, 80.0, 60.0], index=index)
    flags = pd.DataFrame({"is_price_floor": [True, False, False, False],
                          "is_price_cap": [False, False, True, False]}, index=index)
    s = scores(y_true, y_pred, flags)
    assert s["mae"] == 7.5 and s["mae_floor_hours"] == 10 and s["mae_cap_hours"] == 20
    assert s["floor_hours"] == 1 and s["cap_hours"] == 1


def test_scores_by_month():
    index = pd.date_range("2026-01-31 23:00", periods=2, freq="h", tz=TZ)   # one hour in Jan, one in Feb
    by_month = scores_by_month(pd.Series([1.0, 1.0], index=index), pd.Series([3.0, 0.0], index=index))
    assert by_month.loc["2026-01", "mae"] == 2 and by_month.loc["2026-02", "mae"] == 1


def test_weekly_retrain_windows_cover_the_period_once():
    from src.evaluation import retrain_windows
    windows = retrain_windows("2026-01-01", "2026-09-30", "week")
    assert windows[0] == (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-04"))   # Thursday .. Sunday
    assert all(s.dayofweek == 0 for s, _ in windows[1:])                          # later ones start on Monday
    days = sum((e - s).days + 1 for s, e in windows)
    assert days == 273 and len(windows) == 40
