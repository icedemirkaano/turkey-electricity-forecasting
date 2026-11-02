"""Tests for src/sequences.py and src/gru.py: sample shapes, no future information, prediction alignment."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pytest

from src.sequences import SEQ_COLUMNS, SEQ_HOURS, day_samples

TZ = "Europe/Istanbul"


def table(start="2023-12-20", end="2024-02-29 23:00"):
    """Synthetic hourly table where every value is its own hour number, so positions can be checked."""
    index = pd.date_range(start, end, freq="h", tz=TZ)
    n = np.arange(len(index), dtype=float)
    return pd.DataFrame({"ptf_usd": n, "load_forecast": n + 1000, "wind_forecast": np.full(len(n), 100.0),
                         "gen_solar": n, "price_cap_try": np.full(len(n), 3000.0)}, index=index), index


def test_shapes_and_targets():
    t, _ = table()
    s = day_samples(t, "2024-01-01", "2024-01-31")
    assert s.seq.shape == (31, SEQ_HOURS, len(SEQ_COLUMNS)) and s.target.shape == (31, 24)
    assert s.future.shape[:2] == (31, 24)
    expected = t.loc[s.days[0]: s.days[0] + pd.Timedelta(hours=23), "ptf_usd"].to_numpy()
    assert np.array_equal(s.target[0], expected)                       # targets are the 24 prices of D+1
    assert s.target_series().index.equals(t.loc["2024-01-01":"2024-01-31"].index)


def test_sequence_uses_only_information_known_on_day_d():
    t, index = table()
    s = day_samples(t, "2024-01-10", "2024-01-10")
    target_start = index.get_loc(pd.Timestamp("2024-01-10", tz=TZ))   # value = position of the hour
    ptf, solar = s.seq[0, :, 0], s.seq[0, :, 2]
    assert ptf[-1] == target_start - 1          # PTF up to D 23:00 (published on D-1 at 14:00)
    assert solar[-1] == target_start - 25       # realised solar only up to D-1 23:00
    assert ptf.max() < target_start             # nothing from D+1 in the past sequence


def test_days_with_missing_values_are_skipped():
    t, _ = table()
    t.loc["2024-01-05 10:00", "ptf_usd"] = np.nan
    s = day_samples(t, "2024-01-01", "2024-01-31")
    assert pd.Timestamp("2024-01-05", tz=TZ) not in s.days            # missing target hour
    assert pd.Timestamp("2024-01-06", tz=TZ) not in s.days            # missing value inside the past week
    assert pd.Timestamp("2024-01-13", tz=TZ) in s.days                # past week no longer reaches the gap


@pytest.mark.parametrize("seq_hours", [72, 336])
def test_other_history_lengths_still_end_at_day_d(seq_hours):
    t, index = table()
    s = day_samples(t, "2024-01-20", "2024-01-20", seq_hours)
    target_start = index.get_loc(pd.Timestamp("2024-01-20", tz=TZ))
    assert s.seq.shape[1] == seq_hours and s.seq[0, -1, 0] == target_start - 1


def test_two_layer_gru_trains():
    from src.gru import GRU_RESIDUAL_SETTINGS, fit_gru, predict_gru
    t, _ = table()
    train, val = day_samples(t, "2024-01-01", "2024-02-15"), day_samples(t, "2024-02-16", "2024-02-29")
    bundle = fit_gru(train, seed=0, settings={**GRU_RESIDUAL_SETTINGS, "max_epochs": 2, "units": 8, "layers": 2})
    assert np.isfinite(predict_gru(bundle, val)).all()


def test_baseline_part_is_the_same_hour_yesterday():
    from src.gru import baseline_part
    t, _ = table()
    s = day_samples(t, "2024-01-01", "2024-01-31")
    assert np.array_equal(baseline_part(s), s.target - 24)   # synthetic PTF = hour number, so yesterday = -24


@pytest.mark.parametrize("residual", [False, True])
def test_gru_trains_and_predicts_on_the_right_hours(residual):
    from src.gru import GRU_SETTINGS, fit_gru, predict_gru
    t, _ = table()
    train, val = day_samples(t, "2024-01-01", "2024-02-15"), day_samples(t, "2024-02-16", "2024-02-29")
    bundle = fit_gru(train, seed=0, settings={**GRU_SETTINGS, "max_epochs": 2, "units": 8, "residual": residual})
    pred = predict_gru(bundle, val)
    assert pred.index.equals(val.target_series().index) and np.isfinite(pred).all()
