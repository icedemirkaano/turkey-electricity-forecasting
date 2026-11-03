"""Round-2 inputs (temperature, hydro share): known at forecast time, and round-1 inputs unchanged."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_SPEC, FEATURE_SPEC_V2, build_features, check_spec

TZ = "Europe/Istanbul"


def table(seed=0):
    index = pd.date_range("2023-12-01", "2024-02-29 23:00", freq="h", tz=TZ)
    rng = np.random.default_rng(seed)
    n = len(index)
    return pd.DataFrame({
        "ptf_usd": rng.uniform(0, 100, n), "load_forecast": rng.uniform(30000, 40000, n),
        "wind_forecast": rng.uniform(1000, 5000, n), "price_cap_try": np.full(n, 3000.0),
        "gen_solar": rng.uniform(0, 500, n), "gen_total": np.full(n, 35000.0),
        "gen_dammed_hydro": rng.uniform(3000, 9000, n), "gen_river": rng.uniform(500, 2000, n),
        "temperature_forecast": rng.uniform(-5, 35, n)}, index=index)


def test_v2_spec_passes_the_leakage_check_and_adds_five_inputs():
    check_spec(FEATURE_SPEC_V2)
    X1, _ = build_features(table(), FEATURE_SPEC)
    X2, _ = build_features(table(), FEATURE_SPEC_V2)
    assert X2.shape[1] == X1.shape[1] + 5
    assert X2[X1.columns].equals(X1)          # round-1 inputs are exactly the same


@pytest.mark.parametrize("bad", [("hydro_share", 24), ("hydro_share_7d", 47)])
def test_hydro_share_needs_48_hours(bad):
    with pytest.raises(ValueError):
        check_spec(FEATURE_SPEC_V2 + [bad])


def test_hydro_inputs_use_nothing_from_the_last_48_hours():
    t = table()
    changed = t.copy()
    hour = pd.Timestamp("2024-02-10 12:00", tz=TZ)
    changed.loc[hour - pd.Timedelta(hours=47):, "gen_dammed_hydro"] = 1e6   # the 47 hours before, and later
    X, _ = build_features(t, FEATURE_SPEC_V2)
    X2, _ = build_features(changed, FEATURE_SPEC_V2)
    assert X.loc[hour, ["hydro_share_lag48", "hydro_share_7d_lag48"]].equals(
        X2.loc[hour, ["hydro_share_lag48", "hydro_share_7d_lag48"]])


def test_heating_and_cooling_degrees():
    X, _ = build_features(table(), FEATURE_SPEC_V2)
    t = X["temperature_forecast"]
    assert (X["temperature_heating"] == (18 - t).clip(lower=0)).all()
    assert ((X["temperature_heating"] > 0) & (X["temperature_cooling"] > 0)).sum() == 0
