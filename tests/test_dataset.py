"""Tests for src/dataset.py. The last two run on the real data and are skipped if it is not downloaded."""

import pandas as pd
import pytest

from src.config import RAW_DIR
from src.dataset import COLUMN_AVAILABILITY, build_ptf_hourly, price_cap_per_hour

TZ = "Europe/Istanbul"
HAS_DATA = (RAW_DIR / "mcp").exists()


def test_price_cap_follows_the_schedule():
    hours = pd.DatetimeIndex([pd.Timestamp(t, tz=TZ) for t in [
        "2023-01-15 10:00", "2023-03-31 23:00", "2023-04-01 00:00", "2023-07-03 23:00",
        "2023-07-04 00:00", "2025-04-04 23:00", "2025-04-05 00:00", "2026-10-01 12:00"]])
    assert price_cap_per_hour(hours).tolist() == [4200, 3050, 2600, 2600, 2700, 3000, 3400, 4500]


def test_every_availability_label_is_valid():
    assert set(COLUMN_AVAILABILITY.values()) <= {"target", "known_ahead", "past_only", "evaluation_only"}
    assert list(COLUMN_AVAILABILITY.values()).count("target") == 1


@pytest.fixture(scope="module")
def table():
    if not HAS_DATA:
        pytest.skip("raw data not downloaded")
    return build_ptf_hourly()[0]


def test_one_row_per_hour_without_gaps(table):
    assert table.index.is_unique and table.index.is_monotonic_increasing
    expected = pd.date_range(table.index.min(), table.index.max(), freq="h")
    assert table.index.equals(expected)


def test_model_inputs_have_no_missing_values_and_rules_hold(table):
    inputs = [c for c, a in COLUMN_AVAILABILITY.items() if a in ("target", "known_ahead")]
    assert table[inputs].notna().all().all()
    gen_cols = [c for c in table.columns if c.startswith("gen_")]
    assert (table[gen_cols] >= 0).all().all()
    assert (table["ptf_try"] <= table["price_cap_try"]).all()
