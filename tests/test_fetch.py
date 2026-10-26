"""Tests for the month logic in src/fetch.py (no API calls)."""

import pandas as pd

from src.fetch import is_complete, month_ranges


def test_month_ranges_full_months_in_the_past():
    months = list(month_ranges("2025-01", "2025-02", today=pd.Timestamp("2026-10-26")))
    assert months == [("2025-01", "2025-01-01", "2025-01-31"), ("2025-02", "2025-02-01", "2025-02-28")]


def test_month_ranges_current_month_stops_at_yesterday():
    months = list(month_ranges("2026-10", "2026-10", today=pd.Timestamp("2026-10-26 11:00")))
    assert months == [("2026-10", "2026-10-01", "2026-10-25")]


def test_month_ranges_skips_months_that_have_not_started():
    assert list(month_ranges("2026-11", "2026-11", today=pd.Timestamp("2026-10-26"))) == []


def test_is_complete_detects_partial_month(tmp_path):
    path = tmp_path / "2026-10.csv"
    hours = pd.date_range("2026-10-01", "2026-10-21 23:00", freq="h", tz="Europe/Istanbul")
    pd.DataFrame({"date": hours.strftime("%Y-%m-%dT%H:%M:%S%z"), "price": 1.0}).to_csv(path, index=False)
    assert not is_complete(path, "2026-10-25")  # stopped on the 21st -> must be fetched again
    assert is_complete(path, "2026-10-21")


def test_is_complete_missing_file(tmp_path):
    assert not is_complete(tmp_path / "missing.csv", "2026-10-25")
