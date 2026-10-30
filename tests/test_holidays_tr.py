"""Tests for the hand-entered holiday calendar."""

import pandas as pd
import pytest

from src.holidays_tr import holiday_calendar, holiday_flags

TZ = "Europe/Istanbul"


def test_calendar_counts_per_year():
    cal = holiday_calendar([2025])
    assert (cal["kind"] == "full").sum() == 7 + 3 + 4  # fixed + Ramadan + Sacrifice
    assert (cal["kind"] == "half").sum() == 3          # 28 Oct + two eves


def test_flags_full_day_and_eve_afternoon():
    hours = pd.DatetimeIndex([pd.Timestamp(t, tz=TZ) for t in [
        "2026-03-19 12:00",  # Ramadan eve, before 13:00 -> working hours
        "2026-03-19 13:00",  # Ramadan eve, afternoon -> half-day holiday
        "2026-03-20 08:00",  # Ramadan Feast day 1 -> holiday
        "2026-10-28 14:00",  # Republic Day eve afternoon
        "2026-07-15 10:00",  # Democracy and National Unity Day
        "2026-05-25 10:00",  # administrative leave only -> NOT a holiday
    ]])
    flags = holiday_flags(hours)
    assert flags["is_holiday"].tolist() == [False, False, True, False, True, False]
    assert flags["is_holiday_eve_afternoon"].tolist() == [False, True, False, True, False, False]


def test_unverified_year_is_rejected():
    with pytest.raises(ValueError):
        holiday_calendar([2023])


def test_no_date_is_both_full_and_half():
    cal = holiday_calendar()
    assert not cal.duplicated("date").any()
