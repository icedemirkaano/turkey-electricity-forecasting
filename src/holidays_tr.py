"""Official public holidays in Turkey, 2024-2026, entered by hand (as advised by the instructor).

Covers the modelling period only (training starts 2024-01-01); other years are not filled in.

Scope: official holidays only (Law 2429 on National Holidays and General Holidays). Administrative leave
days given to public employees around religious holidays are NOT included (decision of 2026-10-30).
Eves of religious holidays and 28 October are official half-day holidays from 13:00.

Sources:
- Fixed holidays: Law 2429; 15 July added by Law 6752 (Resmi Gazete 29.10.2016, no. 29872).
- Religious holiday dates: published holiday calendars for each year (NTV, CNN Türk), consistent across sources.
"""

import pandas as pd

# Fixed-date holidays: (month, day, name). 28 October is a half day (eve of Republic Day).
FIXED_FULL_DAYS = [
    (1, 1, "New Year's Day"),
    (4, 23, "National Sovereignty and Children's Day"),
    (5, 1, "Labour and Solidarity Day"),
    (5, 19, "Commemoration of Atatürk, Youth and Sports Day"),
    (7, 15, "Democracy and National Unity Day"),
    (8, 30, "Victory Day"),
    (10, 29, "Republic Day"),
]
FIXED_HALF_DAYS = [(10, 28, "Republic Day eve")]

# Religious holidays: eve (half day from 13:00) and full days.
RELIGIOUS = {
    2024: [("Ramadan Feast", "2024-04-09", ["2024-04-10", "2024-04-11", "2024-04-12"]),
           ("Sacrifice Feast", "2024-06-15", ["2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19"])],
    2025: [("Ramadan Feast", "2025-03-29", ["2025-03-30", "2025-03-31", "2025-04-01"]),
           ("Sacrifice Feast", "2025-06-05", ["2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09"])],
    2026: [("Ramadan Feast", "2026-03-19", ["2026-03-20", "2026-03-21", "2026-03-22"]),
           ("Sacrifice Feast", "2026-05-26", ["2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30"])],
}
HALF_DAY_START_HOUR = 13


COVERED_YEARS = sorted(RELIGIOUS)


def holiday_calendar(years=COVERED_YEARS) -> pd.DataFrame:
    """One row per holiday date: date, name, kind ('full' or 'half'). Only covered years are allowed."""
    missing = sorted(set(years) - set(COVERED_YEARS))
    if missing:
        raise ValueError(f"No verified holiday dates for {missing}; covered years: {COVERED_YEARS}")
    rows = []
    for year in years:
        rows += [(pd.Timestamp(year, m, d), name, "full") for m, d, name in FIXED_FULL_DAYS]
        rows += [(pd.Timestamp(year, m, d), name, "half") for m, d, name in FIXED_HALF_DAYS]
        for name, eve, days in RELIGIOUS.get(year, []):
            rows.append((pd.Timestamp(eve), f"{name} eve", "half"))
            rows += [(pd.Timestamp(day), name, "full") for day in days]
    return pd.DataFrame(rows, columns=["date", "name", "kind"]).sort_values("date").reset_index(drop=True)


def holiday_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly flags for a tz-aware index: is_holiday (full-day holiday) and
    is_holiday_eve_afternoon (half-day holiday, hours from 13:00)."""
    calendar = holiday_calendar(sorted(set(index.year)))  # raises for years without verified dates
    days = index.normalize().tz_localize(None)
    full = set(calendar.loc[calendar["kind"] == "full", "date"])
    half = set(calendar.loc[calendar["kind"] == "half", "date"])
    return pd.DataFrame({
        "is_holiday": days.isin(full),
        "is_holiday_eve_afternoon": days.isin(half) & (index.hour >= HALF_DAY_START_HOUR),
    }, index=index)
