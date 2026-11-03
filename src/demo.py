"""Calculations behind the demo app (app/streamlit_app.py), kept here so they can be tested without Streamlit.

The app reads app/data/demo_forecasts.csv (written by scripts/make_demo_forecasts.py): hourly actual price,
the weekly retrained XGBoost forecast and the same-hour-yesterday baseline, for days no model has seen.
"""

from pathlib import Path

import pandas as pd

from src.business import chosen_hours
from src.config import ROOT, TIMEZONE

DEMO_FILE = ROOT / "app" / "data" / "demo_forecasts.csv"
FORECAST, BASELINE = "xgboost (weekly retrain)", "same hour yesterday"


def load_demo(path: Path = DEMO_FILE) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame


def days(demo: pd.DataFrame) -> list:
    return sorted(set(demo.index.date))


def one_day(demo: pd.DataFrame, day) -> pd.DataFrame:
    """The 24 hours of `day`, indexed by hour of day (0-23)."""
    part = demo[demo.index.date == day]
    if len(part) != 24:
        raise ValueError(f"{day} does not have 24 hours")
    return part.set_index(part.index.hour.rename("hour"))


def day_errors(day: pd.DataFrame) -> dict:
    """MAE of the forecast and of the baseline on one day (USD/MWh)."""
    return {name: (day[name] - day["actual"]).abs().mean() for name in (FORECAST, BASELINE)}


def extreme_hours(day: pd.DataFrame, n_hours: int, cheapest=True) -> pd.DataFrame:
    """The n cheapest (or most expensive) hours according to the forecast, with their actual prices."""
    forecast = day[FORECAST] if cheapest else -day[FORECAST]
    pick = chosen_hours(forecast.set_axis(pd.date_range("2000-01-01", periods=24, freq="h")), n_hours).to_numpy()
    return day.loc[pick, [FORECAST, "actual"]].rename(columns={FORECAST: "forecast"})


def buying_cost(day: pd.DataFrame, mwh_per_hour: float) -> dict:
    """Cost of buying the same amount in every hour: as forecast on the day before, and as it turned out (USD)."""
    forecast, actual = (day[FORECAST] * mwh_per_hour).sum(), (day["actual"] * mwh_per_hour).sum()
    return {"forecast": forecast, "actual": actual, "difference": forecast - actual}


def shifting_result(day: pd.DataFrame, n_hours: int) -> dict:
    """Average price paid for a flexible job of n hours: no shifting, following the forecast, perfect foresight."""
    cheapest = extreme_hours(day, n_hours)
    return {"no shifting": day["actual"].mean(), "following the forecast": cheapest["actual"].mean(),
            "perfect foresight": day["actual"].nsmallest(n_hours).mean()}


def daily_errors(demo: pd.DataFrame) -> pd.DataFrame:
    """MAE of forecast and baseline for every day of the demo period."""
    errors = demo[[FORECAST, BASELINE]].sub(demo["actual"], axis=0).abs()
    return errors.groupby(demo.index.date).mean()
