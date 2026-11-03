"""Calculations behind the demo app (app/streamlit_app.py), kept here so they can be tested without Streamlit.

The app reads app/data/demo_forecasts.csv (written by scripts/make_demo_forecasts.py): hourly actual price,
the weekly retrained XGBoost forecast and the same-hour-yesterday baseline, for days no model has seen.
User story: a business with a flexible job (n hours at a given power) plans tomorrow's consumption from the
forecast: run the job in the cheapest hours, avoid the most expensive ones.
"""

from pathlib import Path

import pandas as pd

from src.business import chosen_hours, daily_costs, shifting_summary
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


def extreme_hours(prices: pd.Series, n_hours: int, cheapest=True) -> list[int]:
    """Hours of day (sorted) with the n lowest (or highest) prices; ties: earlier hour first."""
    values = prices if cheapest else -prices
    pick = chosen_hours(values.set_axis(pd.date_range("2000-01-01", periods=24, freq="h")), n_hours).to_numpy()
    return sorted(prices.index[pick])


def plan(day: pd.DataFrame, n_hours: int, power_mw: float) -> dict:
    """What the forecast says on the day before: when to run the job, when to avoid, and the expected saving."""
    run, avoid = extreme_hours(day[FORECAST], n_hours), extreme_hours(day[FORECAST], n_hours, cheapest=False)
    expected, day_mean = day.loc[run, FORECAST].mean(), day[FORECAST].mean()
    return {"run": run, "avoid": avoid, "expected_price": expected, "day_mean": day_mean,
            "expected_saving": (day_mean - expected) * n_hours * power_mw}


def outcome(day: pd.DataFrame, n_hours: int, power_mw: float) -> dict:
    """What actually happened: price paid when following the plan, against no shifting and perfect foresight."""
    run = extreme_hours(day[FORECAST], n_hours)
    paid, day_mean = day.loc[run, "actual"].mean(), day["actual"].mean()
    perfect = day["actual"].nsmallest(n_hours).mean()
    return {"paid": paid, "day_mean": day_mean, "perfect": perfect,
            "saving": (day_mean - paid) * n_hours * power_mw,
            "possible_saving": (day_mean - perfect) * n_hours * power_mw,
            "mae": (day[FORECAST] - day["actual"]).abs().mean()}


def period_accuracy(demo: pd.DataFrame) -> dict:
    """MAE of forecast and baseline over the whole demo period, and on how many days the forecast was better."""
    errors = demo[[FORECAST, BASELINE]].sub(demo["actual"], axis=0).abs()
    daily = errors.groupby(demo.index.date).mean()
    return {"mae": errors.mean().to_dict(), "days": len(daily),
            "days_better": int((daily[FORECAST] < daily[BASELINE]).sum()), "daily": daily}


def period_shifting(demo: pd.DataFrame, n_hours: int) -> pd.DataFrame:
    """Load shifting over the demo period (src/business.py) when hours are chosen by the forecast or by the baseline."""
    return pd.DataFrame({name: shifting_summary(daily_costs(demo["actual"], demo[name], n_hours), n_hours)
                         for name in (FORECAST, BASELINE)}).T
