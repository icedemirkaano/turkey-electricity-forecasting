"""Calculations behind the demo app (app/streamlit_app.py), kept here so they can be tested without Streamlit.

The app reads app/data/demo_forecasts.csv (written by scripts/make_demo_forecasts.py): hourly actual price,
the round-2 model's forecast (weekly retrained) and the same-hour-yesterday baseline, for days no choice used.
User story: the operator of a power plant decides on day D, before the 12:30 offer deadline and before the prices
are known, whether to prepare the plant for D+1 and which block of hours to offer (src/plant.py).
"""

from pathlib import Path

import pandas as pd

from src.config import ROOT, TIMEZONE
from src.plant import best_block, plant_summary

DEMO_FILE = ROOT / "app" / "data" / "demo_forecasts.csv"
FORECAST, BASELINE = "model (weekly retrain)", "same hour yesterday"


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


def block_hours(block: tuple[int, int] | None) -> list[int]:
    return [] if block is None else list(range(block[0], block[1] + 1))


def realised_profit(day: pd.DataFrame, block: tuple[int, int] | None, cost: float, start_cost: float) -> float:
    """Profit (USD per MW) of running `block` at the ACTUAL prices of the day; 0 when the plant stays off."""
    hours = block_hours(block)
    return float((day.loc[hours, "actual"] - cost).sum() - start_cost) if hours else 0.0


def plan(day: pd.DataFrame, cost: float, min_hours: int, start_cost: float) -> dict:
    """What the forecast says on the day before: the block to offer (or stay off) and the expected profit."""
    block = best_block(day[FORECAST].to_numpy() - cost, min_hours, start_cost)
    hours = block_hours(block)
    expected = float((day.loc[hours, FORECAST] - cost).sum() - start_cost) if hours else 0.0
    return {"block": block, "hours": hours, "expected_profit": expected}


def outcome(day: pd.DataFrame, cost: float, min_hours: int, start_cost: float) -> dict:
    """What actually happened, per decision source: the block chosen and its profit at the actual prices."""
    sources = {"model": day[FORECAST], "same hour yesterday": day[BASELINE], "perfect foresight": day["actual"]}
    result = {}
    for name, prices in sources.items():
        block = best_block(prices.to_numpy() - cost, min_hours, start_cost)
        result[name] = {"block": block, "profit": realised_profit(day, block, cost, start_cost)}
    return result


def period_accuracy(demo: pd.DataFrame) -> dict:
    """MAE of forecast and baseline over the whole demo period, and on how many days the forecast was better."""
    errors = demo[[FORECAST, BASELINE]].sub(demo["actual"], axis=0).abs()
    daily = errors.groupby(demo.index.date).mean()
    return {"mae": errors.mean().to_dict(), "days": len(daily),
            "days_better": int((daily[FORECAST] < daily[BASELINE]).sum()), "daily": daily}


def period_plant(demo: pd.DataFrame, cost: float, min_hours: int, start_cost: float) -> pd.DataFrame:
    """Plant decision over the demo period (src/plant.py): profit per decision source, plus the references."""
    forecasts = {"model": demo[FORECAST], "same hour yesterday": demo[BASELINE]}
    return plant_summary(demo["actual"], forecasts, cost, min_hours, start_cost)


def profit_by_cost(demo: pd.DataFrame, costs, min_hours: int, start_cost: float) -> pd.DataFrame:
    """Total profit (USD per MW) of each decision source for a range of plant costs, one row per cost."""
    table = pd.DataFrame({c: period_plant(demo, c, min_hours, start_cost)["profit"] for c in costs}).T
    table.index.name = "cost"
    return table
