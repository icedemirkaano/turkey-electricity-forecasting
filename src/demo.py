"""Calculations behind the demo app (app/streamlit_app.py), kept here so they can be tested without Streamlit.

User story: an energy analyst reads tomorrow's price (Turkish day-ahead market, PTF) on day D, before the 12:30
offer deadline and before the prices are known. The demo answers three questions about D+1:
  - what will the price look like? (daily and tariff-period means, cheapest / priciest hours, day type, error band)
  - what is the analysis worth? A site with a flexible job of n hours a day (pump, cold store, mill) puts it in
    the n cheapest hours by the forecast and pays the actual prices (src/business.py). Depends only on the price
  - can I trust it? the same forecasts checked against the actual prices
Files (app/data/): demo_forecasts.csv (scripts/make_demo_forecasts.py); error_band.csv, validation_shifting.csv,
demo_inputs.csv (scripts/ptf_analysis.py). Nothing is trained here.
"""

from pathlib import Path

import pandas as pd

from src import analysis as an
from src.business import daily_costs, shifting_summary
from src.config import ROOT, TIMEZONE

DATA = ROOT / "app" / "data"
DEMO_FILE = DATA / "demo_forecasts.csv"
FORECAST, BASELINE = "model (weekly retrain)", "same hour yesterday"
SOURCES = {"model": FORECAST, "same hour yesterday": BASELINE, "perfect foresight": "actual"}
PRICIEST_HOURS = 4


def _read_hourly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame


def load_demo(path: Path = DEMO_FILE) -> pd.DataFrame:
    return _read_hourly(path)


def load_band(path: Path = DATA / "error_band.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col="hour")


def load_inputs(path: Path = DATA / "demo_inputs.csv") -> pd.DataFrame:
    return _read_hourly(path)


def load_validation(path: Path = DATA / "validation_shifting.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def days(demo: pd.DataFrame) -> list:
    return sorted(set(demo.index.date))


def one_day(frame: pd.DataFrame, day) -> pd.DataFrame:
    """The 24 hours of `day`, indexed by hour of day (0-23)."""
    part = frame[frame.index.date == day]
    if len(part) != 24:
        raise ValueError(f"{day} does not have 24 hours")
    return part.set_index(part.index.hour.rename("hour"))


def hour_ranges(hours) -> list[tuple[int, int]]:
    """[10, 11, 12, 13, 18] -> [(10, 14), (18, 19)]: runs of consecutive hours as (first, end) with end exclusive."""
    runs = []
    for h in sorted(hours):
        if runs and runs[-1][1] == h:
            runs[-1] = (runs[-1][0], h + 1)
        else:
            runs.append((h, h + 1))
    return runs


def cheapest(prices: pd.Series, n: int) -> list[int]:
    return an.picked_hours(prices, n, cheapest=True)


def outlook(day: pd.DataFrame, band: pd.DataFrame, n_hours: int) -> dict:
    """What the forecast says on the day before about one delivery day (uses the forecast column only)."""
    forecast = day[FORECAST]
    profile = forecast.to_frame().T
    summary = an.day_summary(profile).iloc[0]
    return {
        "mean": summary["mean"], "day": summary["day"], "peak": summary["peak"], "night": summary["night"],
        "spread": summary["spread"],
        "cheapest": cheapest(forecast, n_hours),
        "priciest": an.picked_hours(forecast, PRICIEST_HOURS, cheapest=False),
        "type": an.day_type(profile).iloc[0],
        "low": (forecast + band["low"]).clip(lower=0), "high": forecast + band["high"],
    }


def outcome(day: pd.DataFrame, n_hours: int) -> dict:
    """What actually happened: per source the hours chosen and what they cost at the actual prices (USD per MW)."""
    result = {}
    for name, column in SOURCES.items():
        hours = cheapest(day[column], n_hours)
        result[name] = {"hours": hours, "paid": float(day.loc[hours, "actual"].sum())}
    result["no shifting"] = {"hours": [], "paid": float(day["actual"].mean() * n_hours)}
    return result


def day_check(day: pd.DataFrame, band: pd.DataFrame) -> dict:
    """How the forecast did on one day: error, hours inside the band, day type forecast vs actual."""
    low, high = (day[FORECAST] + band["low"]).clip(lower=0), day[FORECAST] + band["high"]
    return {"error": float((day[FORECAST] - day["actual"]).abs().mean()),
            "inside": int(((day["actual"] >= low) & (day["actual"] <= high)).sum()),
            "type_forecast": an.day_type(day[[FORECAST]].T).iloc[0],
            "type_actual": an.day_type(day[["actual"]].T).iloc[0]}


def input_change(inputs: pd.DataFrame, day) -> pd.DataFrame:
    """Daily means of the known-ahead inputs for the delivery day and the day before, side by side."""
    tomorrow = one_day(inputs, day).mean()
    today = one_day(inputs, day - pd.Timedelta(days=1)).mean()
    return pd.DataFrame({"today": today, "tomorrow": tomorrow})


def period_accuracy(demo: pd.DataFrame, band: pd.DataFrame, n_hours: int) -> dict:
    """Over the whole demo period: MAE, days better, band coverage, cheapest-hour hit rate, day-type agreement."""
    errors = demo[[FORECAST, BASELINE]].sub(demo["actual"], axis=0).abs()
    daily = errors.groupby(demo.index.date).mean()
    banded = an.apply_band(demo[FORECAST], band)
    actual = an.daily_profile(demo["actual"])
    profiles = {name: an.daily_profile(demo[col]) for name, col in [("model", FORECAST), ("baseline", BASELINE)]}
    true_type = an.day_type(actual)
    return {"mae": errors.mean().to_dict(), "days": len(daily),
            "days_better": int((daily[FORECAST] < daily[BASELINE]).sum()),
            "coverage": an.band_coverage(demo["actual"], banded),
            "hits": {n: an.hour_hit_rate(actual, p, n_hours, cheapest=True).mean() for n, p in profiles.items()},
            "types": {n: (an.day_type(p) == true_type).mean() for n, p in profiles.items()},
            "by_hour": errors.groupby(demo.index.hour).mean()}


def period_shifting(demo: pd.DataFrame, n_hours: int) -> dict:
    """Flexible consumption over the demo period: price paid per source, captured share, day-by-day count."""
    costs = {name: daily_costs(demo["actual"], demo[col], n_hours) for name, col in
             [("model", FORECAST), ("same hour yesterday", BASELINE)]}
    summary = {name: shifting_summary(table, n_hours) for name, table in costs.items()}
    gap = costs["same hour yesterday"]["forecast"] - costs["model"]["forecast"]
    return {"summary": summary, "model_cheaper": int((gap > 1e-9).sum()), "equal": int((gap.abs() <= 1e-9).sum()),
            "baseline_cheaper": int((gap < -1e-9).sum()), "gap_per_day": float(gap.mean())}


def validation_shifting(validation: pd.DataFrame, n_hours: int) -> dict:
    """The same measures on the 638 validation days (saved by scripts/ptf_analysis.py)."""
    rows = validation[validation["n_hours"] == n_hours].set_index("forecast")
    model = rows.loc["model"]
    return {"days": int(model["days"]), "captured": rows["captured_%"].to_dict(),
            "paid": rows["price_forecast"].to_dict(), "no_shifting": float(model["price_no_shifting"]),
            "perfect": float(model["price_perfect"]), "model_cheaper": int(model["model cheaper"]),
            "baseline_cheaper": int(model["baseline cheaper"]), "equal": int(model["equal"]),
            "gap_per_day": float(model["mean gap per day"])}
