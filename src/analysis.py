"""PTF analysis built on the saved forecasts: what the next day's price looks like, how sure we can be, what kind
of day it is. No new model: everything is computed from the round-2 model's forecasts and past errors.

Pieces (used by scripts/ptf_analysis.py and the demo app):
  - daily profile: the 24 hourly prices of each day as one row
  - day summary: daily mean, mean in the three tariff periods (day 06-17, peak 17-22, night 22-06), cheapest and
    most expensive hours, intraday spread
  - hour hit rate: share of the k hours picked from the forecast that really were among the k cheapest (or most
    expensive) of the day; ties count as hits, so a day with eight zero-price hours is fair
  - error band: per hour of day, the 10th and 90th percentile of past errors (actual - forecast) added to the
    forecast. Calibrated on the validation period only, checked on the test
  - day type: how deep the midday dip is, midday (10-16) mean divided by evening (17-23) mean
"""

import numpy as np
import pandas as pd

TARIFF_PERIODS = {"day": range(6, 17), "peak": range(17, 22), "night": [22, 23, 0, 1, 2, 3, 4, 5]}
MIDDAY, EVENING = range(10, 16), range(17, 23)
DAY_TYPES = [("deep midday dip", 0.3), ("midday dip", 0.7), ("flat day", np.inf)]   # midday/evening ratio upper bounds
BAND_QUANTILES = (0.1, 0.9)


def daily_profile(series: pd.Series) -> pd.DataFrame:
    """Rows = days, columns = hour of day 0-23."""
    frame = series.to_frame("value").assign(day=series.index.date, hour=series.index.hour)
    profile = frame.pivot_table(index="day", columns="hour", values="value")
    if profile.shape[1] != 24 or profile.isna().any().any():
        raise ValueError("every day needs 24 hours")
    return profile


def day_summary(profile: pd.DataFrame) -> pd.DataFrame:
    """One row per day: mean, tariff period means, cheapest/most expensive hour, spread."""
    out = pd.DataFrame({"mean": profile.mean(axis=1)})
    for name, hours in TARIFF_PERIODS.items():
        out[name] = profile[list(hours)].mean(axis=1)
    out["cheapest_hour"] = profile.idxmin(axis=1)
    out["priciest_hour"] = profile.idxmax(axis=1)
    out["spread"] = profile.max(axis=1) - profile.min(axis=1)
    return out


def picked_hours(row: pd.Series, k: int, cheapest: bool) -> list[int]:
    ordered = row.sort_values(kind="stable", ascending=cheapest)
    return sorted(ordered.index[:k])


def hour_hit_rate(actual: pd.DataFrame, forecast: pd.DataFrame, k: int, cheapest: bool) -> pd.Series:
    """Per day, share of the k hours picked from the forecast that are among the actual k cheapest (or priciest)."""
    rates = {}
    for day in actual.index:
        threshold = actual.loc[day].sort_values(ascending=cheapest).iloc[k - 1]
        chosen = actual.loc[day, picked_hours(forecast.loc[day], k, cheapest)]
        hits = (chosen <= threshold) if cheapest else (chosen >= threshold)
        rates[day] = hits.mean()
    return pd.Series(rates)


def error_band(actual: pd.Series, forecast: pd.Series, quantiles=BAND_QUANTILES) -> pd.DataFrame:
    """Per hour of day: the low and high percentile of past errors (actual - forecast)."""
    error = (actual - forecast).groupby(actual.index.hour)
    return pd.DataFrame({"low": error.quantile(quantiles[0]), "high": error.quantile(quantiles[1])})


def apply_band(forecast: pd.Series, band: pd.DataFrame) -> pd.DataFrame:
    """Forecast with its band; the low edge is not allowed below 0 (the price floor)."""
    hours = forecast.index.hour
    return pd.DataFrame({"forecast": forecast,
                         "low": (forecast + band["low"].reindex(hours).to_numpy()).clip(lower=0),
                         "high": forecast + band["high"].reindex(hours).to_numpy()}, index=forecast.index)


def band_coverage(actual: pd.Series, banded: pd.DataFrame) -> float:
    return float(((actual >= banded["low"]) & (actual <= banded["high"])).mean())


def midday_ratio(profile: pd.DataFrame) -> pd.Series:
    evening = profile[list(EVENING)].mean(axis=1)
    return profile[list(MIDDAY)].mean(axis=1) / evening.where(evening > 0)


def day_type(profile: pd.DataFrame) -> pd.Series:
    ratio = midday_ratio(profile).fillna(1.0)
    labels = pd.Series(DAY_TYPES[-1][0], index=profile.index)
    for name, upper in reversed(DAY_TYPES[:-1]):
        labels[ratio <= upper] = name
    return labels
