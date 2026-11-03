"""Business layer (Phase 4): what a price forecast is worth for a buying decision.

Load shifting: a consumer has a flexible job that needs `n_hours` hours of 1 MW every day and may run it in any
hours of the day (not necessarily consecutive). On day D it picks the n cheapest hours of D+1 according to a
forecast, and then pays the ACTUAL prices of those hours. Two references:
  - no shifting:       the average actual price of the day (what hours chosen without any forecast cost on average)
  - perfect foresight: the n cheapest hours by actual price (the best possible choice)
Captured share = (no shifting - forecast) / (no shifting - perfect foresight):
  100% = as good as knowing tomorrow's prices, 0% = no better than not shifting, < 0 = worse than not shifting.
Only the ORDER of the forecast prices within a day matters, not their level: a forecast that is too high by the
same amount in every hour picks the same hours as an unbiased one.
"""

import pandas as pd


def _days(series: pd.Series) -> pd.DatetimeIndex:
    return series.index.normalize()


def chosen_hours(forecast: pd.Series, n_hours: int) -> pd.Series:
    """True for the n hours with the lowest forecast price on each day (ties: earlier hour first)."""
    if not 1 <= n_hours <= 24:
        raise ValueError("n_hours must be between 1 and 24")
    rank = forecast.groupby(_days(forecast)).rank(method="first")
    return rank <= n_hours


def daily_costs(actual: pd.Series, forecast: pd.Series, n_hours: int) -> pd.DataFrame:
    """Per day, cost in USD of n_hours MWh: without shifting, with perfect foresight, following the forecast."""
    if not actual.index.equals(forecast.index):
        raise ValueError("actual and forecast must have the same hours")
    hours_per_day = actual.groupby(_days(actual)).size()
    if (hours_per_day != 24).any():
        raise ValueError("every day must have 24 hours")
    pick = chosen_hours(forecast, n_hours)
    days = _days(actual)
    return pd.DataFrame({
        "no_shifting": actual.groupby(days).mean() * n_hours,
        "perfect": actual.groupby(days).apply(lambda day: day.nsmallest(n_hours).sum()),
        "forecast": actual[pick].groupby(days[pick]).sum(),
    })


def shifting_summary(costs: pd.DataFrame, n_hours: int) -> dict:
    """Average price paid (USD/MWh), saving versus no shifting and the captured share of the possible saving."""
    total = costs.sum()
    mwh = n_hours * len(costs)
    saving, possible = total["no_shifting"] - total["forecast"], total["no_shifting"] - total["perfect"]
    return {
        "price_no_shifting": total["no_shifting"] / mwh,
        "price_perfect": total["perfect"] / mwh,
        "price_forecast": total["forecast"] / mwh,
        "saving_per_mwh": saving / mwh,
        "captured_%": 100 * saving / possible,
        "days_worse_than_no_shifting": int((costs["forecast"] > costs["no_shifting"]).sum()),
        "days": len(costs),
    }
