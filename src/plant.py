"""Business layer, round 2: should a power plant run tomorrow, and when?

A 1 MW plant with a variable cost of `cost` USD per MWh (fuel, ...) decides on day D, before the 12:30 offer
deadline and before any price is known, in which hours of D+1 it will generate. Hour by hour the auction would
decide this itself (an hourly offer at the cost is accepted only when the price is higher); what it cannot decide
is what must be fixed BEFORE the prices: which block to offer, the gas to nominate, the crew and the start-up.
That pre-auction decision is simulated here, as simplified unit commitment, as for a gas plant:
  - it runs in at most ONE block of consecutive hours per day, at least `min_hours` long
  - every start costs `start_cost` USD
  - it picks the block where (forecast price - cost) adds up to the most; if even that block does not cover the
    start cost, it stays off for the day
It then earns the ACTUAL price in the hours it runs. Profit of a day = sum over the block of (actual - cost)
minus the start cost (0 when off). A wrong forecast costs money in two ways: running at a loss (the price came out
below the cost) and staying off when running would have paid.
Unlike load shifting (src/business.py), the LEVEL of the forecast matters here, not only the order of the hours:
a forecast that is too high makes the plant run at a loss.
"""

import numpy as np
import pandas as pd


def best_block(margins: np.ndarray, min_hours: int, start_cost: float) -> tuple[int, int] | None:
    """(first hour, last hour) of the block of at least `min_hours` consecutive hours with the largest sum of
    `margins`, or None if no block earns more than `start_cost`. Every block is tried (24 hours: about 250)."""
    if not 1 <= min_hours <= len(margins):
        raise ValueError("min_hours must be between 1 and the number of hours")
    cumulative = np.concatenate([[0.0], np.cumsum(margins)])
    best, best_sum = None, start_cost
    for first in range(len(margins)):
        for last in range(first + min_hours - 1, len(margins)):
            total = cumulative[last + 1] - cumulative[first]
            if total > best_sum:
                best, best_sum = (first, last), total
    return best


def daily_profits(actual: pd.Series, forecast: pd.Series, cost: float, min_hours: int, start_cost: float
                  ) -> pd.DataFrame:
    """Per day: hours run and realised profit (USD) when the run decision follows `forecast`."""
    if not actual.index.equals(forecast.index):
        raise ValueError("actual and forecast must have the same hours")
    rows = {}
    for day, hours in actual.groupby(actual.index.normalize()):
        if len(hours) != 24:
            raise ValueError(f"{day:%Y-%m-%d} does not have 24 hours")
        block = best_block(forecast.loc[hours.index].to_numpy() - cost, min_hours, start_cost)
        if block is None:
            rows[day] = {"hours_run": 0, "profit": 0.0, "loss_hours": 0}
            continue
        first, last = block
        run = hours.to_numpy()[first:last + 1] - cost
        rows[day] = {"hours_run": len(run), "profit": float(run.sum() - start_cost), "loss_hours": int((run < 0).sum())}
    return pd.DataFrame.from_dict(rows, orient="index")


def plant_summary(actual: pd.Series, forecasts: dict[str, pd.Series], cost: float, min_hours: int,
                  start_cost: float) -> pd.DataFrame:
    """One row per decision source: total profit, days run, hours run at a loss and the captured share of
    the profit with perfect foresight. Adds the references 'perfect foresight' and 'always run'."""
    sources = {"perfect foresight": actual, **forecasts}
    days = {name: daily_profits(actual, f, cost, min_hours, start_cost) for name, f in sources.items()}
    always = (actual - cost).groupby(actual.index.normalize()).sum() - start_cost
    best = days["perfect foresight"]["profit"].sum()
    rows = {name: {"profit": d["profit"].sum(), "days_run": int((d["hours_run"] > 0).sum()),
                   "hours_run": int(d["hours_run"].sum()), "loss_hours": int(d["loss_hours"].sum())}
            for name, d in days.items()}
    rows["always run"] = {"profit": always.sum(), "days_run": len(always), "hours_run": len(actual),
                          "loss_hours": int((actual < cost).sum())}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["captured_%"] = 100 * table["profit"] / best if best > 0 else np.nan
    return table
