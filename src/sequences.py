"""Day-level samples for the sequence model (GRU, Phase 3.4).

The tabular models see one row per target hour. The GRU sees one sample per forecast day instead:
  - past sequence: the last SEQ_HOURS hours up to the end of day D (forecast day), one row per hour
  - future block: what is known about each of the 24 hours of D+1 (forecasts, price cap, calendar)
  - target: the 24 PTF values of D+1
Availability follows the same rules as src/features.py, applied at day level:
  - PTF of day D was published on D-1 at 14:00 -> the sequence may include PTF up to D 23:00
  - other past_only columns are known up to the end of D-1 -> shifted by SEQ_EXTRA_SHIFT hours
  - known_ahead columns: their value for each hour (also for D+1 in the future block)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import TIMEZONE
from src.features import MODEL_START, TARGET
from src.holidays_tr import holiday_flags

SEQ_HOURS = 168  # one week of history
# column -> extra shift (hours) so that the last step of the sequence (D 23:00) is known at forecast time
SEQ_COLUMNS = {"ptf_usd": 0, "net_load_forecast": 0, "gen_solar": 24}
FUTURE_COLUMNS = ["load_forecast", "wind_forecast", "net_load_forecast", "price_cap_try"]
HOUR = pd.Timedelta(hours=1)


@dataclass
class DaySamples:
    seq: np.ndarray      # (days, SEQ_HOURS, len(SEQ_COLUMNS))
    future: np.ndarray   # (days, 24, number of future features)
    target: np.ndarray   # (days, 24)
    days: pd.DatetimeIndex  # target day (D+1) of each sample, at 00:00

    def hours(self) -> pd.DatetimeIndex:
        """Hourly timestamps of all targets, in sample order."""
        return pd.DatetimeIndex([d + h * HOUR for d in self.days for h in range(24)])

    def target_series(self) -> pd.Series:
        return pd.Series(self.target.ravel(), index=self.hours(), name=TARGET)


def hourly_inputs(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Hourly frames for the past sequence and the future block, plus the target."""
    source = table.copy()
    source["net_load_forecast"] = source["load_forecast"] - source["wind_forecast"]
    seq = pd.DataFrame({c: source[c].shift(s) for c, s in SEQ_COLUMNS.items()}, index=source.index)

    future = source.loc[MODEL_START:, FUTURE_COLUMNS].copy()   # holiday list starts in 2024
    future = future.join(holiday_flags(future.index)).astype(float)
    # Day of week and month as points on a circle: Sunday is next to Monday, December next to January
    for name, value, period in [("dow", future.index.dayofweek, 7), ("month", future.index.month - 1, 12)]:
        future[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        future[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
    return seq, future, source[TARGET]


def day_samples(table: pd.DataFrame, first_day: str, last_day: str) -> DaySamples:
    """Samples for target days first_day..last_day (both included). Days with missing values are skipped."""
    seq, future, target = hourly_inputs(table)
    days = pd.date_range(first_day, last_day, freq="D", tz=TIMEZONE)
    out = {"seq": [], "future": [], "target": [], "days": []}
    for day in days:
        window = seq.loc[day - SEQ_HOURS * HOUR: day - HOUR]          # ends at D 23:00
        ahead = future.loc[day: day + 23 * HOUR]
        values = target.loc[day: day + 23 * HOUR]
        if len(window) != SEQ_HOURS or len(ahead) != 24 or len(values) != 24:
            continue
        if window.isna().any().any() or ahead.isna().any().any() or values.isna().any():
            continue
        out["seq"].append(window.to_numpy())
        out["future"].append(ahead.to_numpy())
        out["target"].append(values.to_numpy())
        out["days"].append(day)
    return DaySamples(np.array(out["seq"]), np.array(out["future"]), np.array(out["target"]),
                      pd.DatetimeIndex(out["days"]))
