"""Cleaning steps that turn raw EPİAŞ tables into hourly, model-ready columns.

Every decision here is documented in the function docstring, so the reason for each step
can be traced back (e.g. why a column is dropped).
"""

import pandas as pd

# Wind forecast quantile bands (Q5/Q25/Q75/Q95). They stopped being published on 2026-05-13,
# so a model could not use them at forecast time today -> not carried into the hourly table.
WIND_BAND_COLUMNS = ["quarter1", "quarter2", "quarter3", "quarter4"]

# Gap filling (instructor's advice: same-hour average). Only PAST days are used -> no leakage.
# Load plan: consumption depends on the weekday -> same hour, same weekday, previous 4 weeks.
LOAD_PLAN_FILL_LAGS_DAYS = [7, 14, 21, 28]
# Wind forecast: weather does not follow the weekday -> same hour, previous 7 days.
WIND_FILL_LAGS_DAYS = [1, 2, 3, 4, 5, 6, 7]


# rt-gen columns that are NOT generation amounts and must keep their sign:
# importExport is the net cross-border flow: positive = net import, negative = net export
# (checked against published figures in scripts/check_import_export_sign.py).
SIGNED_COLUMNS = ["importExport"]


def clip_negative_generation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Set negative generation values to 0.

    Decision (with the instructor, after plotting): the negative solar values (99 hours, all at night,
    -0.01 to -2.98 MWh) and the negative monitored wind values (5 hours, March 2025) are measurement
    noise around zero, not real negative output. Columns in SIGNED_COLUMNS are left untouched.

    Returns (clipped table, number of values clipped per column).
    """
    columns = [c for c in df.select_dtypes("number").columns if c not in SIGNED_COLUMNS]
    clipped_counts = (df[columns] < 0).sum()
    out = df.copy()
    out[columns] = out[columns].clip(lower=0)
    return out, clipped_counts[clipped_counts > 0]


def fill_from_past_same_hour(series: pd.Series, lags_days: list[int]) -> tuple[pd.Series, pd.Series]:
    """Fill missing hours with the mean of the same clock hour `lags_days` days earlier.

    Returns (filled series, boolean mask of the hours that were filled).
    - Only earlier days are looked at, so a filled value never uses information from its future.
    - Candidates that are themselves missing are skipped; if all are missing the hour stays NaN.
    - The series must have a complete hourly index (missing hours present as NaN).
    """
    past = pd.concat(
        [series.shift(freq=pd.Timedelta(days=d)).reindex(series.index) for d in lags_days], axis=1
    )
    estimate = past.mean(axis=1, skipna=True)
    to_fill = series.isna() & estimate.notna()
    filled = series.where(~to_fill, estimate)
    return filled, to_fill


def to_hourly_wind(wind: pd.DataFrame) -> pd.DataFrame:
    """Convert the sub-hourly wind forecast table to hourly values.

    - Values are power (MW) every 10 min (until May 2026) or 15 min (after); the hourly MEAN
      equals the energy in that hour (MWh), comparable with the hourly rt-gen table.
    - Keeps `forecast` (published the day before -> usable as a model input) and
      `generation` (realized output of the monitored wind farms -> evaluation only, never an input).
    - Drops the quantile bands (see WIND_BAND_COLUMNS).
    - `wind_n_obs` counts the sub-hourly forecast values behind each hour: since June 2026 the
      23:15-23:45 slots are never published, so the 23:00 hour rests on a single value.
    - Hours with no data at all stay in the index as NaN, so gaps remain visible.
    """
    resampled = wind[["forecast", "generation"]].resample("h")
    hourly = resampled.mean()
    hourly["n_obs"] = resampled["forecast"].count()
    return hourly.rename(columns={
        "forecast": "wind_forecast",
        "generation": "wind_generation_monitored",
        "n_obs": "wind_n_obs",
    })
