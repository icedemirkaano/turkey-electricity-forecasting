"""Classical time series models for the next-day PTF (added on review request, after the test).

Three families:
  - Holt-Winters exponential smoothing (additive, statsmodels): level + seasonal pattern, updated with every new price
  - SARIMAX (statsmodels): seasonal ARIMA with an external input (the net load forecast, known for D+1 in advance)
  - Prophet: trend + daily/weekly(/yearly) seasonality + official holidays + the net load forecast as a regressor.
    Prophet has no state to update, so it is re-fitted every day on all prices known so far (about 1-2 s per fit).
Day-ahead protocol, the same information as every other model in this project:
  `fit` estimates the parameters on the data up to a given day; then `day_ahead_forecasts` goes day by day:
  the model forecasts the 24 hours of D+1 from the prices up to D 23:00, and only afterwards receives D+1's
  actual prices (state update, no re-estimation). So the forecast for a day never sees that day's prices.
  How often `fit` is called is up to the caller: once per validation quarter (scripts/evaluate_timeseries.py),
  or once / every week on the test period (scripts/evaluate_timeseries_test.py).
State update:
  - SARIMAX: `extend` of the state space results (filters only the new day; equal to `append`, tested).
  - Holt-Winters: the classic statsmodels implementation; the smoothing recursion is re-run with the
    estimated parameters over all prices known so far. (The state space version was tried first: with
    "concentrated" initialisation its `extend` re-computed the initial state from the new day alone and
    broke every second forecast, and `append` was far too slow on two years of hourly data.)
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import TIMEZONE
from src.features import MODEL_START, TARGET
from src.holidays_tr import holiday_calendar

# Prophet's optimiser (cmdstanpy) logs two lines per fit through its own handler; setLevel alone did not stop it.
# Fit failures still raise exceptions.
logging.getLogger("cmdstanpy").disabled = True

HOUR = pd.Timedelta(hours=1)
SARIMAX_WINDOW_DAYS = 56   # SARIMAX is estimated on the last 8 weeks before the forecast period (memory, see fit)
EXOG = ["net_load_forecast"]   # known ahead for every hour of D+1 (lag 0 is allowed, see src/features.py)
# A spec may set its own inputs with "exog": [column names of the exog frame], or [] for none

# Candidates; the one with the lowest mean validation MAE per family is kept
TS_MODELS = {
    "holt-winters, 24h season": {"kind": "ets", "seasonal": 24},
    "holt-winters, 168h season": {"kind": "ets", "seasonal": 168},
    "sarimax (1,0,1)(1,0,1,24)": {"kind": "sarimax", "order": (1, 0, 1), "seasonal_order": (1, 0, 1, 24)},
    "sarimax (1,0,1)(0,1,1,24)": {"kind": "sarimax", "order": (1, 0, 1), "seasonal_order": (0, 1, 1, 24)},
    "prophet": {"kind": "prophet", "yearly": False},
    "prophet, yearly season": {"kind": "prophet", "yearly": True},
}
USES_EXOG = {"sarimax", "prophet"}


def prophet_holidays() -> pd.DataFrame:
    """Official holidays in Prophet's format (holiday name + date); eves are separate 'holidays'."""
    calendar = holiday_calendar()
    return pd.DataFrame({"holiday": calendar["name"], "ds": calendar["date"]})


@dataclass
class ProphetState:
    """All prices known so far; Prophet is fitted on them again for every forecast."""
    yearly: bool
    history: pd.DataFrame   # ds (local time, no time zone: Turkey has no daylight saving), y, net_load_forecast

    def forecast(self, steps: int, exog: np.ndarray) -> np.ndarray:
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=self.yearly,
                        holidays=prophet_holidays())
        model.add_regressor("net_load_forecast")
        model.fit(self.history)
        hours = pd.date_range(self.history["ds"].iloc[-1] + HOUR, periods=steps, freq="h")
        return model.predict(pd.DataFrame({"ds": hours, "net_load_forecast": exog[:, 0]}))["yhat"].to_numpy()

    def extend(self, actual: np.ndarray, exog: np.ndarray) -> "ProphetState":
        hours = pd.date_range(self.history["ds"].iloc[-1] + HOUR, periods=len(actual), freq="h")
        new = pd.DataFrame({"ds": hours, "y": actual, "net_load_forecast": exog[:, 0]})
        return ProphetState(self.yearly, pd.concat([self.history, new], ignore_index=True))


@dataclass
class HoltWintersState:
    """Estimated Holt-Winters parameters plus every price known so far."""
    seasonal: int
    params: dict          # smoothing_level, smoothing_seasonal, initial_level, initial_seasons
    history: np.ndarray   # prices from MODEL_START up to the last known hour

    def forecast(self, steps: int) -> np.ndarray:
        model = ExponentialSmoothing(self.history, seasonal="add", seasonal_periods=self.seasonal,
                                     initialization_method="known", initial_level=self.params["initial_level"],
                                     initial_seasonal=self.params["initial_seasons"])
        fitted = model.fit(smoothing_level=self.params["smoothing_level"],
                           smoothing_seasonal=self.params["smoothing_seasonal"], optimized=False)
        return np.asarray(fitted.forecast(steps))

    def extend(self, actual: np.ndarray) -> "HoltWintersState":
        return HoltWintersState(self.seasonal, self.params, np.concatenate([self.history, actual]))


def hourly_inputs(table: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """The price series and the external input, hourly."""
    exog = pd.DataFrame({"net_load_forecast": table["load_forecast"] - table["wind_forecast"]}, index=table.index)
    return table[TARGET], exog


def _slice(frame, first_day, last_day):
    return frame.loc[pd.Timestamp(first_day, tz=TIMEZONE): pd.Timestamp(last_day, tz=TIMEZONE) + 23 * HOUR]


def exog_columns(spec: dict) -> list[str]:
    """External inputs of a model: spec["exog"] if given (an empty list = none), else EXOG; none for Holt-Winters."""
    return list(spec.get("exog", EXOG)) if spec["kind"] in USES_EXOG else []


def fit(spec: dict, y: pd.Series, exog: pd.DataFrame, train_last: str):
    """Estimate the model on MODEL_START .. train_last. Returns (state, number of convergence warnings)."""
    columns = exog_columns(spec)
    y_tr = _slice(y, MODEL_START, train_last).to_numpy()
    x_tr = _slice(exog[columns], MODEL_START, train_last).to_numpy()
    if np.isnan(y_tr).any() or np.isnan(x_tr).any():
        raise ValueError("training data has missing values")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if spec["kind"] == "ets":
            # Initial level/pattern from the first seasons of the training data; the two smoothing rates are estimated
            fitted = ExponentialSmoothing(y_tr, seasonal="add", seasonal_periods=spec["seasonal"],
                                          initialization_method="heuristic").fit()
            keys = ["smoothing_level", "smoothing_seasonal", "initial_level", "initial_seasons"]
            state = HoltWintersState(spec["seasonal"], {k: fitted.params[k] for k in keys}, y_tr)
        elif spec["kind"] == "prophet":
            ds = _slice(y, MODEL_START, train_last).index.tz_localize(None)
            state = ProphetState(spec["yearly"], pd.DataFrame({"ds": ds, "y": y_tr, "net_load_forecast": x_tr[:, 0]}))
        else:
            # Parameters from the last SARIMAX_WINDOW_DAYS of training only: memory grows linearly with the data
            # (~370 MB for 30 days, ~760 MB for 90 days measured), two years needed more than the 7.7 GB of WSL
            window = SARIMAX_WINDOW_DAYS * 24
            model = SARIMAX(y_tr[-window:], exog=x_tr[-window:] if columns else None, order=spec["order"],
                            seasonal_order=spec["seasonal_order"])
            state = model.fit(disp=False, maxiter=spec.get("maxiter", 50))   # 50 = statsmodels' default
    return state, sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def day_ahead_forecasts(state, spec: dict, y: pd.Series, exog: pd.DataFrame, first_day: str, last_day: str,
                        update: str = "extend") -> pd.Series:
    """Forecast every day first_day..last_day from the prices up to the previous day, one day at a time.
    `state` must be fitted up to the day before first_day. `update="append"` (SARIMAX only) re-filters the
    whole history instead of only the new day; used in the tests to check that both agree."""
    columns = exog_columns(spec)
    uses_exog = bool(columns)
    out = []
    for day in pd.date_range(first_day, last_day, freq="D", tz=TIMEZONE):
        hours = pd.date_range(day, periods=24, freq="h")
        x_day = exog.loc[hours, columns].to_numpy()
        forecast = state.forecast(24, exog=x_day) if uses_exog else state.forecast(24)
        out.append(pd.Series(np.asarray(forecast), index=hours))
        # Only now, after the forecast, the day's actual prices become known
        actual = y.loc[hours].to_numpy()
        step = getattr(state, update)
        state = step(actual, exog=x_day) if uses_exog else step(actual)
    return pd.concat(out).rename("prediction")
