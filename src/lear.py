"""LEAR: the lasso-estimated autoregressive model of the price forecasting literature (post-hoc, review question).

Follows the open-source implementation of Lago, Marcjasz, De Schutter & Weron (2021, Applied Energy 293, 116983),
epftoolbox/models/_lear.py:
  - one linear model per delivery hour (24 models), each forecasting its hour of the target day T
  - 247 inputs, the same for every hour model:
      prices of all 24 hours of T-1, T-2, T-3 and T-7                       4 x 24 = 96
      for each exogenous series: its 24 values on T, T-1 and T-7           2 x 72 = 144
      day of week of T (one-hot)                                                    7
  - inputs (except the day dummies) and target scaled per column with asinh((x - median) / MAD), fitted on training
  - lasso; the penalty is chosen with an information criterion (LassoLarsIC, AIC), then a Lasso is refitted with it
Adapted to this project: the exogenous series are EPİAŞ's next-day load and wind forecasts (both known ahead, so
their value on T is allowed); the price of T-1 is day D, published on D-1 at 14:00. Training uses the same
expanding window as every other model here (from MODEL_START), not epftoolbox's 3-year window.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoLarsIC
from statsmodels.robust import mad

PRICE_DAYS = [1, 2, 3, 7]
EXOG = ["load_forecast", "wind_forecast"]
EXOG_DAYS = [0, 1, 7]
N_FEATURES = 24 * len(PRICE_DAYS) + 24 * len(EXOG_DAYS) * len(EXOG) + 7


def day_matrix(series: pd.Series) -> pd.DataFrame:
    """Hourly series -> one row per day, one column per hour (0-23)."""
    frame = pd.DataFrame({"day": series.index.normalize(), "hour": series.index.hour, "value": series.to_numpy()})
    return frame.pivot(index="day", columns="hour", values="value")


def day_inputs(table: pd.DataFrame, target="ptf_usd", exog=EXOG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(inputs, targets), one row per target day T. Days with any missing value are dropped.
    `exog`: known-ahead columns, each used on T, T-1 and T-7 (round 2 adds the temperature forecast)."""
    prices = day_matrix(table[target])
    parts = [prices.shift(d).add_prefix(f"price_T-{d}_h") for d in PRICE_DAYS]
    for column in exog:
        values = day_matrix(table[column])
        parts += [values.shift(d).add_prefix(f"{column}_T-{d}_h") for d in EXOG_DAYS]
    dummies = pd.get_dummies(prices.index.dayofweek, prefix="dow").astype(float)
    dummies.index = prices.index
    X = pd.concat(parts + [dummies], axis=1)
    X.columns = X.columns.astype(str)
    keep = X.notna().all(axis=1) & prices.notna().all(axis=1)
    return X[keep], prices[keep]


class InvariantScaler:
    """asinh((x - median) / MAD) per column, as epftoolbox's 'Invariant' scaling."""

    def fit(self, values: np.ndarray):
        self.median = np.median(values, axis=0)
        self.mad = mad(values, axis=0)
        self.mad[self.mad == 0] = 1.0
        return self

    def transform(self, values):
        return np.arcsinh((values - self.median) / self.mad)

    def inverse(self, values):
        return np.sinh(values) * self.mad + self.median


class LEAR:
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.n_dummies = sum(c.startswith("dow_") for c in X.columns)
        self.x_scale = InvariantScaler().fit(X.to_numpy()[:, :-self.n_dummies])
        self.y_scale = InvariantScaler().fit(Y.to_numpy())
        x, y = self._x(X), self.y_scale.transform(Y.to_numpy())
        self.models = []
        for h in range(24):
            alpha = LassoLarsIC(criterion="aic", max_iter=2500).fit(x, y[:, h]).alpha_
            self.models.append(Lasso(alpha=alpha, max_iter=2500).fit(x, y[:, h]))
        return self

    def _x(self, X):
        values = X.to_numpy()
        return np.hstack([self.x_scale.transform(values[:, :-self.n_dummies]), values[:, -self.n_dummies:]])

    def predict(self, X: pd.DataFrame, tz) -> pd.Series:
        """Hourly forecasts for the target days in X."""
        scaled = np.column_stack([m.predict(self._x(X)) for m in self.models])
        prices = self.y_scale.inverse(scaled)
        hours = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in X.index for h in range(24)]).tz_convert(tz)
        return pd.Series(prices.ravel(), index=hours, name="prediction")
