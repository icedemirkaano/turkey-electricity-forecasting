"""Forecasts from a model that is retrained regularly, as a model in daily use would be.

The period is split into windows (calendar months, or weeks starting on Monday). For each window the model is
trained on everything from MODEL_START to the day before the window, then forecasts the window. No data from
the window or later is used for training.
Used by the post-hoc retraining experiment (scripts/posthoc_retrain.py) and the demo (scripts/make_demo_forecasts.py).
"""

import pandas as pd

from src.evaluation import period, retrain_windows
from src.features import MODEL_START
from src.models import fit, predict


def retrained_forecasts(X: pd.DataFrame, y: pd.Series, first: str, last: str, every: str,
                        name="xgboost", **params) -> pd.Series:
    """Hourly forecasts for first..last (both days included), retraining model `name` at the start of each window."""
    parts = []
    for start, end in retrain_windows(first, last, every):
        train_last = (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        model = fit(name, period(X, MODEL_START, train_last), period(y, MODEL_START, train_last), **params)
        parts.append(predict(model, period(X, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))))
    return pd.concat(parts)
