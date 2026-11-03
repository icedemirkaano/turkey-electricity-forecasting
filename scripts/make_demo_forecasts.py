"""Forecasts for the demo period (October 2026 onwards), which no model choice or training has used so far.

The tuned XGBoost is retrained every Monday on all data up to the day before (the lesson of the post-hoc
experiment, scripts/posthoc_retrain.py), then forecasts each day of that week from data known on the day before.
The demo app (app/streamlit_app.py) only reads the saved file, so it starts instantly and needs no EPİAŞ account.
Output: app/data/demo_forecasts.csv (hourly: actual price, forecast, baseline, floor/cap flags)
Usage (from the repository root):
    python -m scripts.make_demo_forecasts
"""

import json

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import DEMO_START, baseline_forecast, period, retrain_windows, scores
from src.features import build_features
from src.retraining import retrained_forecasts

OUT = ROOT / "app" / "data" / "demo_forecasts.csv"
PARAMS = ROOT / "reports" / "results" / "tuning" / "best_params.json"
FORECAST, BASELINE = "xgboost (weekly retrain)", "same hour yesterday"


def last_complete_day(y: pd.Series) -> str:
    """Last day whose 24 prices are all known."""
    hours = y.dropna().groupby(y.dropna().index.normalize()).size()
    return hours[hours == 24].index[-1].strftime("%Y-%m-%d")


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table)
    last = last_complete_day(y)
    params = json.loads(PARAMS.read_text())["xgboost"]

    forecast = retrained_forecasts(X, y, DEMO_START, last, "week", **params)
    X_demo = period(X, DEMO_START, last)
    demo = pd.DataFrame({"actual": period(y, DEMO_START, last), FORECAST: forecast,
                         BASELINE: baseline_forecast(X_demo, BASELINE)})
    demo = demo.join(table[["is_price_floor", "is_price_cap"]])
    if demo.isna().any().any():
        raise ValueError("demo table has missing values")

    flags = demo[["is_price_floor", "is_price_cap"]]
    summary = pd.DataFrame({n: scores(demo["actual"], demo[n], flags) for n in [FORECAST, BASELINE]}).T
    print(f"demo period {DEMO_START} .. {last}: {len(retrain_windows(DEMO_START, last, 'week'))} weekly models, "
          f"{len(demo) // 24} days")
    print(summary.round(2).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(OUT, index_label="time")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
