"""Error analysis on the test period (Phase 3.6): where and when do the forecasts go wrong?

Reads the saved hourly test predictions; nothing is retrained and nothing is changed.
Compares the reported (frozen) model, the post-hoc monthly retrained model and the baseline:
  1. daily mean actual vs forecasts over time
  2. bias (forecast - actual) by month: is the error systematic?
  3. MAE by hour of day
  4. MAE by actual price level (is the model weak at low prices?)
  5. the 10 worst days of the reported model
Figures: reports/figures/models/02-05. Tables: reports/results/error_analysis/.
Usage (from the repository root, after evaluate_test and posthoc_retrain):
    python -m scripts.analyse_errors
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src import plots
from src.config import ROOT, TIMEZONE

RESULTS = ROOT / "reports" / "results"
OUT_DIR = RESULTS / "error_analysis"
FROZEN, RETRAINED, BASELINE = "xgboost (tuned)", "xgboost (monthly retrain, post-hoc)", "same hour yesterday"
FORECASTS = {  # display name -> (column, color)
    "XGBoost, frozen (reported)": (FROZEN, plots.ORANGE),
    "XGBoost, monthly retrain (post-hoc)": (RETRAINED, plots.BLUE),
    "same hour yesterday": (BASELINE, plots.GRAY),
}
PRICE_BINS = [-0.01, 0.01, 20, 40, 60, 80, np.inf]
PRICE_LABELS = ["0 (floor)", "0-20", "20-40", "40-60", "60-80", "80+"]


def read_hourly(path):
    frame = pd.read_csv(path, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame


def main():
    pd.set_option("display.width", 200)
    p = read_hourly(RESULTS / "test_predictions.csv")
    p[RETRAINED] = read_hourly(RESULTS / "posthoc" / "monthly_retrain_predictions.csv")[RETRAINED]
    columns = [FROZEN, RETRAINED, BASELINE]
    errors = p[columns].sub(p["actual"], axis=0)        # forecast - actual: positive = too high
    month = p.index.strftime("%Y-%m")

    bias = errors.groupby(month).mean()
    by_hour = errors.abs().groupby(p.index.hour).mean()
    level = pd.cut(p["actual"], PRICE_BINS, labels=PRICE_LABELS)
    by_level = errors.abs().groupby(level, observed=False).mean().assign(hours=level.value_counts())
    daily = errors.abs().groupby(p.index.normalize()).mean()
    worst = (daily.nlargest(10, FROZEN)
             .assign(actual_mean=p["actual"].resample("D").mean(), floor_hours=p["is_price_floor"].resample("D").sum()))

    for title, table in [("bias by month (forecast - actual)", bias), ("MAE by hour of day", by_hour),
                         ("MAE by actual price level", by_level), ("10 worst days of the reported model", worst)]:
        print(f"\n=== {title}")
        print(table.round(2).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in [("bias_by_month", bias), ("mae_by_hour", by_hour), ("mae_by_price_level", by_level),
                        ("worst_days", worst)]:
        table.to_csv(OUT_DIR / f"{name}.csv")

    figures = {
        "02_test_daily_actual_vs_forecasts": plots.daily_forecast_comparison(
            p, FORECASTS, "Test period 2026: daily mean price, actual vs forecasts"),
        "03_test_bias_by_month": plots.grouped_bars(
            bias, FORECASTS, "Bias by month (forecast - actual; above 0 = too high)", "month",
            ylabel="mean error (USD/MWh)"),
        "04_test_mae_by_hour": plots.grouped_bars(by_hour, FORECASTS, "Test MAE by hour of day", "hour"),
        "05_test_mae_by_price_level": plots.grouped_bars(
            by_level, FORECASTS, "Test MAE by actual price level", "actual price (USD/MWh)"),
    }
    for name, fig in figures.items():
        print("saved", plots.save(fig, name, subfolder="models"))


if __name__ == "__main__":
    main()
