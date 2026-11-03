"""Business layer (Phase 4): load shifting with the test-period forecasts (src/business.py).

No model is trained here; the saved test predictions are reused:
  - reported:  frozen XGBoost (the one-time test result), same hour yesterday, same hour last week
  - post-hoc:  weekly retrained XGBoost and SARIMAX (decided after the test, labelled as such)
Outputs (reports/results/business/): load_shifting.csv, load_shifting_by_month.csv
Usage (from the repository root, after evaluate_test, posthoc_retrain --every week and evaluate_timeseries_test):
    python -m scripts.evaluate_load_shifting
"""

import pandas as pd

from src.business import daily_costs, shifting_summary
from src.config import ROOT, TIMEZONE

RESULTS = ROOT / "reports" / "results"
OUT_DIR = RESULTS / "business"
N_HOURS = [4, 8]   # flexible job: 4 or 8 hours of 1 MW per day
MAIN_N = 4         # used for the monthly table


def read_hourly(path):
    frame = pd.read_csv(path, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame


def main():
    pd.set_option("display.width", 200)
    test = read_hourly(RESULTS / "test_predictions.csv")
    weekly_xgb = read_hourly(RESULTS / "posthoc" / "weekly_retrain_predictions.csv").iloc[:, 0]
    timeseries = read_hourly(RESULTS / "posthoc" / "timeseries_test_predictions.csv")
    actual = test["actual"]
    forecasts = {
        "xgboost (once, reported)": test["xgboost (tuned)"],
        "same hour yesterday": test["same hour yesterday"],
        "same hour last week": test["same hour last week"],
        "xgboost (weekly, post-hoc)": weekly_xgb,
        "sarimax (weekly, post-hoc)": timeseries["sarimax (1,0,1)(1,0,1,24) (weekly)"],
    }

    rows, monthly = [], {}
    for n in N_HOURS:
        for name, forecast in forecasts.items():
            costs = daily_costs(actual, forecast.reindex(actual.index), n)
            rows.append({"n_hours": n, "forecast": name, **shifting_summary(costs, n)})
            if n == MAIN_N:
                monthly[name] = {month.strftime("%Y-%m"): shifting_summary(part, n)["captured_%"]
                                 for month, part in costs.groupby(costs.index.tz_localize(None).to_period("M"))}
    table = pd.DataFrame(rows)
    by_month = pd.DataFrame(monthly).T

    print("=== Load shifting on the test period (2026-01..09), USD/MWh")
    for n, part in table.groupby("n_hours"):
        print(f"\n--- {n} flexible hours per day")
        print(part.drop(columns="n_hours").set_index("forecast").round(2).to_string())
    print(f"\n=== Captured share of the possible saving by month (%), {MAIN_N} hours")
    print(by_month.round(1).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "load_shifting.csv", index=False)
    by_month.to_csv(OUT_DIR / "load_shifting_by_month.csv", index_label="forecast")
    print(f"\nsaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
