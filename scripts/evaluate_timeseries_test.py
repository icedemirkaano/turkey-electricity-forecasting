"""ADDED AFTER THE TEST (review request): the best classical time series model of each family on the test period.

The models were chosen on the 2025 validation quarters only (scripts/evaluate_timeseries.py) and are scored
here with the same two protocols as XGBoost, so that the rows are comparable:
  - once:   parameters estimated on data up to 2025-12-31, never re-estimated (states still updated daily)
  - weekly: parameters re-estimated every Monday on data up to the day before (as scripts/posthoc_retrain.py)
Prophet has no state to update and is re-fitted every day in both cases, so it has one row only.
Outputs (reports/results/posthoc/): timeseries_test_scores.csv, timeseries_test_by_month.csv,
timeseries_test_predictions.csv
Usage (from the repository root, after evaluate_test and posthoc_retrain --every week):
    python -m scripts.evaluate_timeseries_test
"""

import pandas as pd

from src.config import PROCESSED_DIR, ROOT, TIMEZONE
from src.evaluation import TEST_PERIOD, TRAIN_END, period, retrain_windows, scores, scores_by_month
from src.timeseries import TS_MODELS, day_ahead_forecasts, fit, hourly_inputs

RESULTS = ROOT / "reports" / "results"
OUT_DIR = RESULTS / "posthoc"
CHOSEN = ["sarimax (1,0,1)(1,0,1,24)", "prophet, yearly season", "holt-winters, 24h season"]  # best per family, 2025


def one_day_before(day) -> str:
    return (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def forecast_once(spec, y, exog):
    state, _ = fit(spec, y, exog, TRAIN_END)
    return day_ahead_forecasts(state, spec, y, exog, *TEST_PERIOD)


def forecast_weekly(spec, y, exog):
    parts = []
    for start, end in retrain_windows(*TEST_PERIOD, "week"):
        state, _ = fit(spec, y, exog, one_day_before(start))
        parts.append(day_ahead_forecasts(state, spec, y, exog, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return pd.concat(parts)


def read_hourly(path):
    frame = pd.read_csv(path, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame


def main():
    pd.set_option("display.width", 220)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]
    y, exog = hourly_inputs(table)
    y_test = period(y, *TEST_PERIOD)

    forecasts = {}
    for name in CHOSEN:
        spec = TS_MODELS[name]
        if spec["kind"] == "prophet":
            forecasts[f"{name} (re-fitted daily)"] = forecast_once(spec, y, exog)
        else:
            forecasts[f"{name} (once)"] = forecast_once(spec, y, exog)
            forecasts[f"{name} (weekly)"] = forecast_weekly(spec, y, exog)
        print(f"{name} done", flush=True)

    # Earlier rows for comparison (not recomputed)
    frozen = read_hourly(RESULTS / "test_predictions.csv")
    weekly_xgb = read_hourly(OUT_DIR / "weekly_retrain_predictions.csv").iloc[:, 0]
    reference = {"xgboost (once, reported)": frozen["xgboost (tuned)"], "xgboost (weekly)": weekly_xgb,
                 "same hour yesterday": frozen["same hour yesterday"]}
    every = {**reference, **forecasts}

    overall = pd.DataFrame({n: scores(y_test, f.reindex(y_test.index), flags) for n, f in every.items()}).T
    overall["vs_baseline_%"] = 100 * (1 - overall["mae"] / overall.loc["same hour yesterday", "mae"])
    print("\n=== ADDED AFTER THE TEST: test period, all hours")
    print(overall.sort_values("mae").round(2).to_string())
    by_month = pd.concat({n: scores_by_month(y_test, f.reindex(y_test.index))["mae"] for n, f in every.items()}, axis=1)
    print("\n=== MAE by month")
    print(by_month.round(2).T.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT_DIR / "timeseries_test_scores.csv", index_label="model")
    by_month.to_csv(OUT_DIR / "timeseries_test_by_month.csv")
    pd.DataFrame(forecasts).to_csv(OUT_DIR / "timeseries_test_predictions.csv", index_label="time")
    print(f"\nsaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
