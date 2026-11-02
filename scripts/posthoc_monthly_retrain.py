"""POST-HOC experiment (decided AFTER seeing the test result): monthly retraining over the test period.

The reported test result is the frozen model (scripts/evaluate_test.py, trained once on 2024-2025).
This script asks a different question: what if the model had been retrained every month, as a model in
daily use would be? Same features, same tuned settings, no new tuning. For each test month M, the model
is trained on everything from MODEL_START to the last day before M, then forecasts M. No data from M or
later is used for training. January's model is identical to the frozen one, so January should match.
Outputs (reports/results/posthoc/): monthly_retrain_scores.csv, monthly_retrain_by_month.csv,
monthly_retrain_predictions.csv
Usage (from the repository root):
    python -m scripts.posthoc_monthly_retrain
"""

import json

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import TEST_PERIOD, period, scores, scores_by_month
from src.features import MODEL_START, build_features
from src.models import fit, predict

RESULTS = ROOT / "reports" / "results"
OUT_DIR = RESULTS / "posthoc"
PARAMS = RESULTS / "tuning" / "best_params.json"
NAME = "xgboost (monthly retrain, post-hoc)"


def main():
    pd.set_option("display.width", 200)
    params = json.loads(PARAMS.read_text())["xgboost"]
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table)
    flags = table[["is_price_floor", "is_price_cap"]]

    parts = []
    for month_start in pd.date_range(TEST_PERIOD[0], TEST_PERIOD[1], freq="MS"):
        first = month_start.strftime("%Y-%m-%d")
        last = (month_start + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        train_last = (month_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        model = fit("xgboost", period(X, MODEL_START, train_last), period(y, MODEL_START, train_last), **params)
        parts.append(predict(model, period(X, first, last)))
        print(f"{first[:7]}: trained on {MODEL_START} .. {train_last}")
    retrained = pd.concat(parts).rename(NAME)

    frozen = pd.read_csv(RESULTS / "test_predictions.csv", index_col="time")
    frozen.index = pd.to_datetime(frozen.index, utc=True).tz_convert(y.index.tz)
    y_test = period(y, *TEST_PERIOD)
    forecasts = {NAME: retrained, "xgboost (frozen, reported)": frozen["xgboost (tuned)"],
                 "same hour yesterday": frozen["same hour yesterday"]}

    overall = pd.DataFrame({name: scores(y_test, f.reindex(y_test.index), flags) for name, f in forecasts.items()}).T
    overall["vs_baseline_%"] = 100 * (1 - overall["mae"] / overall.loc["same hour yesterday", "mae"])
    print("\n=== POST-HOC: test period, all hours")
    print(overall.round(2).to_string())

    by_month = pd.concat({name: scores_by_month(y_test, f.reindex(y_test.index))["mae"]
                          for name, f in forecasts.items()}, axis=1)
    print("\n=== POST-HOC: MAE by month")
    print(by_month.round(2).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT_DIR / "monthly_retrain_scores.csv", index_label="model")
    by_month.to_csv(OUT_DIR / "monthly_retrain_by_month.csv")
    retrained.to_csv(OUT_DIR / "monthly_retrain_predictions.csv", index_label="time")   # hourly, for error analysis
    print(f"\nsaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
