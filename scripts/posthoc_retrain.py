"""POST-HOC experiment (decided AFTER seeing the test result): regular retraining over the test period.

The reported test result is the frozen model (scripts/evaluate_test.py, trained once on 2024-2025).
This script asks a different question: what if the model had been retrained regularly, as a model in
daily use would be? Same features, same tuned settings, no new tuning. The test period is split into
windows (calendar months, or weeks starting on Monday); for each window the model is trained on everything
from MODEL_START to the day before the window, then forecasts the window. No data from the window or later
is used for training. The first window's model is identical to the frozen one, so its days should match.
Every retraining frequency that is tried is reported, not only the best one.
Outputs (reports/results/posthoc/): <every>ly_retrain_scores.csv, _by_month.csv, _predictions.csv
Usage (from the repository root):
    python -m scripts.posthoc_retrain                 # monthly (default)
    python -m scripts.posthoc_retrain --every week    # weekly
"""

import argparse
import json

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import TEST_PERIOD, period, scores, scores_by_month
from src.features import MODEL_START, build_features
from src.models import fit, predict

RESULTS = ROOT / "reports" / "results"
OUT_DIR = RESULTS / "posthoc"
PARAMS = RESULTS / "tuning" / "best_params.json"
WINDOW_STARTS = {"month": "MS", "week": "W-MON"}   # pandas frequency of the first day of each window


def windows(every: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(first day, last day) of each retraining window inside the test period."""
    first, last = pd.Timestamp(TEST_PERIOD[0]), pd.Timestamp(TEST_PERIOD[1])
    starts = sorted({first, *pd.date_range(first, last, freq=WINDOW_STARTS[every])})
    ends = [s - pd.Timedelta(days=1) for s in starts[1:]] + [last]
    return list(zip(starts, ends))


def read_hourly(path, tz):
    frame = pd.read_csv(path, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(tz)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", choices=list(WINDOW_STARTS), default="month")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    name = f"xgboost ({args.every}ly retrain, post-hoc)"
    params = json.loads(PARAMS.read_text())["xgboost"]
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table)
    flags = table[["is_price_floor", "is_price_cap"]]

    parts = []
    for start, end in windows(args.every):
        first, last = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        train_last = (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        model = fit("xgboost", period(X, MODEL_START, train_last), period(y, MODEL_START, train_last), **params)
        parts.append(predict(model, period(X, first, last)))
    retrained = pd.concat(parts).rename(name)
    print(f"{len(parts)} windows, each trained on {MODEL_START} .. the day before the window")

    y_test = period(y, *TEST_PERIOD)
    frozen = read_hourly(RESULTS / "test_predictions.csv", y.index.tz)
    forecasts = {name: retrained, "xgboost (frozen, reported)": frozen["xgboost (tuned)"],
                 "same hour yesterday": frozen["same hour yesterday"]}
    for other in sorted(OUT_DIR.glob("*ly_retrain_predictions.csv")):   # other frequencies tried earlier
        column = read_hourly(other, y.index.tz).iloc[:, 0]
        forecasts.setdefault(column.name, column)

    overall = pd.DataFrame({n: scores(y_test, f.reindex(y_test.index), flags) for n, f in forecasts.items()}).T
    overall["vs_baseline_%"] = 100 * (1 - overall["mae"] / overall.loc["same hour yesterday", "mae"])
    print("\n=== POST-HOC: test period, all hours")
    print(overall.round(2).to_string())

    by_month = pd.concat({n: scores_by_month(y_test, f.reindex(y_test.index))["mae"]
                          for n, f in forecasts.items()}, axis=1)
    print("\n=== POST-HOC: MAE by month")
    print(by_month.round(2).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.every}ly_retrain"
    overall.to_csv(OUT_DIR / f"{prefix}_scores.csv", index_label="model")
    by_month.to_csv(OUT_DIR / f"{prefix}_by_month.csv")
    retrained.to_csv(OUT_DIR / f"{prefix}_predictions.csv", index_label="time")   # hourly, for error analysis
    print(f"\nsaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
