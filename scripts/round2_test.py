"""Round 2, final test: score the chosen model ONCE on October 2026 onwards, next to the baselines.

Chosen before this run (scripts/round2_weekly.py, rule: lowest mean of the 7 validation quarters):
hist gradient boosting, round-2 tuned settings, inputs v2, retrained every Monday on all data up to the day before.
The test runs from ROUND2_TEST_START to the last day whose 24 prices AND all model inputs are known (the newest
days can lack realized generation, which the hydro inputs need). Nothing here changes the model: whatever this
prints is the result that gets reported. The tuned XGBoost (runner-up, 0.11 behind on validation) is printed as
context only; it does not change the choice.
Note: October 2026 was already shown in the round-1 demo (with a different model); no round-2 choice used it.
Outputs (reports/results/round2/): test_scores.csv, test_scores_by_week.csv, test_predictions.csv
Usage (from the repository root):
    python -m scripts.round2_test
"""

import json

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import BASELINES, ROUND2_TEST_START, baseline_forecast, mae, period, retrain_windows, scores
from src.features import FEATURE_SETS, build_features
from src.retraining import retrained_forecasts

OUT_DIR = ROOT / "reports" / "results" / "round2"
CHOSEN, RUNNER_UP = "hist gradient boosting", "xgboost"
REFERENCE = "same hour yesterday"


def last_complete_day(X: pd.DataFrame, y: pd.Series) -> str:
    """Last day whose 24 hours have a price and every input (HGB would silently accept missing inputs)."""
    complete = X.notna().all(axis=1) & y.notna()
    hours = complete[complete].groupby(complete[complete].index.normalize()).size()
    return hours[hours == 24].index[-1].strftime("%Y-%m-%d")


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]
    X, y = build_features(table, FEATURE_SETS["v2"])
    params = json.loads((OUT_DIR / "tuning" / "best_params.json").read_text())
    last = last_complete_day(X, y)
    actual = period(y, ROUND2_TEST_START, last)
    n_weeks = len(retrain_windows(ROUND2_TEST_START, last, "week"))
    print(f"test {ROUND2_TEST_START} .. {last}: {len(actual)} hours, {n_weeks} weekly models")

    forecasts = {f"{CHOSEN} (chosen)": retrained_forecasts(X, y, ROUND2_TEST_START, last, "week", CHOSEN,
                                                           **params[CHOSEN]),
                 f"{RUNNER_UP} (context)": retrained_forecasts(X, y, ROUND2_TEST_START, last, "week", RUNNER_UP,
                                                               **params[RUNNER_UP])}
    for name in BASELINES:
        forecasts[name] = baseline_forecast(period(X, ROUND2_TEST_START, last), name)
    frame = pd.DataFrame({"actual": actual, **{n: f.reindex(actual.index) for n, f in forecasts.items()}})

    summary = pd.DataFrame({n: scores(actual, frame[n], flags) for n in forecasts}).T
    summary.to_csv(OUT_DIR / "test_scores.csv", index_label="model")
    print("\n=== ROUND 2 TEST (run once)")
    print(summary[["hours", "mae", "rmse", "mae_floor_hours", "floor_hours", "mae_cap_hours", "cap_hours"]]
          .round(2).to_string())

    weeks = frame.index.to_period("W-SUN").start_time.strftime("%Y-%m-%d")
    by_week = frame.groupby(weeks).apply(lambda w: pd.Series({n: mae(w["actual"], w[n]) for n in forecasts}))
    by_week.to_csv(OUT_DIR / "test_scores_by_week.csv", index_label="week from")
    print("\n=== MAE per week (weeks start on Monday)")
    print(by_week.round(2).to_string())

    daily = frame.groupby(frame.index.normalize()).apply(
        lambda d: mae(d["actual"], d[f"{CHOSEN} (chosen)"]) < mae(d["actual"], d[REFERENCE]))
    print(f"\ndays where the chosen model beats '{REFERENCE}': {int(daily.sum())} of {len(daily)}")
    frame.to_csv(OUT_DIR / "test_predictions.csv", index_label="time")
    print(f"saved {OUT_DIR}")


if __name__ == "__main__":
    main()
