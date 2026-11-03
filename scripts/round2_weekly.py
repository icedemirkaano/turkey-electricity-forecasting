"""Round 2, final model choice: the candidates retrained every Monday, as the model will be used, over the 7
validation quarters 2025Q1-2026Q3. October 2026 on (the new test) is NOT used.

Candidates: the round-2 tuned XGBoost, random forest and HistGradientBoosting (inputs v2, settings from
scripts.tune_models --round 2) and LEAR (round-1 inputs; it did better without the temperature forecast).
Every Monday each model is retrained on all data from MODEL_START to the Sunday before, then forecasts that week.
Rule, fixed before the run: the candidate with the lowest mean of the 7 quarterly MAEs is chosen.
Outputs (reports/results/round2/): weekly_selection.csv (scores per quarter), weekly_predictions.csv (hourly)
Usage (from the repository root):
    python -m scripts.round2_weekly
"""

import json
import time

import pandas as pd

from src.config import PROCESSED_DIR, ROOT, TIMEZONE
from src.evaluation import ROUND2_FOLDS, baseline_forecast, period, retrain_windows, scores
from src.features import FEATURE_SETS, MODEL_START, build_features
from src.lear import LEAR, day_inputs
from src.retraining import retrained_forecasts

OUT_DIR = ROOT / "reports" / "results" / "round2"
FIRST, LAST = ROUND2_FOLDS[0][1], ROUND2_FOLDS[-1][2]
TABULAR = ["xgboost", "random forest", "hist gradient boosting"]


def lear_weekly(table: pd.DataFrame) -> pd.Series:
    X, Y = day_inputs(table)
    days = lambda frame, a, b: frame.loc[pd.Timestamp(a).tz_localize(TIMEZONE): pd.Timestamp(b).tz_localize(TIMEZONE)]
    parts = []
    for start, end in retrain_windows(FIRST, LAST, "week"):
        train_last = start - pd.Timedelta(days=1)
        model = LEAR().fit(days(X, MODEL_START, train_last), days(Y, MODEL_START, train_last))
        parts.append(model.predict(days(X, start, end), TIMEZONE))
    return pd.concat(parts)


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]
    X, y = build_features(table, FEATURE_SETS["v2"])
    params = json.loads((OUT_DIR / "tuning" / "best_params.json").read_text())

    forecasts = {"same hour yesterday": baseline_forecast(period(X, FIRST, LAST), "same hour yesterday")}
    for name in TABULAR:
        start = time.time()
        forecasts[f"{name} (tuned, v2)"] = retrained_forecasts(X, y, FIRST, LAST, "week", name, **params[name])
        print(f"{name}: {time.time() - start:.0f}s", flush=True)
    start = time.time()
    forecasts["lear (v1)"] = lear_weekly(table)
    print(f"lear: {time.time() - start:.0f}s", flush=True)

    rows = []
    for name, forecast in forecasts.items():
        for fold, first, last in ROUND2_FOLDS:
            actual = period(y, first, last)
            rows.append({"model": name, "fold": fold, **scores(actual, forecast.reindex(actual.index), flags)})
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "weekly_selection.csv", index=False)
    pd.DataFrame(forecasts).to_csv(OUT_DIR / "weekly_predictions.csv", index_label="time")

    per_fold = results.pivot_table(index="model", columns="fold", values="mae")
    per_fold["mean"] = per_fold.mean(axis=1)
    per_fold["mean 2025"] = per_fold[[f for f, _, _ in ROUND2_FOLDS if f.startswith("2025")]].mean(axis=1)
    per_fold["mean 2026"] = per_fold[[f for f, _, _ in ROUND2_FOLDS if f.startswith("2026")]].mean(axis=1)
    print("\n=== ROUND 2: weekly retraining, MAE on the 7 validation quarters")
    print(per_fold.sort_values("mean").round(2).to_string())
    chosen = per_fold.drop(index="same hour yesterday")["mean"].idxmin()
    print(f"\nchosen by the pre-set rule (lowest mean of the 7 quarters): {chosen}")
    print(f"saved {OUT_DIR}")


if __name__ == "__main__":
    main()
