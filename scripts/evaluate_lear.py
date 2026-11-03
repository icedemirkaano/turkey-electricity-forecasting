"""POST-HOC (instructor's question, after the test): LEAR, the strong linear time series benchmark of the price
forecasting literature (src/lear.py), on the same 2025 walk-forward quarters as every other model.

Same protocol as XGBoost: for each quarter, trained once on all days from MODEL_START to the day before the
quarter, then each day of the quarter is forecast from data known on the day before. The test is NOT used.
Output: reports/results/posthoc/lear_walk_forward.csv
Usage (from the repository root):
    python -m scripts.evaluate_lear
"""

import time

import pandas as pd

from src.config import PROCESSED_DIR, ROOT, TIMEZONE
from src.evaluation import WALK_FORWARD_FOLDS, scores
from src.features import MODEL_START
from src.lear import LEAR, day_inputs

RESULTS = ROOT / "reports" / "results"
OUT = RESULTS / "posthoc" / "lear_walk_forward.csv"


def days_between(X, first, last):
    return X.loc[pd.Timestamp(first, tz=TIMEZONE): pd.Timestamp(last, tz=TIMEZONE)]


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]
    X, Y = day_inputs(table)

    rows = []
    for fold, first, last in WALK_FORWARD_FOLDS:
        train_last = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        X_tr, Y_tr = days_between(X, MODEL_START, train_last), days_between(Y, MODEL_START, train_last)
        X_val = days_between(X, first, last)
        start = time.time()
        pred = LEAR().fit(X_tr, Y_tr).predict(X_val, TIMEZONE)
        rows.append({"model": "lear", "fold": fold, "seconds": time.time() - start, "train_days": len(X_tr),
                     **scores(table.loc[pred.index, "ptf_usd"], pred, flags)})
        print(f"lear | {fold}: MAE {rows[-1]['mae']:.2f} ({len(X_tr)} training days, "
              f"{rows[-1]['seconds']:.0f}s)", flush=True)
    results = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)

    tuned = pd.read_csv(RESULTS / "tuning" / "tuned_comparison.csv")
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    timeseries = pd.read_csv(RESULTS / "timeseries_walk_forward.csv")
    every = pd.concat([tuned[tuned["model"] == "xgboost (tuned)"],
                       comparison[comparison["model"].isin(["ridge", "same hour yesterday"])],
                       timeseries[timeseries["model"] == "sarimax (1,0,1)(1,0,1,24)"], results])
    per_fold = every.pivot_table(index="model", columns="fold", values="mae")
    per_fold["mean"] = per_fold.mean(axis=1)
    print("\n=== POST-HOC: MAE on the 2025 validation quarters")
    print(per_fold.sort_values("mean").round(2).to_string())
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
