"""Score the classical time series models (src/timeseries.py) on the same walk-forward folds as all other models.

Added on review request, after the test of the chosen model. Model choice here uses only the
2025 validation quarters; the test period is NOT touched by this script.
Output: reports/results/timeseries_walk_forward.csv
Usage (from the repository root):
    python -m scripts.evaluate_timeseries
    python -m scripts.evaluate_timeseries --models "holt-winters, 24h season"   # one model only
"""

import argparse
import time

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import WALK_FORWARD_FOLDS, scores
from src.timeseries import TS_MODELS, day_ahead_forecasts, fit, hourly_inputs

RESULTS = ROOT / "reports" / "results"
OUT = RESULTS / "timeseries_walk_forward.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(TS_MODELS), choices=list(TS_MODELS))
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]
    y, exog = hourly_inputs(table)

    rows = []
    for name in args.models:
        spec = TS_MODELS[name]
        for fold, first, last in WALK_FORWARD_FOLDS:
            train_last = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            start = time.time()
            results, n_warnings = fit(spec, y, exog, train_last)
            pred = day_ahead_forecasts(results, spec, y, exog, first, last)
            rows.append({"model": name, "fold": fold, "seconds": time.time() - start, "fit_warnings": n_warnings,
                         **scores(y.loc[pred.index], pred, flags)})
            print(f"{name} | {fold}: MAE {rows[-1]['mae']:.2f} ({rows[-1]['seconds']:.0f}s, {n_warnings} warnings)")
        save(pd.DataFrame(rows))   # after every model: a crash later does not lose finished models
    results = pd.DataFrame(rows)

    tuned = pd.read_csv(RESULTS / "tuning" / "tuned_comparison.csv")
    others = pd.read_csv(RESULTS / "model_comparison.csv")
    others = pd.concat([tuned[tuned["model"] == "xgboost (tuned)"], others[others["model"] == "same hour yesterday"]])
    both = pd.concat([others, results])
    print("\n=== mean over the 4 folds")
    print(both.groupby("model")[["mae", "rmse", "mae_floor_hours", "mae_cap_hours"]].mean()
          .sort_values("mae").round(2).to_string())
    print("\n=== MAE per fold")
    print(both.pivot(index="model", columns="fold", values="mae").round(2).to_string())

    print(f"\nsaved {OUT}")


def save(results: pd.DataFrame):
    """Write results, keeping earlier rows of models that are not in `results`."""
    if OUT.exists():
        earlier = pd.read_csv(OUT)
        results = pd.concat([earlier[~earlier["model"].isin(results["model"])], results])
    results.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
