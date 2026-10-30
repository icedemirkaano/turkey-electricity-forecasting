"""Compare every model in the catalogue (src/models.py) with the baselines on the walk-forward folds.

All models use their starting settings (no tuning yet) and exactly the same folds, features and metrics.
The test period (2026) is NOT touched.
Output: reports/results/model_comparison.csv (one row per model and fold).
Usage (from the repository root):
    python -m scripts.compare_models
"""

import time

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import BASELINES, baseline_forecast, scores, walk_forward_splits
from src.features import build_features
from src.models import MODELS, fit, predict

OUT = ROOT / "reports" / "results" / "model_comparison.csv"
REFERENCE = "same hour yesterday"  # the baseline every model has to beat


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table)
    flags = table[["is_price_floor", "is_price_cap"]]

    rows = []
    for fold, X_tr, y_tr, X_val, y_val in walk_forward_splits(X, y):
        for name in BASELINES:
            rows.append({"fold": fold, "model": name, "seconds": 0.0,
                         **scores(y_val, baseline_forecast(X_val, name), flags)})
        for name in MODELS:
            start = time.time()
            model = fit(name, X_tr, y_tr)
            rows.append({"fold": fold, "model": name, "seconds": time.time() - start,
                         **scores(y_val, predict(model, X_val), flags)})
        print(f"{fold} done")
    results = pd.DataFrame(rows)

    print("\n=== MAE per fold")
    print(results.pivot(index="model", columns="fold", values="mae").round(2).to_string())

    summary = (results.groupby("model")[["mae", "rmse", "mae_floor_hours", "mae_cap_hours", "seconds"]]
               .mean().sort_values("mae"))
    # Positive = better than the reference baseline, in percent of its MAE
    summary["vs_baseline_%"] = 100 * (1 - summary["mae"] / summary.loc[REFERENCE, "mae"])
    print(f"\n=== mean over the 4 folds (vs_baseline_% = MAE improvement over '{REFERENCE}')")
    print(summary.round(2).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
