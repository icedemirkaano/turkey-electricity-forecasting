"""Score the two baselines on the walk-forward folds (the line every model has to beat).

Prints per-fold and mean scores and saves them to reports/results/baselines_walk_forward.csv.
The test period is NOT touched here.
Usage (from the repository root):
    python -m scripts.evaluate_baselines
"""

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import BASELINES, baseline_forecast, scores, walk_forward_splits
from src.features import build_features

OUT = ROOT / "reports" / "results" / "baselines_walk_forward.csv"


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table)
    flags = table[["is_price_floor", "is_price_cap"]]

    rows = []
    for fold, _, _, X_val, y_val in walk_forward_splits(X, y):
        for name in BASELINES:
            rows.append({"fold": fold, "model": name, **scores(y_val, baseline_forecast(X_val, name), flags)})
    results = pd.DataFrame(rows)

    print(results.round(2).to_string(index=False))
    print("\n=== mean over the 4 folds")
    print(results.groupby("model")[["mae", "rmse", "mae_floor_hours", "mae_cap_hours"]].mean().round(2).to_string())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
