"""Score the GRU on the same walk-forward folds as the tabular models and compare.

A neural network's result depends on its random starting weights, so every fold is trained with
several seeds; the mean over seeds is the GRU's score and the spread shows how stable it is.
The test period (2026) is NOT touched.
Output: reports/results/gru_walk_forward.csv
Usage (from the repository root):
    python -m scripts.evaluate_gru
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's start-up messages

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import WALK_FORWARD_FOLDS, scores
from src.features import MODEL_START
from src.gru import GRU_RESIDUAL_SETTINGS, GRU_SETTINGS, fit_gru, predict_gru
from src.sequences import day_samples

SEEDS = [1, 2, 3]
VARIANTS = {"gru": GRU_SETTINGS, "gru (residual)": GRU_RESIDUAL_SETTINGS}
RESULTS = ROOT / "reports" / "results"
OUT = RESULTS / "gru_walk_forward.csv"


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]

    rows = []
    for fold, first, last in WALK_FORWARD_FOLDS:
        train_last = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        train, val = day_samples(table, MODEL_START, train_last), day_samples(table, first, last)
        print(f"{fold}: {len(train.days)} training days, {len(val.days)} validation days")
        y_val = val.target_series()
        for model, settings in VARIANTS.items():
            for seed in SEEDS:
                bundle = fit_gru(train, seed, settings)
                rows.append({"model": model, "fold": fold, "seed": seed, "epochs": bundle["epochs"],
                             "seconds": bundle["seconds"], **scores(y_val, predict_gru(bundle, val), flags)})
                print(f"  {model}, seed {seed}: MAE {rows[-1]['mae']:.2f} "
                      f"({bundle['epochs']} epochs, {bundle['seconds']:.0f}s)")
    results = pd.DataFrame(rows)

    print("\n=== per fold (mean over seeds, and spread)")
    print(results.groupby(["model", "fold"])["mae"].agg(["mean", "min", "max"]).round(2).to_string())

    gru = results.groupby(["model", "fold"])[["mae", "rmse", "mae_floor_hours", "mae_cap_hours"]].mean()
    tuned = pd.read_csv(RESULTS / "tuning" / "tuned_comparison.csv")
    others = pd.read_csv(RESULTS / "model_comparison.csv")
    others = pd.concat([tuned[tuned["model"] == "xgboost (tuned)"], others[others["model"] == "same hour yesterday"]])
    both = pd.concat([others, gru.reset_index()])
    print("\n=== mean over the 4 folds")
    print(both.groupby("model")[["mae", "rmse", "mae_floor_hours", "mae_cap_hours"]].mean()
          .sort_values("mae").round(2).to_string())
    print("\n=== MAE per fold")
    print(both.pivot(index="model", columns="fold", values="mae").round(2).to_string())

    results.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
