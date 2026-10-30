"""Tune the chosen models with Optuna on the walk-forward folds and compare them with their untuned versions.

For each model: search (src/tuning.py) -> best settings -> retrain those settings on every fold to get the
full scores (floor/cap hours too). The test period (2026) is NOT touched.
Outputs (reports/results/tuning/): <model>_trials.csv, best_params.json, tuned_comparison.csv
Usage (from the repository root):
    python -m scripts.tune_models                              # all models, default number of trials
    python -m scripts.tune_models --models xgboost --trials 5  # quick check
"""

import argparse
import json
import time

import optuna
import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import scores, walk_forward_splits
from src.features import build_features
from src.models import fit, predict
from src.tuning import SEARCH_SPACES, tune

OUT_DIR = ROOT / "reports" / "results" / "tuning"
UNTUNED = ROOT / "reports" / "results" / "model_comparison.csv"


def progress(start):
    def report(study, trial):
        state = "pruned" if trial.state == optuna.trial.TrialState.PRUNED else f"MAE {trial.value:.3f}"
        print(f"  trial {trial.number + 1}: {state} | best {study.best_value:.3f} | {time.time() - start:.0f}s")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(SEARCH_SPACES), choices=list(SEARCH_SPACES))
    parser.add_argument("--trials", type=int, default=None, help="trials per model (default: N_TRIALS)")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 200)

    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table)
    flags = table[["is_price_floor", "is_price_cap"]]
    folds = list(walk_forward_splits(X, y))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_params, rows = {}, []
    for name in args.models:
        print(f"\n=== {name}")
        study = tune(name, folds, args.trials, callbacks=[progress(time.time())])
        study.trials_dataframe().to_csv(OUT_DIR / f"{name.replace(' ', '_')}_trials.csv", index=False)
        best_params[name] = study.best_params
        print("best:", study.best_params)
        for fold, X_tr, y_tr, X_val, y_val in folds:
            model = fit(name, X_tr, y_tr, **study.best_params)
            rows.append({"fold": fold, "model": f"{name} (tuned)", **scores(y_val, predict(model, X_val), flags)})

    tuned = pd.DataFrame(rows)
    untuned = pd.read_csv(UNTUNED)
    untuned = untuned[untuned["model"].isin(args.models + ["same hour yesterday"])]
    both = pd.concat([untuned, tuned])
    print("\n=== mean over the 4 folds: untuned vs tuned")
    print(both.groupby("model")[["mae", "rmse", "mae_floor_hours", "mae_cap_hours"]].mean()
          .sort_values("mae").round(2).to_string())
    print("\n=== MAE per fold")
    print(both.pivot(index="model", columns="fold", values="mae").round(2).to_string())

    tuned.to_csv(OUT_DIR / "tuned_comparison.csv", index=False)
    (OUT_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    print(f"\nsaved {OUT_DIR}")


if __name__ == "__main__":
    main()
