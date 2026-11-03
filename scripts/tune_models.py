"""Tune the chosen models with Optuna on the walk-forward folds and compare them with their untuned versions.

For each model: search (src/tuning.py) -> best settings -> retrain those settings on every fold to get the
full scores (floor/cap hours too). The test period is NOT touched.
  --round 1: 4 quarters of 2025, round-1 inputs (v1); test = 2026 Jan-Sep        -> reports/results/tuning/
  --round 2: 7 quarters 2025Q1-2026Q3, inputs v2 (+ temperature, hydro); test = October 2026 on
                                                                                  -> reports/results/round2/tuning/
Outputs: <model>_trials.csv, best_params.json, tuned_comparison.csv
Usage (from the repository root):
    python -m scripts.tune_models                              # round 1, all models, default number of trials
    python -m scripts.tune_models --round 2                    # round 2
    python -m scripts.tune_models --models xgboost --trials 5  # quick check
"""

import argparse
import json
import time

import optuna
import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import ROUND2_FOLDS, WALK_FORWARD_FOLDS, scores, walk_forward_splits
from src.features import FEATURE_SETS, build_features
from src.models import fit, predict
from src.tuning import SEARCH_SPACES, tune

RESULTS = ROOT / "reports" / "results"
ROUNDS = {
    1: {"folds": WALK_FORWARD_FOLDS, "inputs": "v1", "out": RESULTS / "tuning", "untuned": RESULTS / "model_comparison.csv"},
    2: {"folds": ROUND2_FOLDS, "inputs": "v2", "out": RESULTS / "round2" / "tuning",
        "untuned": RESULTS / "round2" / "comparison.csv"},
}


def progress(start):
    def report(study, trial):
        state = "pruned" if trial.state == optuna.trial.TrialState.PRUNED else f"MAE {trial.value:.3f}"
        print(f"  trial {trial.number + 1}: {state} | best {study.best_value:.3f} | {time.time() - start:.0f}s",
              flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(SEARCH_SPACES), choices=list(SEARCH_SPACES))
    parser.add_argument("--trials", type=int, default=None, help="trials per model (default: N_TRIALS)")
    parser.add_argument("--round", type=int, choices=list(ROUNDS), default=1)
    args = parser.parse_args()
    setup = ROUNDS[args.round]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 200)

    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table, FEATURE_SETS[setup["inputs"]])
    flags = table[["is_price_floor", "is_price_cap"]]
    folds = list(walk_forward_splits(X, y, setup["folds"]))
    out_dir = setup["out"]
    out_dir.mkdir(parents=True, exist_ok=True)

    best_params, rows = {}, []
    for name in args.models:
        print(f"\n=== {name}", flush=True)
        study = tune(name, folds, args.trials, callbacks=[progress(time.time())])
        study.trials_dataframe().to_csv(out_dir / f"{name.replace(' ', '_')}_trials.csv", index=False)
        best_params[name] = study.best_params
        print("best:", study.best_params)
        for fold, X_tr, y_tr, X_val, y_val in folds:
            model = fit(name, X_tr, y_tr, **study.best_params)
            rows.append({"fold": fold, "model": f"{name} (tuned)", **scores(y_val, predict(model, X_val), flags)})
        (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))   # after every model

    tuned = pd.DataFrame(rows)
    untuned = pd.read_csv(setup["untuned"])
    if "inputs" in untuned:
        untuned = untuned[untuned["inputs"].eq(setup["inputs"]) | untuned["model"].eq("same hour yesterday")]
    untuned = untuned[untuned["model"].isin(args.models + ["same hour yesterday"])]
    both = pd.concat([untuned, tuned])
    print(f"\n=== mean over the {len(folds)} folds: untuned vs tuned")
    print(both.groupby("model")[["mae", "rmse", "mae_floor_hours", "mae_cap_hours"]].mean()
          .sort_values("mae").round(2).to_string())
    print("\n=== MAE per fold")
    print(both.pivot_table(index="model", columns="fold", values="mae").round(2).to_string())

    tuned.to_csv(out_dir / "tuned_comparison.csv", index=False)
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    main()
