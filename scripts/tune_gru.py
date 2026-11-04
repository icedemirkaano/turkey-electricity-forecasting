"""Optuna search for the residual GRU on the walk-forward folds (same folds as every other model).

Each trial: settings -> one GRU per fold (seed 1) -> mean validation MAE. Trials whose running mean is clearly
worse than earlier trials are stopped early (pruned). The best settings are then trained with the 3 seeds of
scripts/evaluate_gru.py, so the result is comparable with the untuned GRU rows. The test period is NOT used.
Outputs (reports/results/tuning/): gru_trials.csv, gru_best_params.json, gru_tuned_walk_forward.csv
Usage (from the repository root):
    python -m scripts.tune_gru               # 25 trials
    python -m scripts.tune_gru --trials 2    # quick check
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import WALK_FORWARD_FOLDS, mae, scores
from src.features import MODEL_START
from src.gru import GRU_RESIDUAL_SETTINGS, fit_gru, predict_gru
from src.sequences import day_samples

OUT_DIR = ROOT / "reports" / "results" / "tuning"
SEEDS = [1, 2, 3]
SEARCH_SEED = 1


def space(trial) -> dict:
    return {
        "seq_hours": trial.suggest_categorical("seq_hours", [72, 168, 336]),   # 3, 7 or 14 days of history
        "units": trial.suggest_categorical("units", [16, 32, 64]),
        "layers": trial.suggest_int("layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


class FoldData:
    """Day samples per fold and history length, built once and reused by every trial."""

    def __init__(self, table):
        self.table, self.cache = table, {}

    def get(self, seq_hours):
        if seq_hours not in self.cache:
            folds = []
            for fold, first, last in WALK_FORWARD_FOLDS:
                train_last = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                folds.append((fold, day_samples(self.table, MODEL_START, train_last, seq_hours),
                              day_samples(self.table, first, last, seq_hours)))
            self.cache[seq_hours] = folds
        return self.cache[seq_hours]


def settings_from(params: dict) -> dict:
    return {**GRU_RESIDUAL_SETTINGS, **{k: v for k, v in params.items() if k != "seq_hours"}}


def objective(data: FoldData):
    def run(trial):
        params = space(trial)
        fold_maes = []
        for step, (_, train, val) in enumerate(data.get(params["seq_hours"])):
            pred = predict_gru(fit_gru(train, SEARCH_SEED, settings_from(params)), val)
            fold_maes.append(mae(val.target_series(), pred))
            trial.report(float(np.mean(fold_maes)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_mae", dict(zip([f for f, _, _ in WALK_FORWARD_FOLDS], np.round(fold_maes, 3))))
        return float(np.mean(fold_maes))
    return run


def best_from_saved_trials() -> dict:
    """Best completed trial in gru_trials.csv (used when the search was stopped before n_trials)."""
    trials = pd.read_csv(OUT_DIR / "gru_trials.csv")
    row = trials[trials["state"] == "COMPLETE"].sort_values("value").iloc[0]
    print(f"best of {len(trials)} saved trials: trial {int(row['number']) + 1}, MAE {row['value']:.3f}")
    best = {k.removeprefix("params_"): row[k] for k in trials.columns if k.startswith("params_")}
    return {k: int(v) if k in ("seq_hours", "units", "layers", "batch_size") else float(v) for k, v in best.items()}


def search(data: FoldData, n_trials: int) -> dict:
    start = time.time()

    def progress(study, trial):
        state = "pruned" if trial.state == optuna.trial.TrialState.PRUNED else f"MAE {trial.value:.3f}"
        print(f"  trial {trial.number + 1}: {state} | best {study.best_value:.3f} | {time.time() - start:.0f}s",
              flush=True)
        study.trials_dataframe().to_csv(OUT_DIR / "gru_trials.csv", index=False)   # saved after every trial

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.optimize(objective(data), n_trials=n_trials, callbacks=[progress])
    return study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--from-trials", action="store_true",
                        help="skip the search, take the best trial from gru_trials.csv")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]
    data = FoldData(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best = best_from_saved_trials() if args.from_trials else search(data, args.trials)
    print("best:", best)
    (OUT_DIR / "gru_best_params.json").write_text(json.dumps(best, indent=2))

    # Best settings with the same 3 seeds as the untuned GRU rows
    rows = []
    for fold, train, val in data.get(best["seq_hours"]):
        for seed in SEEDS:
            bundle = fit_gru(train, seed, settings_from(best))
            rows.append({"model": "gru (residual, tuned)", "fold": fold, "seed": seed, "epochs": bundle["epochs"],
                         **scores(val.target_series(), predict_gru(bundle, val), flags)})
    tuned = pd.DataFrame(rows)
    tuned.to_csv(OUT_DIR / "gru_tuned_walk_forward.csv", index=False)
    untuned = pd.read_csv(ROOT / "reports" / "results" / "gru_walk_forward.csv")
    both = pd.concat([untuned, tuned])
    print("\n=== mean over folds and seeds")
    print(both.groupby("model")[["mae", "rmse", "mae_floor_hours", "mae_cap_hours"]].mean().round(2).to_string())
    print("\n=== MAE per fold (mean over seeds)")
    print(both.groupby(["model", "fold"])["mae"].mean().unstack().round(2).to_string())
    print(f"\nsaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
