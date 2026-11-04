"""Hyperparameter search with Optuna on the walk-forward folds (Phase 3.3).

Each trial: Optuna proposes settings -> the model is trained on every fold -> score = mean validation MAE.
Optuna's sampler (TPE) learns from earlier trials which regions of the search space look good.
A trial whose running mean is clearly worse than earlier trials at the same fold is stopped early
(pruned), which saves most of the time on bad settings. The test period is never involved.
"""

import numpy as np
import optuna

from src.evaluation import mae
from src.models import fit, predict

SEED = 42
N_TRIALS = {"hist gradient boosting": 50, "xgboost": 50, "random forest": 30}  # forest is the slowest


def space_xgboost(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"]),
    }


def space_hist_gradient_boosting(trial) -> dict:
    return {
        "max_iter": trial.suggest_int("max_iter", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
        "loss": trial.suggest_categorical("loss", ["squared_error", "absolute_error"]),
    }


def space_random_forest(trial) -> dict:
    # criterion stays "squared_error": "absolute_error" is far too slow for random forests on this data size
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


SEARCH_SPACES = {
    "xgboost": space_xgboost,
    "hist gradient boosting": space_hist_gradient_boosting,
    "random forest": space_random_forest,
}


def objective(name: str, folds: list):
    """Optuna objective for model `name`: mean validation MAE over `folds` (from walk_forward_splits)."""
    def run(trial):
        params = SEARCH_SPACES[name](trial)
        fold_maes = []
        for step, (_, X_tr, y_tr, X_val, y_val) in enumerate(folds):
            fold_maes.append(mae(y_val, predict(fit(name, X_tr, y_tr, **params), X_val)))
            trial.report(float(np.mean(fold_maes)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_mae", {f[0]: round(m, 3) for f, m in zip(folds, fold_maes)})
        return float(np.mean(fold_maes))
    return run


def tune(name: str, folds: list, n_trials: int | None = None, callbacks=None) -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        # no pruning during the first 10 trials (nothing to compare with yet) or after only the first fold
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(objective(name, folds), n_trials=n_trials or N_TRIALS[name], callbacks=callbacks)
    return study
