"""Round 2, step 1: every model family with the round-1 inputs (v1) and with temperature + hydro added (v2),
on the 7 validation quarters 2025Q1-2026Q3 (src/evaluation.ROUND2_FOLDS). October 2026 on is the new test and
is NOT used here.

For each quarter every model is trained on all data from MODEL_START to the day before the quarter.
Tabular models use their starting settings (src/models.py) plus the round-1 tuned XGBoost settings; SARIMAX and
LEAR as in round 1. LEAR only takes known-ahead series (its inputs include the target day's own values), so its
v2 adds the temperature forecast but not the hydro share. Results are saved after every model; a re-run skips
the models already saved.
Output: reports/results/round2/comparison.csv
Usage (from the repository root):
    python -m scripts.round2_compare
"""

import json
import time

import pandas as pd

from src.config import PROCESSED_DIR, ROOT, TIMEZONE
from src.evaluation import BASELINES, ROUND2_FOLDS, baseline_forecast, scores, walk_forward_splits
from src.features import FEATURE_SETS, MODEL_START, build_features
from src.lear import EXOG as LEAR_EXOG
from src.lear import LEAR, day_inputs
from src.models import MODELS, fit, predict
from src.timeseries import day_ahead_forecasts
from src.timeseries import fit as fit_ts

RESULTS = ROOT / "reports" / "results"
OUT = RESULTS / "round2" / "comparison.csv"
SARIMAX_ORDER = {"kind": "sarimax", "order": (1, 0, 1), "seasonal_order": (1, 0, 1, 24)}
SARIMAX_EXOG = {"v1": ["net_load_forecast"],
                "v2": ["net_load_forecast", "temperature_forecast", "hydro_share_7d_lag48"]}
LEAR_INPUTS = {"v1": LEAR_EXOG, "v2": [*LEAR_EXOG, "temperature_forecast"]}


def train_last(first: str) -> str:
    return (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def run_tabular(name, X, y, flags, **params):
    rows = []
    for fold, X_tr, y_tr, X_val, y_val in walk_forward_splits(X, y, ROUND2_FOLDS):
        start = time.time()
        pred = predict(fit(name, X_tr, y_tr, **params), X_val) if name in MODELS else baseline_forecast(X_val, name)
        rows.append({"fold": fold, "seconds": time.time() - start, **scores(y_val, pred, flags)})
    return rows


def run_sarimax(columns, X, y, flags):
    spec, rows = {**SARIMAX_ORDER, "exog": columns}, []
    for fold, first, last in ROUND2_FOLDS:
        start = time.time()
        state, n_warnings = fit_ts(spec, y, X, train_last(first))
        pred = day_ahead_forecasts(state, spec, y, X, first, last)
        rows.append({"fold": fold, "seconds": time.time() - start, "fit_warnings": n_warnings,
                     **scores(y.loc[pred.index], pred, flags)})
    return rows


def run_lear(exog, table, flags):
    X, Y = day_inputs(table, exog=exog)
    days = lambda frame, a, b: frame.loc[pd.Timestamp(a, tz=TIMEZONE): pd.Timestamp(b, tz=TIMEZONE)]
    rows = []
    for fold, first, last in ROUND2_FOLDS:
        start = time.time()
        model = LEAR().fit(days(X, MODEL_START, train_last(first)), days(Y, MODEL_START, train_last(first)))
        pred = model.predict(days(X, first, last), TIMEZONE)
        rows.append({"fold": fold, "seconds": time.time() - start, **scores(table.loc[pred.index, "ptf_usd"], pred, flags)})
    return rows


def main():
    pd.set_option("display.width", 220)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]
    tuned = json.loads((RESULTS / "tuning" / "best_params.json").read_text())["xgboost"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["model", "inputs"])

    for inputs, spec in FEATURE_SETS.items():
        X, y = build_features(table, spec)
        jobs = {name: (lambda n=name: run_tabular(n, X, y, flags)) for name in [*BASELINES, *MODELS]}
        jobs["xgboost (round-1 tuned settings)"] = lambda: run_tabular("xgboost", X, y, flags, **tuned)
        jobs["sarimax (1,0,1)(1,0,1,24)"] = lambda: run_sarimax(SARIMAX_EXOG[inputs], X, y, flags)
        jobs["lear"] = lambda: run_lear(LEAR_INPUTS[inputs], table, flags)
        for name, job in jobs.items():
            if name in BASELINES and inputs == "v2":
                continue   # baselines use no inputs: one row is enough
            if ((done["model"] == name) & (done["inputs"] == inputs)).any():
                continue
            start = time.time()
            rows = [{"model": name, "inputs": inputs, **row} for row in job()]
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(OUT, index=False)   # after every model
            print(f"{name} [{inputs}]: mean MAE {pd.DataFrame(rows)['mae'].mean():.2f} "
                  f"({time.time() - start:.0f}s)", flush=True)

    per_fold = done.pivot_table(index=["model", "inputs"], columns="fold", values="mae")
    years = {"mean 2025": [f for f in per_fold.columns if f.startswith("2025")],
             "mean 2026": [f for f in per_fold.columns if f.startswith("2026")]}
    per_fold["mean"] = per_fold.mean(axis=1)
    for label, folds in years.items():
        per_fold[label] = per_fold[folds].mean(axis=1)
    print("\n=== ROUND 2: MAE on the 7 validation quarters (v1 = round-1 inputs, v2 = + temperature, hydro)")
    print(per_fold.sort_values("mean").round(2).to_string())
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
