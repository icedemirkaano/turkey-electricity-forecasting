"""Round 2, step 6: which inputs matter for the chosen model (hist gradient boosting, round-2 tuned settings).

Two complementary checks on the 7 validation quarters 2025Q1-2026Q3. October 2026 on (the test) is NOT used.
  A. Ablation: drop a group of inputs, retrain every Monday as the chosen model is used, compare the MAE.
     Answers "do we need these inputs at all?" (the model may find the same information elsewhere).
  B. Permutation importance: one model per quarter (trained up to the day before it); on that quarter, shuffle
     one input (or a group, with the same shuffle for every column of the group) and measure how much the MAE
     rises. Answers "how much does the trained model rely on this input?". Groups are needed because e.g. the
     three temperature columns carry the same information: shuffled one by one each looks unimportant.
Expectation written before the run: hydro matters more than temperature, mainly in 2026Q1-Q2 (spring with
high hydro); temperature adds little on top of the load forecast, which already contains it.
Caveat: the settings were tuned with all v2 inputs, which slightly favours the full input set in A.
Outputs (reports/results/round2/): ablation.csv, permutation.csv
Usage (from the repository root):
    python -m scripts.round2_importance
"""

import json
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import ROUND2_FOLDS, mae, period, walk_forward_splits
from src.features import FEATURE_SETS, build_features
from src.models import fit, predict
from src.retraining import retrained_forecasts

OUT_DIR = ROOT / "reports" / "results" / "round2"
NAME = "hist gradient boosting"
FIRST, LAST = ROUND2_FOLDS[0][1], ROUND2_FOLDS[-1][2]
GROUPS = {
    "temperature": ["temperature_forecast", "temperature_heating", "temperature_cooling"],
    "hydro": ["hydro_share_lag48", "hydro_share_7d_lag48"],
}
ABLATIONS = {"all v2 inputs": [], "without temperature": GROUPS["temperature"], "without hydro": GROUPS["hydro"],
             "without both (= v1)": GROUPS["temperature"] + GROUPS["hydro"]}
N_REPEATS = 5
SEED = 42


def per_quarter(y: pd.Series, forecast: pd.Series) -> dict:
    return {fold: mae(period(y, a, b), forecast.reindex(period(y, a, b).index)) for fold, a, b in ROUND2_FOLDS}


def ablation(X, y, params) -> pd.DataFrame:
    rows = []
    for label, dropped in ABLATIONS.items():
        start = time.time()
        forecast = retrained_forecasts(X.drop(columns=dropped), y, FIRST, LAST, "week", NAME, **params)
        rows.append({"inputs": label, **per_quarter(y, forecast)})
        print(f"ablation, {label}: {time.time() - start:.0f}s", flush=True)
    return pd.DataFrame(rows).set_index("inputs")


def permutation(X, y, params) -> pd.DataFrame:
    """Rise in MAE when an input (or group) is shuffled, per quarter, mean over N_REPEATS shuffles."""
    targets = {c: [c] for c in X.columns} | {f"[group] {g}": cols for g, cols in GROUPS.items()}
    rng = np.random.default_rng(SEED)
    rows = []
    for fold, X_tr, y_tr, X_val, y_val in walk_forward_splits(X, y, ROUND2_FOLDS):
        model = fit(NAME, X_tr, y_tr, **params)
        base = mae(y_val, predict(model, X_val))
        for label, cols in targets.items():
            rises = []
            for _ in range(N_REPEATS):
                shuffled = X_val.copy()
                order = rng.permutation(len(X_val))
                shuffled[cols] = X_val[cols].to_numpy()[order]   # same shuffle for every column of a group
                rises.append(mae(y_val, predict(model, shuffled)) - base)
            rows.append({"fold": fold, "input": label, "mae_rise": float(np.mean(rises))})
        print(f"permutation, {fold}: base MAE {base:.2f}", flush=True)
    return pd.DataFrame(rows).pivot(index="input", columns="fold", values="mae_rise")


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    folds = [f for f, _, _ in ROUND2_FOLDS]
    table = table[folds].copy()
    table["mean"] = table[folds].mean(axis=1)
    table["mean 2025"] = table[[f for f in folds if f.startswith("2025")]].mean(axis=1)
    table["mean 2026"] = table[[f for f in folds if f.startswith("2026")]].mean(axis=1)
    return table


def main():
    pd.set_option("display.width", 220)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table, FEATURE_SETS["v2"])
    params = json.loads((OUT_DIR / "tuning" / "best_params.json").read_text())[NAME]

    abl = summarise(ablation(X, y, params))
    abl.to_csv(OUT_DIR / "ablation.csv")
    print("\n=== A. ablation: MAE with weekly retraining (lower is better)")
    print(abl.round(2).to_string())

    perm = summarise(permutation(X, y, params)).sort_values("mean", ascending=False)
    perm.to_csv(OUT_DIR / "permutation.csv")
    print("\n=== B. permutation importance: rise in MAE when the input is shuffled (higher = relied on more)")
    print(perm.round(2).to_string())
    print(f"\nsaved {OUT_DIR}")


if __name__ == "__main__":
    main()
