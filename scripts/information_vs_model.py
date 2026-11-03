"""POST-HOC experiment (instructor's question, after the test): why did XGBoost beat the time series models?

Two possible reasons, separated with a 2 x 2 comparison on the 2025 validation quarters (the test is NOT used):
                       own past prices only          all inputs
  time series model    SARIMA, no external input     SARIMAX with the inputs XGBoost has
  XGBoost              price lags + hour             the tuned model (6.88, already saved)
- information effect: how much a model gains from the extra inputs (row-wise difference)
- model effect: how much XGBoost gains over the time series model with the same information (column-wise)
Both SARIMA rows use the order of the best classical model, (1,0,1)(1,0,1,24), estimated on the last 8 weeks
before each quarter and updated daily, exactly as in scripts/evaluate_timeseries.py.
XGBoost keeps its tuned settings (not re-tuned for fewer inputs, which slightly favours the full model).
SARIMAX inputs come from build_features (so the same leakage rules apply); megawatt-hours and TRY are divided
by 1000 (fixed constants, no statistics from the data) so the estimation is well scaled. The price lags and the
daily pattern are left to SARIMA's own autoregressive and seasonal terms; day-of-week dummies replace the
weekend flag; month is left out because each SARIMAX window is only 8 weeks long.
Output: reports/results/posthoc/information_vs_model.csv
Usage (from the repository root):
    python -m scripts.information_vs_model
"""

import json
import time

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import WALK_FORWARD_FOLDS, scores, walk_forward_splits
from src.features import build_features
from src.models import fit as fit_tabular
from src.models import predict
from src.timeseries import day_ahead_forecasts, fit as fit_ts

RESULTS = ROOT / "reports" / "results"
OUT = RESULTS / "posthoc" / "information_vs_model.csv"
ORDER = {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 24)}

PRICE_ONLY = ["ptf_usd_lag24", "ptf_usd_lag48", "ptf_usd_lag168", "ptf_usd_prev_day_mean", "hour"]
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat"]   # Sunday is the reference day
SARIMAX_INPUTS = ["load_forecast", "wind_forecast", "price_cap_try", "gen_solar_lag48", "gen_solar_lag168",
                  "is_holiday", "is_holiday_eve_afternoon", *DAYS]
PER_THOUSAND = ["load_forecast", "wind_forecast", "price_cap_try", "gen_solar_lag48", "gen_solar_lag168"]

TS_VARIANTS = {
    "sarima, own past only": {"kind": "sarimax", **ORDER, "exog": []},
    "sarimax, all inputs": {"kind": "sarimax", **ORDER, "exog": SARIMAX_INPUTS},
    # Added after the first run: "all inputs" did not converge within statsmodels' default 50 iterations in any quarter
    "sarimax, all inputs (maxiter 200)": {"kind": "sarimax", **ORDER, "exog": SARIMAX_INPUTS, "maxiter": 200},
}


def sarimax_inputs(X: pd.DataFrame) -> pd.DataFrame:
    exog = X.copy()
    exog[PER_THOUSAND] = exog[PER_THOUSAND] / 1000
    for number, name in enumerate(DAYS):
        exog[name] = (exog["day_of_week"] == number).astype(float)
    return exog[SARIMAX_INPUTS].astype(float)


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    flags = table[["is_price_floor", "is_price_cap"]]
    X, y = build_features(table)
    params = json.loads((RESULTS / "tuning" / "best_params.json").read_text())["xgboost"]

    rows = []
    for fold, X_tr, y_tr, X_val, y_val in walk_forward_splits(X, y):
        start = time.time()
        model = fit_tabular("xgboost", X_tr[PRICE_ONLY], y_tr, **params)
        rows.append({"model": "xgboost, own past only", "fold": fold, "seconds": time.time() - start,
                     **scores(y_val, predict(model, X_val[PRICE_ONLY]), flags)})
        print(f"xgboost, own past only | {fold}: MAE {rows[-1]['mae']:.2f}", flush=True)

    exog = sarimax_inputs(X)
    for name, spec in TS_VARIANTS.items():
        for fold, first, last in WALK_FORWARD_FOLDS:
            train_last = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            start = time.time()
            state, n_warnings = fit_ts(spec, y, exog, train_last)
            pred = day_ahead_forecasts(state, spec, y, exog, first, last)
            rows.append({"model": name, "fold": fold, "seconds": time.time() - start, "fit_warnings": n_warnings,
                         **scores(y.loc[pred.index], pred, flags)})
            print(f"{name} | {fold}: MAE {rows[-1]['mae']:.2f} ({rows[-1]['seconds']:.0f}s, "
                  f"{n_warnings} warnings)", flush=True)
    new = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT, index=False)

    # Rows saved earlier, for the same table
    tuned = pd.read_csv(RESULTS / "tuning" / "tuned_comparison.csv")
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    timeseries = pd.read_csv(RESULTS / "timeseries_walk_forward.csv")
    earlier = pd.concat([tuned[tuned["model"] == "xgboost (tuned)"],
                         comparison[comparison["model"].isin(["ridge", "same hour yesterday"])],
                         timeseries[timeseries["model"] == "sarimax (1,0,1)(1,0,1,24)"]])
    every = pd.concat([earlier, new])
    per_fold = every.pivot_table(index="model", columns="fold", values="mae")
    per_fold["mean"] = per_fold.mean(axis=1)
    print("\n=== POST-HOC: MAE on the 2025 validation quarters")
    print(per_fold.sort_values("mean").round(2).to_string())

    mean = per_fold["mean"]
    grid = pd.DataFrame({"own past only": [mean["sarima, own past only"], mean["xgboost, own past only"]],
                         "all inputs": [mean["sarimax, all inputs (maxiter 200)"], mean["xgboost (tuned)"]]},
                        index=["time series (SARIMA/X)", "XGBoost"])
    print("\n=== 2 x 2 (mean MAE)")
    print(grid.round(2).to_string())
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
