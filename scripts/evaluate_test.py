"""Score the final model ONCE on the test period (January-September 2026), next to the baselines.

The model is the tuned XGBoost saved by scripts/train_final_xgboost.py (trained on 2024-2025 only).
Nothing here changes the model: whatever this prints is the result that gets reported.
Outputs (reports/results/): test_scores.csv, test_scores_by_month.csv, test_predictions.csv
Usage (from the repository root):
    python -m scripts.evaluate_test
"""

import pandas as pd
from xgboost import XGBRegressor

from src.config import PROCESSED_DIR, ROOT
from src.evaluation import BASELINES, TEST_PERIOD, baseline_forecast, period, scores, scores_by_month
from src.features import build_features
from src.models import predict

MODEL = ROOT / "models" / "xgboost_final.json"
RESULTS = ROOT / "reports" / "results"
REFERENCE = "same hour yesterday"


def main():
    pd.set_option("display.width", 200)
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table)
    flags = table[["is_price_floor", "is_price_cap"]]
    X_test, y_test = period(X, *TEST_PERIOD), period(y, *TEST_PERIOD)
    print(f"test hours: {len(X_test)} ({X_test.index.min()} -> {X_test.index.max()})")

    model = XGBRegressor()
    model.load_model(MODEL)
    # The saved model must expect exactly today's feature columns, in the same order
    assert model.get_booster().feature_names == list(X_test.columns), "feature columns changed since training"

    forecasts = {"xgboost (tuned)": predict(model, X_test)}
    forecasts.update({name: baseline_forecast(X_test, name) for name in BASELINES})

    overall = pd.DataFrame({name: scores(y_test, f, flags) for name, f in forecasts.items()}).T
    overall["vs_baseline_%"] = 100 * (1 - overall["mae"] / overall.loc[REFERENCE, "mae"])
    print("\n=== test period, all hours")
    print(overall.round(2).to_string())

    by_month = pd.concat({name: scores_by_month(y_test, f)["mae"] for name, f in forecasts.items()}, axis=1)
    print("\n=== MAE by month")
    print(by_month.round(2).to_string())

    predictions = pd.DataFrame({"actual": y_test, **forecasts}).join(flags)
    overall.to_csv(RESULTS / "test_scores.csv", index_label="model")
    by_month.to_csv(RESULTS / "test_scores_by_month.csv")
    predictions.to_csv(RESULTS / "test_predictions.csv", index_label="time")
    print(f"\nsaved test_scores.csv, test_scores_by_month.csv, test_predictions.csv in {RESULTS}")


if __name__ == "__main__":
    main()
