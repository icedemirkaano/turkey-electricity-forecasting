"""Train the chosen model (tuned XGBoost) on all training data (2024-2025) and look at its feature importance.

Settings come from the Optuna search (reports/results/tuning/best_params.json). The test period (2026) is
NOT used here: it is scored once, in the next step.
Outputs: models/xgboost_final.json (the trained model, not in git), reports/results/xgb_feature_importance.csv,
reports/figures/models/01_xgb_feature_importance.png
Usage (from the repository root):
    python -m scripts.train_final_xgboost
"""

import json

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from src import plots
from src.config import PROCESSED_DIR, ROOT
from src.evaluation import TRAIN_END, period
from src.features import MODEL_START, build_features
from src.models import feature_importance, fit

PARAMS = ROOT / "reports" / "results" / "tuning" / "best_params.json"
MODEL_OUT = ROOT / "models" / "xgboost_final.json"
IMPORTANCE_OUT = ROOT / "reports" / "results" / "xgb_feature_importance.csv"


def main():
    params = json.loads(PARAMS.read_text())["xgboost"]
    table = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")
    X, y = build_features(table)
    X_train, y_train = period(X, MODEL_START, TRAIN_END), period(y, MODEL_START, TRAIN_END)
    print(f"training rows: {len(X_train)} ({X_train.index.min()} -> {X_train.index.max()})")

    model = fit("xgboost", X_train, y_train, **params)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_OUT)

    importance = feature_importance(model)
    importance.rename("share").to_csv(IMPORTANCE_OUT, index_label="feature")
    print("\n=== feature importance (share of total gain)")
    print(importance.round(3).to_string())

    fig = plots.feature_importance_bars(importance, "Tuned XGBoost: feature importance (trained on 2024-2025)")
    print("\nsaved", MODEL_OUT, IMPORTANCE_OUT, plots.save(fig, "01_xgb_feature_importance", subfolder="models"))


if __name__ == "__main__":
    main()
