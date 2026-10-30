"""Model catalogue for the tabular features (Phase 3.3).

Every model follows the scikit-learn interface (fit / predict), so the comparison and tuning scripts treat
them all the same way. One model for all 24 hours: each row is one target hour, and the hour of day is
one of the features. Settings here are reasonable starting values, not tuned ones; tuning comes after the
comparison, only for the best models.
Adding a model = adding one entry to MODELS.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SEED = 42
N_JOBS = 4  # in WSL, n_jobs=-1 (22 threads) was much slower than 4 on small data (see setup guide)

# Calendar columns that a linear model must see as categories: hour 23 is not "23 times" hour 1
CATEGORICAL = ["hour", "day_of_week", "month"]


def _ridge():
    # One-hot for calendar columns, scaling for the rest (a linear model is sensitive to scale; trees are not)
    prepare = ColumnTransformer([("calendar", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                                remainder=StandardScaler())
    return make_pipeline(prepare, Ridge(alpha=1.0))


MODELS = {
    # Sanity check: always predicts the training median (the constant with the lowest MAE)
    "dummy (median)": lambda: DummyRegressor(strategy="median"),
    "ridge": _ridge,
    "random forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                   n_jobs=N_JOBS, random_state=SEED),
    # early_stopping off: otherwise it holds out a RANDOM 10% of rows, which ignores time order
    "hist gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                                                    early_stopping=False, random_state=SEED),
    "xgboost": lambda: XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                    colsample_bytree=0.8, random_state=SEED, n_jobs=N_JOBS),
}


def make_model(name: str, **params):
    """A fresh, untrained model from the catalogue; `params` override its starting settings."""
    model = MODELS[name]()
    return model.set_params(**params) if params else model


def fit(name: str, X: pd.DataFrame, y: pd.Series, **params):
    return make_model(name, **params).fit(X, y)


def predict(model, X: pd.DataFrame) -> pd.Series:
    """Predictions as a Series on the same timestamps as X."""
    return pd.Series(model.predict(X), index=X.index, name="prediction")


def feature_importance(model: XGBRegressor) -> pd.Series:
    """XGBoost only: share of the total gain contributed by each feature (sums to 1), largest first."""
    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    return (gain / gain.sum()).sort_values(ascending=False)
