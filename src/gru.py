"""GRU model for the next-day PTF (Phase 3.4): one sample = one forecast day, 24 outputs at once.

Architecture:
  past sequence (168 h x 3) -> GRU -> summary of the last week
  future block (24 h x features) -> flatten -> dense
  both together -> dense -> 24 prices of D+1
All inputs and the target are standardised with statistics of the TRAINING samples only.
With residual=True the network does not learn the price itself but its deviation from the same hour of
day D (the "same hour yesterday" baseline): it starts from the baseline and only learns corrections.
"""

import time

import keras
import numpy as np
import pandas as pd
from keras import layers

from src.sequences import SEQ_COLUMNS, DaySamples

GRU_SETTINGS = {"units": 64, "dropout": 0.2, "learning_rate": 1e-3, "batch_size": 32,
                "max_epochs": 200, "patience": 15, "early_stop_fraction": 0.15, "residual": False}
# Second attempt: learn the deviation from the baseline, with a smaller network (less to overfit)
GRU_RESIDUAL_SETTINGS = {**GRU_SETTINGS, "units": 32, "residual": True}
PTF_POSITION = list(SEQ_COLUMNS).index("ptf_usd")


def baseline_part(samples: DaySamples) -> np.ndarray:
    """(days, 24): PTF of the 24 hours of day D, i.e. the same hour yesterday for each target hour."""
    return samples.seq[:, -24:, PTF_POSITION]


class Standardiser:
    """Mean/std per feature (last axis), learned on training data only."""

    def fit(self, values: np.ndarray):
        axes = tuple(range(values.ndim - 1))
        self.mean, self.std = values.mean(axis=axes), values.std(axis=axes)
        self.std[self.std == 0] = 1.0   # constant columns (e.g. a flag that never fires in training)
        return self

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse(self, values):
        return values * self.std + self.mean


def build_model(seq_shape, future_shape, units, dropout, learning_rate, n_layers=1) -> keras.Model:
    seq_in = keras.Input(shape=seq_shape, name="past_week")
    future_in = keras.Input(shape=future_shape, name="next_day")
    past = seq_in
    for i in range(n_layers):   # stacked GRUs: all but the last pass the whole sequence on
        past = layers.GRU(units, dropout=dropout, return_sequences=i < n_layers - 1)(past)
    ahead = layers.Dense(32, activation="relu")(layers.Flatten()(future_in))
    x = layers.Concatenate()([past, ahead])
    x = layers.Dropout(dropout)(layers.Dense(64, activation="relu")(x))
    model = keras.Model([seq_in, future_in], layers.Dense(24, name="prices_next_day")(x))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mae")   # same loss as our main metric
    return model


def fit_gru(train: DaySamples, seed: int, settings=GRU_SETTINGS) -> dict:
    """Train on `train`. The last early_stop_fraction of its days (in time order) decide when to stop."""
    keras.backend.clear_session()   # free earlier models (the tuning search trains many in one process)
    keras.utils.set_random_seed(seed)
    goal = train.target - baseline_part(train) if settings["residual"] else train.target
    seq_scale, future_scale = Standardiser().fit(train.seq), Standardiser().fit(train.future)
    y_scale = Standardiser().fit(goal.reshape(-1, 1))
    seq, future = seq_scale.transform(train.seq), future_scale.transform(train.future)
    y = y_scale.transform(goal.reshape(-1, 1)).reshape(goal.shape)

    cut = int(len(y) * (1 - settings["early_stop_fraction"]))   # samples are in day order
    model = build_model(seq.shape[1:], future.shape[1:], settings["units"], settings["dropout"],
                        settings["learning_rate"], settings.get("layers", 1))
    stop = keras.callbacks.EarlyStopping(patience=settings["patience"], restore_best_weights=True)
    start = time.time()
    history = model.fit([seq[:cut], future[:cut]], y[:cut], validation_data=([seq[cut:], future[cut:]], y[cut:]),
                        epochs=settings["max_epochs"], batch_size=settings["batch_size"], shuffle=True,
                        callbacks=[stop], verbose=0)
    return {"model": model, "seq": seq_scale, "future": future_scale, "y": y_scale, "residual": settings["residual"],
            "epochs": len(history.history["loss"]), "seconds": time.time() - start}


def predict_gru(bundle: dict, samples: DaySamples) -> pd.Series:
    """Hourly predictions on the same timestamps as samples.target_series()."""
    scaled = bundle["model"].predict([bundle["seq"].transform(samples.seq),
                                      bundle["future"].transform(samples.future)], verbose=0)
    prices = bundle["y"].inverse(scaled.reshape(-1, 1)).reshape(scaled.shape)
    if bundle["residual"]:
        prices = prices + baseline_part(samples)   # deviation + same hour yesterday = price
    return pd.Series(prices.ravel(), index=samples.hours(), name="prediction")
