"""Figures of round 2 for the README, the notebook and the presentation. Reads saved results only; trains nothing.

  01  one test week (5-11 October 2026): hourly actual price vs the chosen model and the baseline
  02  why retraining matters: MAE per validation quarter, trained once per quarter vs every Monday
  03  which inputs the chosen model relies on (permutation importance, validation)
  04  what the cheap hours are worth: share of the possible saving captured, by hours per day (validation)
  05  example day (Sunday 18 October 2026): forecasts, band, actual price and the 4 hours each forecast chose
  06  error by day of week: the model vs yesterday, last week and the same weekday last year (validation)
  07  the three day types: mean actual price profile of each (validation)
Output: reports/figures/round2/
Usage (from the repository root, after the round-2 scripts, round2_baselines and ptf_analysis):
    python -m scripts.round2_figures
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import plots
from src.config import ROOT, TIMEZONE
from src import analysis as an
from src.demo import cheapest

ROUND2 = ROOT / "reports" / "results" / "round2"
MODEL, BASELINE = "hist gradient boosting (chosen)", "same hour yesterday"
EXAMPLE_DAY, N_HOURS = "2026-10-18", 4
ANALYSIS = ROUND2 / "analysis"
LABELS = {"ptf_usd_lag24": "price, same hour yesterday", "net_load_forecast": "net load forecast (load - wind)",
          "ptf_usd_lag168": "price, same hour last week", "gen_solar_lag48": "solar generation, 2 days ago",
          "gen_solar_lag168": "solar generation, last week", "hour": "hour of day", "day_of_week": "day of week",
          "wind_forecast": "wind forecast", "ptf_usd_lag48": "price, 2 days ago",
          "ptf_usd_prev_day_mean": "yesterday's mean price", "[group] temperature": "temperature (3 inputs together)",
          "[group] hydro": "hydro share (2 inputs together)"}


def test_predictions() -> pd.DataFrame:
    frame = pd.read_csv(ROUND2 / "test_predictions.csv", index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame


def test_week(frame, first="2026-10-05", last="2026-10-11"):
    frame = frame.loc[first:last]
    fig, ax = plt.subplots(figsize=(13, 4.2), facecolor=plots.SURFACE)
    ax.plot(frame["actual"], color=plots.INK, linewidth=1.3, label="actual price")
    ax.plot(frame[MODEL], color=plots.ORANGE, linewidth=1.1, label="model (test month MAE 7.74)")
    ax.plot(frame[BASELINE], color=plots.GRAY, linewidth=1.0, linestyle="--", label="same hour yesterday (11.37)")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    plots.style(ax, "One test week, hour by hour: 5-11 October 2026 (the model never saw these days)", "USD/MWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=TIMEZONE))
    fig.tight_layout()
    return fig


def retraining_by_quarter():
    tuned = pd.read_csv(ROUND2 / "tuning" / "tuned_comparison.csv")
    weekly = pd.read_csv(ROUND2 / "weekly_selection.csv")
    table = pd.DataFrame({
        "same hour yesterday": weekly.query("model == 'same hour yesterday'").set_index("fold")["mae"],
        "model, trained once per quarter": tuned.query("model == 'hist gradient boosting (tuned)'")
        .set_index("fold")["mae"],
        "model, retrained every Monday": weekly.query("model == 'hist gradient boosting (tuned, v2)'")
        .set_index("fold")["mae"]})
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=plots.SURFACE)
    x, width = np.arange(len(table)), 0.27
    for i, (name, color) in enumerate(zip(table.columns, [plots.GRAY, plots.BLUE, plots.ORANGE])):
        ax.bar(x + (i - 1) * width, table[name], width, color=color, label=name)
    ax.set_xticks(x, table.index)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, table.max().max() * 1.2)
    plots.style(ax, "Validation MAE per quarter: retraining every week fixed 2026", "MAE (USD/MWh)")
    fig.tight_layout()
    return fig


def importance():
    perm = pd.read_csv(ROUND2 / "permutation.csv", index_col="input")["mean"]
    shown = perm.reindex(list(LABELS)).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=plots.SURFACE)
    colors = [plots.BLUE if name.startswith("[group]") else plots.ORANGE for name in shown.index]
    ax.barh([LABELS[n] for n in shown.index], shown, color=colors)
    plots.style(ax, "What the model relies on: rise in MAE when an input is shuffled (validation)",
                xlabel="rise in MAE, USD/MWh")
    ax.grid(axis="x", color=plots.GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def captured_by_hours():
    table = pd.read_csv(ANALYSIS / "shifting.csv").query("period == 'validation'")
    captured = table.pivot(index="n_hours", columns="forecast", values="captured_%")
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=plots.SURFACE)
    ax.plot(captured.index, captured["model"], color=plots.ORANGE, linewidth=2.5, marker="o", label="model")
    ax.plot(captured.index, captured["same hour yesterday"], color=plots.GRAY, linewidth=2.5, marker="o",
            label="same hour yesterday")
    for n in [4, 8]:
        for name in ["model", "same hour yesterday"]:
            ax.text(n, captured.loc[n, name] + 1.5, f"{captured.loc[n, name]:.0f}%", ha="center", fontsize=9,
                    color=plots.MUTED)
    ax.set_xticks(captured.index)
    ax.set_ylim(50, 100)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    plots.style(ax, "Flexible consumption in the cheapest hours: share of the possible saving captured "
                "(100% = knowing the prices; validation, 638 days)", "%", "hours needed per day")
    fig.tight_layout()
    return fig


def example_day(frame):
    day = frame.loc[EXAMPLE_DAY]
    hours = np.arange(24)
    band = pd.read_csv(ANALYSIS / "band.csv", index_col="time")
    forecast = day[MODEL].to_numpy()
    low, high = np.clip(forecast + band["low"].to_numpy(), 0, None), forecast + band["high"].to_numpy()
    actual = day["actual"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=plots.SURFACE)
    ax.fill_between(hours, low, high, color=plots.ORANGE, alpha=0.12, linewidth=0, label="band (80% of past hours)")
    ax.plot(hours, actual, color=plots.INK, linewidth=2.2, marker="o", markersize=4, label="actual price")
    ax.plot(hours, forecast, color=plots.ORANGE, linewidth=1.6, label="model forecast")
    ax.plot(hours, day[BASELINE].to_numpy(), color=plots.GRAY, linewidth=1.6, linestyle="--",
            label="same hour yesterday")
    for prices, name, color, row in [(day[MODEL], "model", plots.ORANGE, -6),
                                     (day[BASELINE], "same hour yesterday", plots.GRAY, -12)]:
        chosen = cheapest(pd.Series(prices.to_numpy()), N_HOURS)
        ax.scatter(chosen, [row] * len(chosen), color=color, marker="s", s=60)
        ax.text(23.6, row, f"{name}: pays {actual[chosen].sum():.1f} USD", va="center", ha="right", fontsize=9,
                color=plots.MUTED)
    ax.set_xticks(hours)
    ax.set_ylim(-16, 100)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    plots.style(ax, "Sunday 18 October 2026: the 4 cheapest hours by each forecast (squares) and what they cost "
                "at the actual prices, 1 MW", "USD/MWh", "hour")
    fig.tight_layout()
    return fig


def day_types():
    from scripts.ptf_analysis import load_periods
    actual = an.daily_profile(load_periods()["validation"]["actual"])
    types = an.day_type(actual)
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=plots.SURFACE)
    for (name, _), color in zip(an.DAY_TYPES, [plots.ORANGE, plots.BLUE, plots.GRAY]):
        ax.plot(range(24), actual[types == name].mean(), color=color, linewidth=2.5,
                label=f"{name} ({(types == name).mean():.0%} of days)")
    ax.axvspan(9.5, 15.5, color=plots.GRID, alpha=0.5, linewidth=0)
    ax.axvspan(16.5, 22.5, color=plots.GRID, alpha=0.25, linewidth=0)
    ax.set_xticks(range(24))
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    plots.style(ax, "Three day types by the midday (10-16) / evening (17-23) price ratio: mean profile, "
                "validation 638 days", "USD/MWh", "hour")
    fig.tight_layout()
    return fig


def by_weekday():
    table = pd.read_csv(ROUND2 / "baselines_by_weekday.csv", index_col=["period", "weekday"]).loc["validation"]
    series = [("model (weekly retrained)", "model", plots.ORANGE),
              ("same hour yesterday", "same hour yesterday", plots.GRAY),
              ("same hour last week", "same hour last week", plots.BLUE),
              ("same weekday last year", "same weekday last year", plots.INK)]
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=plots.SURFACE)
    x, width = np.arange(len(table)), 0.2
    for i, (column, label, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * width, table[column], width, color=color, label=label)
    ax.set_xticks(x, table.index)
    ax.set_ylim(0, table.max().max() * 1.22)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    plots.style(ax, "Error by day of the forecast day: yesterday fails when the day type changes, the model "
                "does not (validation, 21 months)", "MAE (USD/MWh)")
    fig.tight_layout()
    return fig


def main():
    frame = test_predictions()
    figures = {"01_test_week_hourly": test_week(frame), "02_retraining_by_quarter": retraining_by_quarter(),
               "03_permutation_importance": importance(), "04_cheap_hours_captured": captured_by_hours(),
               "05_example_day_2026_10_18": example_day(frame), "06_error_by_weekday": by_weekday(),
               "07_day_types": day_types()}
    for name, fig in figures.items():
        print("saved", plots.save(fig, name, subfolder="round2"))
        plt.close(fig)


if __name__ == "__main__":
    main()
