"""Figures of round 2 for the README, the notebook and the presentation. Reads saved results only; trains nothing.

  01  one test week (5-11 October 2026): hourly actual price vs the chosen model and the baseline
  02  why retraining matters: MAE per validation quarter, trained once per quarter vs every Monday
  03  which inputs the chosen model relies on (permutation importance, validation)
  04  the plant decision: extra profit from the model over the baseline, by plant cost
  05  example day (Sunday 11 October 2026): forecasts, actual price and the block each forecast chose
  06  error by day of week: the model vs yesterday, last week and the same weekday last year (validation)
Output: reports/figures/round2/
Usage (from the repository root, after the round-2 scripts, evaluate_plant and round2_baselines):
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
from src.plant import best_block

ROUND2 = ROOT / "reports" / "results" / "round2"
MODEL, BASELINE = "hist gradient boosting (chosen)", "same hour yesterday"
EXAMPLE_DAY, EXAMPLE_COST, MIN_HOURS, START_COST = "2026-10-11", 50, 4, 50
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


def plant_by_cost():
    plant = pd.read_csv(ROUND2 / "business" / "plant_by_cost.csv")
    profit = plant.pivot(index="cost", columns="source", values="profit")
    gain = (profit["model"] - profit["same hour yesterday"]).loc[20:80]
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=plots.SURFACE)
    ax.bar(gain.index, gain, width=3.6, color=[plots.ORANGE if c == EXAMPLE_COST else plots.BLUE for c in gain.index])
    for cost, value in gain.items():
        ax.text(cost, value + 15, f"{value:,.0f}", ha="center", fontsize=9, color=plots.MUTED)
    ax.set_xticks(gain.index)
    plots.style(ax, "Plant decision: extra profit from the model over the baseline, 32 days, 1 MW",
                "USD", "plant's fuel cost, USD/MWh")
    fig.tight_layout()
    return fig


def example_day(frame):
    day = frame.loc[EXAMPLE_DAY]
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=plots.SURFACE)
    for prices, name, color, row in [(day[MODEL], "model", plots.ORANGE, -6),
                                     (day[BASELINE], "same hour yesterday", plots.GRAY, -12)]:
        first, last = best_block(prices.to_numpy() - EXAMPLE_COST, MIN_HOURS, START_COST)
        ax.plot([first - 0.4, last + 0.4], [row, row], color=color, linewidth=6, solid_capstyle="butt")
        ax.text(last + 0.6, row, f"runs: {name}", va="center", fontsize=9, color=plots.MUTED)
    ax.plot(hours, day["actual"], color=plots.INK, linewidth=2.2, marker="o", markersize=4, label="actual price")
    ax.plot(hours, day[MODEL], color=plots.ORANGE, linewidth=1.6, label="model forecast")
    ax.plot(hours, day[BASELINE], color=plots.GRAY, linewidth=1.6, linestyle="--", label="same hour yesterday")
    ax.axhline(EXAMPLE_COST, color=plots.BLUE, linewidth=1, linestyle=":", label=f"plant cost {EXAMPLE_COST}")
    ax.set_xticks(hours)
    ax.set_ylim(-16, 100)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    plots.style(ax, "Sunday 11 October 2026: midday prices fell to 0; the baseline ran all day, the model "
                "only the evening", "USD/MWh", "hour")
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
               "03_permutation_importance": importance(), "04_plant_extra_profit_by_cost": plant_by_cost(),
               "05_example_day_2026_10_11": example_day(frame), "06_error_by_weekday": by_weekday()}
    for name, fig in figures.items():
        print("saved", plots.save(fig, name, subfolder="round2"))
        plt.close(fig)


if __name__ == "__main__":
    main()
