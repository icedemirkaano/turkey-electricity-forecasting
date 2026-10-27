"""Close look at the drop in monitored wind generation on 25-26 March 2025.

Question: was it a real calm (all wind low) or a data outage in the monitored feed only?
Plots raw 10-min monitored generation + the published forecast + total hourly wind (rt-gen),
and prints the hours where monitored generation is far below its usual share of total wind.
Saves reports/figures/qa/wind_outage_2025_03.png. Read-only otherwise.

Usage (from the repository root):
    python -m scripts.plot_wind_outage
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.clean import to_hourly_wind
from src.config import FIGURES_DIR, TIMEZONE
from src.load import load_raw

START, END = "2025-03-23", "2025-03-28"
BLUE, GRAY, AQUA, INK, SURFACE = "#2a78d6", "#a3a29c", "#1baf7a", "#0b0b0b", "#fcfcfb"
USUAL_SHARE = 0.72  # median monitored / total wind ratio measured earlier


def main():
    raw = load_raw("wind-forecast").loc[START:END]
    hourly = to_hourly_wind(load_raw("wind-forecast")).loc[START:END]
    total = load_raw("rt-gen")["wind"].loc[START:END]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    ax.plot(total, color=GRAY, linewidth=1.6, label="total wind generation, hourly (rt-gen)")
    ax.plot(raw["forecast"], color=AQUA, linewidth=1.2, label="published wind forecast, 10-min")
    ax.plot(raw["generation"], color=BLUE, linewidth=1.2, label="monitored wind generation, 10-min")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_title("Wind, 23-28 March 2025: monitored feed vs total wind", loc="left", fontsize=11, color=INK)
    ax.set_ylabel("MW (10-min) / MWh (hourly)", color=INK)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M", tz=TIMEZONE))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = FIGURES_DIR / "qa" / "wind_outage_2025_03.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)

    # Hours where the monitored feed carries less than a quarter of its usual share of total wind
    share = hourly["wind_generation_monitored"] / total
    suspect = share[share < USUAL_SHARE / 4]
    print(f"\nhours with monitored/total < {USUAL_SHARE / 4:.2f}: {len(suspect)}")
    print(f"from {suspect.index.min():%d.%m.%Y %H:%M} to {suspect.index.max():%d.%m.%Y %H:%M}")
    print("total wind (rt-gen) in those hours: min "
          f"{total[suspect.index].min():.0f}, median {total[suspect.index].median():.0f} MWh")
    print(f"forecast filled in those hours: {hourly.loc[suspect.index, 'wind_forecast'].notna().mean():.0%}")


if __name__ == "__main__":
    main()
