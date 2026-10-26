"""Plot generation around the periods with negative values, to decide how to treat them.

Saves PNGs to reports/figures/qa/ (nothing else is changed).
Usage (from the repository root):
    python -m scripts.plot_negatives
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.clean import to_hourly_wind
from src.config import FIGURES_DIR, TIMEZONE
from src.load import load_raw

OUT = FIGURES_DIR / "qa"
BLUE, ORANGE, GRAY, INK, SURFACE = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#fcfcfb"


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=TIMEZONE))  # labels in Turkish time, not UTC


def plot_series(series, negatives, start, end, title, unit, filename, compare=None, compare_label=None):
    """Two panels of the SAME series: full scale (top) and zoomed around zero (bottom)."""
    s = series.loc[start:end]
    neg = negatives.loc[start:end]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [2, 1]})
    for ax in (top, bottom):
        if compare is not None:
            ax.plot(compare.loc[start:end], color=GRAY, linewidth=1.2, label=compare_label)
        ax.plot(s, color=BLUE, linewidth=1.5, label=series.name)
        ax.scatter(neg.index, neg.values, color=ORANGE, s=28, zorder=3, label=f"negative ({len(neg)} hours)")
        ax.axhline(0, color=INK, linewidth=0.8)
    style(top, title, unit)
    style(bottom, "Zoom around zero (same series)", unit)
    bottom.set_ylim(min(-5, neg.min() * 1.5 if len(neg) else -5), 50)
    top.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=130)
    plt.close(fig)
    print("saved", OUT / filename)


def plot_negative_values(neg, title, filename):
    """Every negative hour as a dot (date vs value), with the three largest labelled."""
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=SURFACE)
    ax.scatter(neg.index, neg.values, color=ORANGE, s=30, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.8)
    offsets = [(8, -4), (-120, 6), (8, 6)]  # left/right/up so labels of nearby points do not overlap
    for (t, v), offset in zip(neg.nsmallest(3).items(), offsets):
        ax.annotate(f"{v:.2f} MWh\n{t:%d.%m.%Y %H:%M}", (t, v), textcoords="offset points", xytext=offset,
                    fontsize=9, color=INK)
    style(ax, title, "MWh")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10], tz=TIMEZONE))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y", tz=TIMEZONE))
    ax.set_ylim(neg.min() * 1.3, 0.3)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=130)
    plt.close(fig)
    print("saved", OUT / filename)


def main():
    sun = load_raw("rt-gen")["sun"].rename("solar generation (rt-gen)")
    plot_negative_values(sun[sun < 0], f"All {int((sun < 0).sum())} negative solar hours (rt-gen 'sun')",
                         "solar_negative_values.png")
    plot_series(sun, sun[sun < 0], "2025-01-01", "2025-01-31",
                "Solar generation, January 2025 (80 of the 99 negative hours)", "MWh", "solar_negative_2025_01.png")

    sun_other = sun[(sun < 0) & ~((sun.index.year == 2025) & (sun.index.month == 1))]
    print("other negative solar hours:", sun_other.round(2).to_dict())

    wind = to_hourly_wind(load_raw("wind-forecast"))["wind_generation_monitored"].rename("monitored wind generation")
    total_wind = load_raw("rt-gen")["wind"]
    plot_series(wind, wind[wind < 0], "2025-03-01", "2025-03-31",
                "Monitored wind generation, March 2025 (all 5 negative hours)", "MWh", "wind_negative_2025_03.png",
                compare=total_wind, compare_label="total wind generation (rt-gen)")


if __name__ == "__main__":
    main()
