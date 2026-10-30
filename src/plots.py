"""Shared chart style and the charts used in the analysis notebooks.

One look for every figure (README, notebooks, LinkedIn): light surface, recessive grid, thin lines,
one accent color per series, axis labels in Turkish time. Each chart function returns the figure;
`save` writes it to reports/figures/<subfolder>/.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, TIMEZONE

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, title=None, ylabel=None, xlabel=None):
    """Apply the shared look to one axis."""
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def save(fig, name, subfolder="eda"):
    out = FIGURES_DIR / subfolder / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    return out


def daily_price_try_usd(table: pd.DataFrame):
    """Daily mean PTF in TRY (with the price cap) and in USD, as two stacked panels on one time axis."""
    daily = table[["ptf_try", "ptf_usd", "price_cap_try"]].resample("D").agg(
        {"ptf_try": "mean", "ptf_usd": "mean", "price_cap_try": "max"})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, facecolor=SURFACE)
    top.plot(daily["price_cap_try"], color=GRAY, linewidth=1.2, linestyle="--", drawstyle="steps-post",
             label="price cap (max limit)")
    top.plot(daily["ptf_try"], color=BLUE, linewidth=1.0, label="daily mean PTF")
    top.legend(frameon=False, loc="upper left")
    style(top, "Daily mean PTF, TRY/MWh", "TRY/MWh")
    bottom.plot(daily["ptf_usd"], color=BLUE, linewidth=1.0)
    style(bottom, "Daily mean PTF, USD/MWh (forecast target)", "USD/MWh")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y", tz=TIMEZONE))
    fig.tight_layout()
    return fig


def price_histogram(table: pd.DataFrame, column="ptf_usd", bins=80):
    """Distribution of hourly prices; the share of floor (0) hours is written on the chart."""
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=SURFACE)
    ax.hist(table[column], bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    floor_share = table["is_price_floor"].mean()
    ax.annotate(f"price = 0: {table['is_price_floor'].sum()} hours ({floor_share:.1%})",
                xy=(0, ax.get_ylim()[1] * 0.9), xytext=(10, 0), textcoords="offset points", color=INK)
    style(ax, f"Hourly PTF distribution ({column})", "hours", "USD/MWh" if column == "ptf_usd" else "TRY/MWh")
    fig.tight_layout()
    return fig


def month_hour_heatmap(values: pd.Series, title: str, cbar_label: str, cmap="Blues", agg="sum"):
    """Month (rows) x hour of day (columns) matrix as a heatmap. `values` is indexed by timestamp.
    agg="sum" counts flags (e.g. floor hours); agg="mean" averages a value (e.g. price)."""
    matrix = values.groupby([values.index.strftime("%Y-%m"), values.index.hour]).agg(agg).unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.22 * len(matrix))), facecolor=SURFACE)
    image = ax.imshow(matrix.values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    step = max(1, len(matrix) // 12)
    ax.set_yticks(range(0, len(matrix), step), matrix.index[::step])
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label(cbar_label, color=MUTED)
    style(ax, title, xlabel="hour of day")
    ax.grid(False)
    fig.tight_layout()
    return fig, matrix


def tariff_period(hours: pd.Index) -> pd.Series:
    """Map hour of day to the three-time tariff periods (T1 06-17 day, T2 17-22 peak, T3 22-06 night).
    Source: EPDK Board decision no. 6561 (27.10.2016), bands for meters on permanent summer time (Turkey since 2016)."""
    labels = np.select([(hours >= 6) & (hours < 17), (hours >= 17) & (hours < 22)], ["T1 day", "T2 peak"], "T3 night")
    return pd.Series(labels, index=hours)


def tariff_period_summary(table: pd.DataFrame, column="ptf_usd") -> pd.DataFrame:
    """Mean price per tariff period and year."""
    period = tariff_period(table.index.hour).values
    return table[column].groupby([table.index.year, period]).mean().unstack().round(1)


def price_by_bins(table: pd.DataFrame, x: str, y="ptf_usd", bins=10, title=None, xlabel=None):
    """Split `x` into equal-count bins (deciles by default) and show the median of `y` per bin with its
    25-75% range. Returns (figure, per-bin table). Shows an association, not a causal effect."""
    data = table[[x, y]].dropna()
    data["bin"] = pd.qcut(data[x], bins, duplicates="drop")
    summary = data.groupby("bin", observed=True).agg(
        x_mid=(x, "median"), y_median=(y, "median"), y_q25=(y, lambda s: s.quantile(0.25)),
        y_q75=(y, lambda s: s.quantile(0.75)), hours=(y, "size"))
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=SURFACE)
    ax.fill_between(summary["x_mid"], summary["y_q25"], summary["y_q75"], color=BLUE, alpha=0.15,
                    label="25-75% of hours")
    ax.plot(summary["x_mid"], summary["y_median"], color=BLUE, linewidth=2, marker="o", markersize=5,
            label="median")
    ax.legend(frameon=False, loc="best")
    style(ax, title or f"{y} by {x} (deciles)", "USD/MWh" if y == "ptf_usd" else y, xlabel or x)
    fig.tight_layout()
    return fig, summary.round({"x_mid": 3, "y_median": 1, "y_q25": 1, "y_q75": 1})


# Generation sources grouped for charts (at most 6 groups -> fixed categorical colors)
SOURCE_GROUPS = {
    "natural gas": ["gen_natural_gas"],
    "coal": ["gen_lignite", "gen_import_coal", "gen_black_coal", "gen_asphaltite_coal"],
    "hydro": ["gen_dammed_hydro", "gen_river"],
    "wind": ["gen_wind"],
    "solar": ["gen_solar"],
    "other": ["gen_geothermal", "gen_biomass", "gen_waste_heat", "gen_fuel_oil"],
}
GROUP_COLORS = {"natural gas": ORANGE, "coal": "#52514e", "hydro": BLUE, "wind": AQUA, "solar": "#eda100",
                "other": GRAY}


def grouped_generation(table: pd.DataFrame) -> pd.DataFrame:
    """Hourly generation per source group (MWh); net import is left out (it can be negative)."""
    return pd.DataFrame({group: table[cols].sum(axis=1) for group, cols in SOURCE_GROUPS.items()})


def generation_mix_by_hour(table: pd.DataFrame, periods: dict):
    """Mean generation by hour of day per source group, stacked, one panel per period (same y-axis).
    `periods` maps a label to a boolean mask over the table index."""
    groups = grouped_generation(table)
    fig, axes = plt.subplots(1, len(periods), figsize=(12, 4.8), sharey=True, facecolor=SURFACE)
    for ax, (label, mask) in zip(np.atleast_1d(axes), periods.items()):
        profile = groups[mask].groupby(groups[mask].index.hour).mean()
        ax.stackplot(profile.index, profile.T.values, labels=profile.columns,
                     colors=[GROUP_COLORS[g] for g in profile.columns], edgecolor=SURFACE, linewidth=0.6)
        ax.set_xticks(range(0, 24, 4), [f"{h:02d}:00" for h in range(0, 24, 4)])
        style(ax, label, "MWh" if ax is np.atleast_1d(axes)[0] else None, "hour of day")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def monthly_source_share(table: pd.DataFrame, groups=("solar", "wind", "hydro", "natural gas")):
    """Monthly share of total generation for a few source groups (one line each)."""
    gen = grouped_generation(table)
    monthly = gen.resample("MS").sum()
    share = monthly.div(monthly.sum(axis=1), axis=0)[list(groups)]
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=SURFACE)
    for g in groups:
        ax.plot(share.index, share[g], color=GROUP_COLORS[g], linewidth=2, label=g)
        ax.annotate(g, (share.index[-1], share[g].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, va="center")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, loc="upper left")
    style(ax, "Monthly share of total generation by source", "share of generation")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y", tz=TIMEZONE))
    fig.tight_layout()
    return fig, share.round(3)


def autocorrelation(series: pd.Series, max_lag=336, marks=(1, 24, 48, 168)):
    """Autocorrelation of an hourly series for lags 1..max_lag hours; returns (figure, values at `marks`)."""
    lags = np.arange(1, max_lag + 1)
    values = pd.Series([series.autocorr(int(lag)) for lag in lags], index=lags)
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=SURFACE)
    ax.plot(values.index, values.values, color=BLUE, linewidth=1.5)
    for lag in marks:
        ax.annotate(f"{lag} h: {values[lag]:.2f}", (lag, values[lag]), xytext=(4, 6), textcoords="offset points",
                    color=INK, fontsize=9)
        ax.scatter([lag], [values[lag]], color=BLUE, s=20, zorder=3)
    ax.set_xticks(range(0, max_lag + 1, 24))
    style(ax, f"Autocorrelation of {series.name} by lag", "correlation", "lag (hours)")
    fig.tight_layout()
    return fig, values[list(marks)].round(3)


def smf_ptf_spread_by_hour(table: pd.DataFrame):
    """Mean SMF - PTF (TRY/MWh) by hour of day. Positive = balancing price above day-ahead price."""
    spread = table["smf_try"] - table["ptf_try"]
    by_hour = spread.groupby(spread.index.hour).mean()
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=SURFACE)
    ax.bar(by_hour.index, by_hour.values, color=[BLUE if v >= 0 else ORANGE for v in by_hour.values],
           edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    style(ax, "Mean SMF - PTF by hour of day (blue: SMF above PTF, orange: below)", "TRY/MWh", "hour of day")
    fig.tight_layout()
    return fig, spread


def weekday_weekend_profile(table: pd.DataFrame, column="ptf_usd", ylabel="USD/MWh"):
    """Mean of `column` by hour of day, weekdays vs weekends (Saturday and Sunday)."""
    weekend = table.index.dayofweek >= 5
    profile = table[column].groupby([np.where(weekend, "weekend", "weekday"), table.index.hour]).mean().unstack(0)
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=SURFACE)
    for name, color in [("weekday", BLUE), ("weekend", ORANGE)]:
        ax.plot(profile.index, profile[name], color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax.annotate(name, (23, profile.loc[23, name]), xytext=(6, 0), textcoords="offset points", color=INK,
                    va="center")
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.legend(frameon=False, loc="upper left")
    style(ax, f"Mean by hour, weekdays vs weekends ({column})", ylabel, "hour of day")
    fig.tight_layout()
    return fig, profile.round(1)


def yearly_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per year: mean PTF in TRY and USD, implied TRY/USD rate, floor and cap hours."""
    year = table.index.year
    out = pd.DataFrame({
        "hours": table.groupby(year).size(),
        "mean_ptf_try": table["ptf_try"].groupby(year).mean(),
        "mean_ptf_usd": table["ptf_usd"].groupby(year).mean(),
        "floor_hours": table["is_price_floor"].groupby(year).sum(),
        "cap_hours": table["is_price_cap"].groupby(year).sum(),
    })
    positive = table["ptf_usd"] > 0
    out["implied_try_per_usd"] = (table.loc[positive, "ptf_try"] / table.loc[positive, "ptf_usd"]).groupby(
        year[positive]).median()
    return out.round(2)
