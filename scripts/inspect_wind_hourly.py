"""Follow-up wind check: which sub-hourly slots are missing, and how the wind forecast compares
with the hourly wind generation in rt-gen. Read-only diagnostic.

Usage (from the repository root):
    python -m scripts.inspect_wind_hourly
"""

import pandas as pd

from src.load import load_raw


def main():
    pd.set_option("display.width", 200)
    wind = load_raw("wind-forecast")

    # 1) Since the switch to 15-minute data: which clock times are missing, and how often?
    recent = wind.loc["2026-06-01":]
    grid = pd.date_range(recent.index.min(), recent.index.max(), freq="15min")
    missing = grid.difference(recent.index)
    print("=== Missing 15-min slots since June 2026, by clock time")
    print(pd.Series(missing.strftime("%H:%M")).value_counts().head(10).to_string())

    # 2) Hourly mean of the sub-hourly values (MW averaged over an hour = MWh in that hour)
    hourly = wind[["forecast", "generation"]].resample("h").mean()
    gen = load_raw("rt-gen")["wind"]
    both = pd.concat([hourly, gen.rename("rtgen_wind")], axis=1).dropna()
    print("\n=== Hourly wind: wind-forecast table vs rt-gen")
    print(both.median().round(1).to_string())
    print(f"\nmedian ratio generation / rt-gen wind: {(both['generation'] / both['rtgen_wind']).median():.2f}")
    print(both.corr().round(3).to_string())

    # 3) Forecast error of the published wind forecast against its own generation column
    err = (both["forecast"] - both["generation"]).abs()
    print(f"\nmean absolute error forecast vs generation: {err.mean():.0f} MWh "
          f"({err.mean() / both['generation'].mean():.1%} of mean generation)")


if __name__ == "__main__":
    main()
