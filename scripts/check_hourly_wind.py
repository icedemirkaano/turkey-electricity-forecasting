"""Run to_hourly_wind on the real wind table and check the result. Read-only (nothing is saved).

Usage (from the repository root):
    python -m scripts.check_hourly_wind
"""

import pandas as pd

from src.clean import to_hourly_wind
from src.load import load_raw


def main():
    pd.set_option("display.width", 200)
    hourly = to_hourly_wind(load_raw("wind-forecast"))
    mcp_hours = load_raw("mcp").index  # the price table defines the hourly axis we will join on

    print(f"hourly rows: {len(hourly):,}   price table hours: {len(mcp_hours):,}")
    print(f"same hours as the price table: {hourly.index.equals(mcp_hours)}")

    print("\n=== How many sub-hourly values each hour rests on (wind_n_obs)")
    print(hourly["wind_n_obs"].value_counts().sort_index().to_string())

    no_forecast = hourly["wind_forecast"].isna()
    print(f"\nhours with no wind forecast at all: {no_forecast.sum()}")
    if no_forecast.any():
        print(hourly.index[no_forecast].strftime("%Y-%m-%d %H:%M").tolist()[:20])

    print("\n=== Summary of the hourly columns")
    print(hourly.describe().round(1).to_string())


if __name__ == "__main__":
    main()
