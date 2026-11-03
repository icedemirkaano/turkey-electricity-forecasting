"""Download archived hourly temperature forecasts for the five largest cities (src/weather.py).

Writes data/raw/weather/<city>.csv. No account needed; re-running downloads everything again (5 requests).
Usage (from the repository root):
    python -m scripts.fetch_weather
"""

import pandas as pd

from src.config import START_MONTH
from src.weather import CITIES, WEATHER_DIR, fetch_city


def main():
    start = f"{START_MONTH}-01"
    end = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")   # the archive lags a little
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    for name in CITIES:
        series = fetch_city(name, start, end)
        series.to_csv(WEATHER_DIR / f"{name}.csv", index_label="time")
        print(f"{name}: {len(series):,} hours {series.index[0]:%Y-%m-%d} -> {series.index[-1]:%Y-%m-%d}, "
              f"missing {int(series.isna().sum())}, mean {series.mean():.1f} °C")


if __name__ == "__main__":
    main()
