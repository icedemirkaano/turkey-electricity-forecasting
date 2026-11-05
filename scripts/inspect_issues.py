"""Details of the open data-quality issues, to discuss in the project review.

Local part (read-only, no API):  python -m scripts.inspect_issues
Ask EPİAŞ again for the gap days: python -m scripts.inspect_issues --refetch
    (needs .env; prints what the API returns today, saves nothing)
"""

import argparse

import pandas as pd
from eptr2 import EPTR2

from src.clean import to_hourly_wind
from src.load import load_raw

GAP_DAYS = {"wind-forecast": "2025-10-29", "load-plan": "2026-05-12"}


def local_checks():
    pd.set_option("display.width", 200)

    # 1) Wind forecast: all hours without any forecast value, and what the raw rows look like that day
    wind = load_raw("wind-forecast")
    hourly = to_hourly_wind(wind)
    empty = hourly.index[hourly["wind_forecast"].isna()]
    print("=== Wind forecast: hours without any forecast value")
    print(pd.Series(empty.strftime("%H:%M")).groupby(empty.date).agg(["count", "first", "last"]).to_string())
    day = wind.loc[GAP_DAYS["wind-forecast"]]
    print(f"\nraw rows on {GAP_DAYS['wind-forecast']}: {len(day)} "
          f"(forecast filled: {day['forecast'].notna().sum()}, generation filled: {day['generation'].notna().sum()})")

    # 2) Load plan gap: are the rows missing, or present but empty?
    plan = load_raw("load-plan")
    print(f"\n=== Load plan rows on {GAP_DAYS['load-plan']}: {len(plan.loc[GAP_DAYS['load-plan']:GAP_DAYS['load-plan']])}")

    # 3) Negative solar generation (rt-gen, column 'sun', MWh): when does it happen?
    sun = load_raw("rt-gen")["sun"]
    neg = sun[sun < 0]
    print(f"\n=== Solar generation < 0: {len(neg)} hours, min {neg.min():.2f} MWh, total {neg.sum():.1f} MWh")
    print("by hour of day:", neg.groupby(neg.index.hour).size().to_dict())
    print("by year-month :", neg.groupby(neg.index.strftime("%Y-%m")).size().to_dict())

    # 4) Negative monitored wind generation (wind-forecast 'generation', hourly mean, MWh)
    gen = hourly["wind_generation_monitored"]
    neg_w = gen[gen < 0]
    print(f"\n=== Monitored wind generation < 0: {len(neg_w)} hours, min {neg_w.min():.2f} MWh")
    print("by year-month :", neg_w.groupby(neg_w.index.strftime("%Y-%m")).size().to_dict())


def refetch():
    eptr = EPTR2(use_dotenv=True, recycle_tgt=True)
    for key, day in GAP_DAYS.items():
        df = eptr.call(key, start_date=day, end_date=day)
        value_col = "forecast" if key == "wind-forecast" else "lep"
        filled = df[value_col].notna().sum() if value_col in df.columns else 0
        print(f"{key} {day}: API returned {len(df)} rows, '{value_col}' filled in {filled}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refetch", action="store_true", help="ask the EPİAŞ API again for the gap days")
    args = parser.parse_args()
    refetch() if args.refetch else local_checks()
