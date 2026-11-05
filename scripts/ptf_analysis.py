"""PTF analysis: how useful are the round-2 forecasts for reading the next day's price? (src/analysis.py)

Measured on the validation period (7 quarters 2025Q1-2026Q3, weekly retrained HGB, scripts/round2_weekly.py) and
on the round-2 test (October 2026, scripts/round2_test.py), next to the same-hour-yesterday baseline.
Nothing is trained: the forecasts are the saved ones. The error band and the day-type thresholds use the
validation period only; the test is the check.
Also measured: flexible consumption (src/business.py). A consumer needs n hours of 1 MW a day and puts them in
the n cheapest hours of tomorrow according to a forecast, then pays the actual prices. Depends only on the price.
Outputs:
  reports/results/round2/analysis/: summary_errors.csv, hour_hits.csv, band.csv, band_coverage.csv,
                                    day_types.csv, error_by_hour.csv, shifting.csv, shifting_days.csv
  app/data/ (read by the demo):     error_band.csv, validation_shifting.csv, demo_inputs.csv
Usage (from the repository root):
    python -m scripts.ptf_analysis
"""

import pandas as pd

from src import analysis as an
from src.business import daily_costs, shifting_summary
from src.config import PROCESSED_DIR, ROOT, TIMEZONE

ROUND2 = ROOT / "reports" / "results" / "round2"
OUT = ROUND2 / "analysis"
APP_DATA = ROOT / "app" / "data"
MODEL, BASELINE = "model", "same hour yesterday"
SHIFT_HOURS = range(1, 13)
DEMO_INPUTS = ["load_forecast", "wind_forecast", "temperature_forecast"]   # known the day before (lag 0 inputs)


def _read(name: str) -> pd.DataFrame:
    frame = pd.read_csv(ROUND2 / name, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame


def load_periods() -> dict[str, pd.DataFrame]:
    price = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")["ptf_usd"]
    val = _read("weekly_predictions.csv")
    test = _read("test_predictions.csv")
    return {
        "validation": pd.DataFrame({"actual": price.reindex(val.index), MODEL: val["hist gradient boosting (tuned, v2)"],
                                    BASELINE: val[BASELINE]}),
        "test": pd.DataFrame({"actual": test["actual"], MODEL: test["hist gradient boosting (chosen)"],
                              BASELINE: test[BASELINE]}),
    }


def shifting_tables(periods: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price paid per source and n, and on how many days the model paid less / the same / more than the baseline."""
    rows, day_rows = [], []
    for name, frame in periods.items():
        for n in SHIFT_HOURS:
            costs = {s: daily_costs(frame["actual"], frame[s], n) for s in [MODEL, BASELINE]}
            for source, table in costs.items():
                rows.append({"period": name, "forecast": source, "n_hours": n, **shifting_summary(table, n)})
            gap = costs[BASELINE]["forecast"] - costs[MODEL]["forecast"]
            day_rows.append({"period": name, "n_hours": n, "model cheaper": int((gap > 1e-9).sum()),
                             "equal": int((gap.abs() <= 1e-9).sum()), "baseline cheaper": int((gap < -1e-9).sum()),
                             "mean gap per day": gap.mean()})
    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def main():
    pd.set_option("display.width", 200)
    OUT.mkdir(parents=True, exist_ok=True)
    periods = load_periods()
    summary_rows, hit_rows, type_rows, by_hour = [], [], [], {}

    for name, frame in periods.items():
        actual = an.daily_profile(frame["actual"])
        true_summary, true_type = an.day_summary(actual), an.day_type(actual)
        for source in [MODEL, BASELINE]:
            forecast = an.daily_profile(frame[source])
            summary = an.day_summary(forecast)
            row = {"period": name, "forecast": source, "days": len(actual),
                   "hourly MAE": (frame["actual"] - frame[source]).abs().mean()}
            for col in ["mean", "day", "peak", "night", "spread"]:
                row[f"{col} MAE"] = (summary[col] - true_summary[col]).abs().mean()
            row["priciest hour within 1 h"] = ((summary["priciest_hour"] - true_summary["priciest_hour"]).abs() <= 1).mean()
            summary_rows.append(row)
            for k in [3, 4, 6]:
                for cheapest in [True, False]:
                    hit_rows.append({"period": name, "forecast": source, "k": k,
                                     "hours": "cheapest" if cheapest else "priciest",
                                     "hit rate": an.hour_hit_rate(actual, forecast, k, cheapest).mean()})
            predicted_type = an.day_type(forecast)
            type_rows.append({"period": name, "forecast": source,
                              "type agreement": (predicted_type == true_type).mean(),
                              **{f"actual {t}": (true_type == t).mean() for t, _ in an.DAY_TYPES}})
            if source == MODEL:
                cross = pd.crosstab(true_type.rename("actual"), predicted_type.rename("forecast"))
                print(f"\n{name}: day types, actual (rows) vs model forecast (columns)\n{cross}")
        by_hour[name] = (frame["actual"] - frame[MODEL]).abs().groupby(frame.index.hour).mean()

    val, test = periods["validation"], periods["test"]
    band = an.error_band(val["actual"], val[MODEL])
    coverage = []
    for name, frame in periods.items():
        banded = an.apply_band(frame[MODEL], band)
        coverage.append({"period": name, "coverage": an.band_coverage(frame["actual"], banded),
                         "mean width": (banded["high"] - banded["low"]).mean()})
    # test coverage by week: is the band steady or does it hold only on average?
    banded_test = an.apply_band(test[MODEL], band)
    inside = (test["actual"] >= banded_test["low"]) & (test["actual"] <= banded_test["high"])
    weekly = inside.groupby(inside.index.tz_localize(None).to_period("W-SUN")).mean()
    shifting, shifting_days = shifting_tables(periods)

    tables = {"summary_errors": pd.DataFrame(summary_rows), "hour_hits": pd.DataFrame(hit_rows),
              "band": band, "band_coverage": pd.DataFrame(coverage), "day_types": pd.DataFrame(type_rows),
              "error_by_hour": pd.DataFrame(by_hour), "shifting": shifting, "shifting_days": shifting_days}
    for file, table in tables.items():
        table.to_csv(OUT / f"{file}.csv", index=file in ("band", "error_by_hour"))
        shown = table[table["n_hours"].isin([4, 8])] if "n_hours" in table else table
        print(f"\n{file}\n{shown.round(3).to_string()}")
    print(f"\ntest band coverage by week\n{weekly.round(3).to_string()}")

    # files for the demo app, which reads app/data only
    APP_DATA.mkdir(parents=True, exist_ok=True)
    band.rename_axis("hour").to_csv(APP_DATA / "error_band.csv")
    validation = shifting[shifting["period"] == "validation"].merge(
        shifting_days[shifting_days["period"] == "validation"], on=["period", "n_hours"])
    validation.to_csv(APP_DATA / "validation_shifting.csv", index=False)
    inputs = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")[DEMO_INPUTS]
    first = test.index.min() - pd.Timedelta(days=1)   # the day before the first demo day, for "what changed"
    inputs.loc[first:test.index.max()].to_csv(APP_DATA / "demo_inputs.csv", index_label="time")
    print(f"\nsaved {OUT} and demo files in {APP_DATA}")


if __name__ == "__main__":
    main()
