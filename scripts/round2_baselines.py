"""Round 2, review question: is "same hour yesterday" a fair bar? Saturday forecast from Friday looks odd.

Adds a third simple forecast suggested in the review, the same hour on the same weekday last year (364 days =
52 weeks earlier, so a Saturday is compared with a Saturday), and breaks the error down by day of week.
Simple forecasts compared with the chosen model (weekly retrained HGB, round 2):
  - same hour yesterday      (lag 24 h)    Saturday <- Friday
  - same hour last week      (lag 168 h)   Saturday <- last Saturday
  - same weekday last year   (lag 8736 h)  Saturday <- the Saturday 52 weeks earlier
  - weekday-aware            yesterday, except on Sundays and Mondays: last week. The rule comes from the
                             validation table (yesterday is worst exactly when the day type changes), so on the
                             test it is a fair, strongest simple forecast; on validation it is optimistic
Periods: the 7 validation quarters (2025Q1-2026Q3) and the round-2 test (1 October - 1 November 2026).
Nothing is trained or changed; the model's forecasts are the saved ones (round2_weekly, round2_test).
Outputs (reports/results/round2/): baselines_overall.csv, baselines_by_weekday.csv
Usage (from the repository root):
    python -m scripts.round2_baselines
"""

import pandas as pd

from src.config import PROCESSED_DIR, ROOT, TIMEZONE
from src.evaluation import ROUND2_FOLDS, ROUND2_TEST_START, mae

ROUND2 = ROOT / "reports" / "results" / "round2"
MODEL = "model (weekly retrained)"
LAGS = {"same hour yesterday": 24, "same hour last week": 168, "same weekday last year": 364 * 24}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_AWARE = "weekday-aware (last week on Sun/Mon)"
LAST_WEEK_DAYS = [6, 0]   # Sunday, Monday


def saved(path, column) -> pd.Series:
    frame = pd.read_csv(path, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame[column].rename(MODEL)


def table_for(price: pd.Series, model: pd.Series) -> pd.DataFrame:
    """Actual price, the model and the three simple forecasts on the model's hours."""
    frame = pd.DataFrame({"actual": price.reindex(model.index), MODEL: model})
    for name, lag in LAGS.items():
        frame[name] = price.shift(lag).reindex(model.index)   # hourly index without gaps: shift = hours
    use_last_week = frame.index.dayofweek.isin(LAST_WEEK_DAYS)
    frame[WEEKDAY_AWARE] = frame["same hour last week"].where(use_last_week, frame["same hour yesterday"])
    if frame.isna().any().any():
        raise ValueError("missing values: a lag reaches before the data starts")
    return frame


def errors(frame: pd.DataFrame) -> pd.Series:
    return pd.Series({name: mae(frame["actual"], frame[name]) for name in [MODEL, *LAGS, WEEKDAY_AWARE]})


def by_weekday(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.groupby(frame.index.dayofweek).apply(errors)
    table.index = [WEEKDAYS[d] for d in table.index]
    return table


def main():
    pd.set_option("display.width", 200)
    price = pd.read_parquet(PROCESSED_DIR / "ptf_hourly.parquet")["ptf_usd"]
    first, last = ROUND2_FOLDS[0][1], ROUND2_FOLDS[-1][2]
    validation = table_for(price, saved(ROUND2 / "weekly_predictions.csv", "hist gradient boosting (tuned, v2)")
                           .loc[first:last])
    test = table_for(price, saved(ROUND2 / "test_predictions.csv", "hist gradient boosting (chosen)")
                     .loc[ROUND2_TEST_START:])

    overall = pd.DataFrame({"validation 2025Q1-2026Q3": errors(validation), "test Oct 2026": errors(test)})
    overall.to_csv(ROUND2 / "baselines_overall.csv", index_label="forecast")
    print("=== MAE (USD/MWh), all hours")
    print(overall.round(2).to_string())

    weekday = pd.concat({"validation": by_weekday(validation), "test": by_weekday(test)})
    weekday.to_csv(ROUND2 / "baselines_by_weekday.csv", index_label=["period", "weekday"])
    print("\n=== MAE by day of the target (validation, 21 months)")
    print(weekday.loc["validation"].round(2).to_string())
    print("\n=== MAE by day of the target (test, 32 days: 4-5 of each weekday)")
    print(weekday.loc["test"].round(2).to_string())
    print(f"\nsaved {ROUND2}")


if __name__ == "__main__":
    main()
