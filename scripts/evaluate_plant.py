"""Round 2 business layer: what the price forecast is worth to a power plant deciding when to run (src/plant.py).

Uses the saved round-2 test forecasts (scripts/round2_test.py, 1 October - 1 November 2026); no model is trained.
Decision sources: the chosen model (hist gradient boosting, weekly retrained), the same hour yesterday, plus the
references perfect foresight and always run.
Assumptions (not measured; shown with alternatives because they are not known):
  - the plant's variable cost is swept over COSTS (USD/MWh), from a cheap plant that should almost always run to
    an expensive one that should almost never run. October's actual prices: median 61.8, 10-90% 30.6-84.9
  - main case: minimum run time 4 hours, start cost 50 USD per start (1 MW); sensitivity: SENSITIVITY
Expectation written before the run: the model captures more of the perfect-foresight profit than the baseline at
every cost; the gap is largest for costs near the typical price (50-70 USD/MWh), where the run / stop decision is
close; at very low and very high costs both are close to perfect, because the answer is obvious.
Outputs (reports/results/round2/business/): plant_by_cost.csv, plant_sensitivity.csv
Usage (from the repository root):
    python -m scripts.evaluate_plant
"""

import pandas as pd

from src.config import ROOT, TIMEZONE
from src.plant import plant_summary

ROUND2 = ROOT / "reports" / "results" / "round2"
OUT_DIR = ROUND2 / "business"
MODEL, BASELINE = "hist gradient boosting (chosen)", "same hour yesterday"
COSTS = list(range(20, 95, 5))
MIN_HOURS, START_COST = 4, 50.0
SENSITIVITY = [(4, 0.0), (4, 50.0), (4, 150.0), (8, 50.0)]
SENSITIVITY_COSTS = [40, 55, 65, 75]


def load_forecasts() -> tuple[pd.Series, dict[str, pd.Series]]:
    frame = pd.read_csv(ROUND2 / "test_predictions.csv", index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
    return frame["actual"], {"model": frame[MODEL], "same hour yesterday": frame[BASELINE]}


def main():
    pd.set_option("display.width", 200)
    actual, forecasts = load_forecasts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"test {actual.index[0]:%Y-%m-%d} .. {actual.index[-1]:%Y-%m-%d}: {len(actual) // 24} days, 1 MW plant")

    rows = []
    for cost in COSTS:
        table = plant_summary(actual, forecasts, cost, MIN_HOURS, START_COST)
        rows += [{"cost": cost, "source": s, **r} for s, r in table.to_dict(orient="index").items()]
    by_cost = pd.DataFrame(rows)
    by_cost.to_csv(OUT_DIR / "plant_by_cost.csv", index=False)
    print(f"\n=== profit (USD) over the test period, min run {MIN_HOURS} h, start cost {START_COST:.0f} USD")
    print(by_cost.pivot(index="cost", columns="source", values="profit").round(0).to_string())
    print("\n=== captured share of the perfect-foresight profit (%)")
    captured = by_cost.pivot(index="cost", columns="source", values="captured_%")
    captured["model - baseline"] = captured["model"] - captured["same hour yesterday"]
    print(captured.drop(columns="perfect foresight").round(1).to_string())
    print("\n=== hours run at a loss (actual price below the cost)")
    print(by_cost.pivot(index="cost", columns="source", values="loss_hours").to_string())

    rows = []
    for min_hours, start_cost in SENSITIVITY:
        for cost in SENSITIVITY_COSTS:
            table = plant_summary(actual, forecasts, cost, min_hours, start_cost)
            rows += [{"min_hours": min_hours, "start_cost": start_cost, "cost": cost, "source": s, **r}
                     for s, r in table.to_dict(orient="index").items()]
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUT_DIR / "plant_sensitivity.csv", index=False)
    print("\n=== sensitivity: captured share (%) under other run rules")
    print(sensitivity[sensitivity["source"].isin(["model", "same hour yesterday"])]
          .pivot_table(index=["min_hours", "start_cost", "cost"], columns="source", values="captured_%")
          .round(1).to_string())
    print(f"\nsaved {OUT_DIR}")


if __name__ == "__main__":
    main()
