"""Inspect the wind forecast table: time step per month, real gaps, and the quarter1-4 columns.

Read-only diagnostic. Usage (from the repository root):
    python -m scripts.inspect_wind
"""

import pandas as pd

from src.load import load_raw
from src.quality import infer_step, missing_timestamps

BANDS = ["quarter1", "quarter2", "quarter3", "quarter4"]


def main():
    pd.set_option("display.width", 200, "display.max_columns", 20)
    df = load_raw("wind-forecast")
    month = df.index.to_period("M")

    # 1) Time step and real gaps per month (each month checked against its own step)
    rows = []
    for m, part in df.groupby(month):
        step = infer_step(part.index)
        rows.append({"month": str(m), "rows": len(part), "step": str(step),
                     "missing_on_own_grid": len(missing_timestamps(part.index, step))})
    steps = pd.DataFrame(rows)
    print("=== Time step per month (showing changes and months with gaps)")
    changed = steps["step"].ne(steps["step"].shift())
    print(steps[changed | (steps["missing_on_own_grid"] > 0)].to_string(index=False))

    # 2) quarter1-4 vs forecast
    print("\n=== Band columns")
    all_zero = (df[BANDS] == 0).all(axis=1)
    print(f"rows where all quarter1-4 are 0: {all_zero.sum():,} of {len(df):,}")
    print("months with all-zero bands:", ", ".join(sorted({str(p) for p in month[all_zero]})) or "none")
    bands = df.loc[~all_zero].dropna(subset=BANDS + ["forecast"])
    checks = {
        "quarter1 <= quarter2": (bands["quarter1"] <= bands["quarter2"]).mean(),
        "quarter2 <= quarter3": (bands["quarter2"] <= bands["quarter3"]).mean(),
        "quarter3 <= quarter4": (bands["quarter3"] <= bands["quarter4"]).mean(),
        "quarter2 <= forecast <= quarter3": bands["forecast"].between(bands["quarter2"], bands["quarter3"]).mean(),
        "quarter1 <= forecast <= quarter4": bands["forecast"].between(bands["quarter1"], bands["quarter4"]).mean(),
    }
    for name, share in checks.items():
        print(f"  {name:34s} holds in {share:6.1%} of rows")
    print("\nMedian of each column (rows with bands):")
    print(bands[BANDS + ["forecast", "generation"]].median().round(1).to_string())

    # 3) Nulls by column and month
    print("\n=== Null values by month (only months with nulls)")
    nulls = df.isna().groupby(month).sum()
    print(nulls[nulls.sum(axis=1) > 0].to_string())


if __name__ == "__main__":
    main()
