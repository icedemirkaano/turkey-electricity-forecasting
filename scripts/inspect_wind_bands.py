"""Check what the wind forecast band columns (quarter1-4) are, using the period where they are filled.

If quarter1-4 are the Q5/Q25/Q75/Q95 quantiles named in EPİAŞ's description, actual generation
should fall inside [quarter1, quarter4] about 90% of the time and inside [quarter2, quarter3]
about 50% of the time (calibration check). Read-only diagnostic.

Usage (from the repository root):
    python -m scripts.inspect_wind_bands
"""

import pandas as pd

from src.load import load_raw

BANDS = ["quarter1", "quarter2", "quarter3", "quarter4"]


def main():
    pd.set_option("display.width", 200)
    wind = load_raw("wind-forecast")
    all_zero = (wind[BANDS] == 0).all(axis=1)

    # 1) Exactly when do the bands stop (and do they restart)? Share of filled rows per day.
    filled_share = (~all_zero).groupby(wind.index.date).mean()
    changes = filled_share[(filled_share > 0.5) != (filled_share.shift() > 0.5)]
    print("=== Days where band availability switches (share of rows with bands)")
    print(changes.round(2).to_string())

    # 2) Calibration in the filled period
    data = wind.loc[~all_zero].dropna(subset=BANDS + ["generation"])
    inside_outer = data["generation"].between(data["quarter1"], data["quarter4"]).mean()
    inside_inner = data["generation"].between(data["quarter2"], data["quarter3"]).mean()
    below = {q: (data["generation"] < data[q]).mean() for q in BANDS}
    print(f"\n=== Calibration on {len(data):,} rows with bands")
    print(f"generation inside [quarter1, quarter4]: {inside_outer:.1%}   (≈90% if Q5–Q95)")
    print(f"generation inside [quarter2, quarter3]: {inside_inner:.1%}   (≈50% if Q25–Q75)")
    print("share of rows where generation is below each band (≈5%, 25%, 75%, 95% if quantiles):")
    for q, share in below.items():
        print(f"  {q}: {share:.1%}")

    # 3) Same check per year, to see if it is stable over time
    yearly = data.groupby(data.index.year).apply(
        lambda d: pd.Series({
            "rows": len(d),
            "inside_q1_q4": d["generation"].between(d["quarter1"], d["quarter4"]).mean(),
            "inside_q2_q3": d["generation"].between(d["quarter2"], d["quarter3"]).mean(),
        }))
    print("\n=== Per year")
    print(yearly.round(3).to_string())


if __name__ == "__main__":
    main()
