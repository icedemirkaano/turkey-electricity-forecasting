"""Check a candidate list of maximum price limits (price caps) against the data.

If a cap is right: (1) no hourly PTF in its period is above it, and (2) in tight periods
PTF sits exactly at the cap for several hours. SMF is checked too (the caps apply to both markets).
The candidate list below is UNVERIFIED (from a summary without sources) - this script tests it.
Read-only.

Usage (from the repository root):
    python -m scripts.check_price_caps
"""

import pandas as pd

from src.load import load_raw

# (first delivery day, cap TL/MWh) - all from EPİAŞ announcements of EPDK Board (Kurul) decisions.
CANDIDATE_CAPS = [
    ("2023-01-01", 4200),  # Kurul Kararı 11499, 29.12.2022
    ("2023-02-01", 3650),  # Kurul Kararı 11588, 26.01.2023
    ("2023-03-01", 3050),  # Kurul Kararı 11682, 28.02.2023
    ("2023-04-01", 2600),  # Kurul Kararı 11772, 30.03.2023
    ("2023-07-04", 2700),  # Kurul Kararı 11933, 03.07.2023
    ("2024-07-01", 3000),  # Kurul Kararı 12716, 27.06.2024
    ("2025-04-05", 3400),  # Kurul Kararı 13423, 27.03.2025
    ("2026-04-04", 4500),  # Kurul Kararı 14459, 02.04.2026
]


def main():
    pd.set_option("display.width", 200, "display.max_rows", 100)
    ptf = load_raw("mcp")["price"]
    smf = load_raw("smp")["systemMarginalPrice"]

    # 1) Monthly maximum and how many hours sit exactly at that maximum
    month = ptf.index.strftime("%Y-%m")
    monthly = pd.DataFrame({
        "ptf_max": ptf.groupby(month).max(),
        "hours_at_ptf_max": ptf.groupby(month).apply(lambda s: int((s == s.max()).sum())),
        "smf_max": smf.groupby(smf.index.strftime("%Y-%m")).max(),
    })
    caps = pd.Series({d: c for d, c in CANDIDATE_CAPS}, dtype=float)
    caps.index = pd.to_datetime(caps.index).tz_localize("Europe/Istanbul")
    cap_per_hour = caps.reindex(ptf.index.union(caps.index)).ffill().reindex(ptf.index)
    monthly["candidate_cap"] = cap_per_hour.groupby(month).max()
    print("=== Monthly maximum PTF / SMF vs candidate cap (TL/MWh)")
    print(monthly.to_string())

    # 2) Any hour above its candidate cap? -> the candidate is wrong for that period
    above_ptf = ptf[ptf > cap_per_hour]
    above_smf = smf[smf > cap_per_hour.reindex(smf.index)]
    print(f"\nPTF hours above candidate cap: {len(above_ptf)}; SMF hours above candidate cap: {len(above_smf)}")
    if len(above_ptf):
        print(above_ptf.groupby(above_ptf.index.strftime("%Y-%m")).agg(["count", "max"]).to_string())

    # 3) Exact switch days: first and last day each distinct "high" value appears
    print("\n=== First/last day of each value that PTF hit in 5+ hours (candidates for caps)")
    counts = ptf.value_counts()
    for value in sorted(counts[counts >= 5].index[counts[counts >= 5].index >= 2000]):
        hits = ptf[ptf == value]
        print(f"  {value:>8.2f} TL: {len(hits):4d} hours, {hits.index.min():%d.%m.%Y} -> {hits.index.max():%d.%m.%Y}")


if __name__ == "__main__":
    main()
