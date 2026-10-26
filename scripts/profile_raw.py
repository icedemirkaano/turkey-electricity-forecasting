"""Print a quality profile of every raw table: time coverage, gaps, duplicates and value ranges.

Usage (from the repository root):
    python -m scripts.profile_raw
"""

import pandas as pd

from src.config import TABLES
from src.load import load_raw
from src.quality import profile_columns, profile_table


def main():
    pd.set_option("display.width", 200, "display.max_columns", 20)
    summary = []
    for key in TABLES:
        df = load_raw(key)
        prof = profile_table(df)
        summary.append({"table": key, **{k: v for k, v in prof.items() if k != "missing_days"}})
        print(f"\n=== {key}: {prof['rows']:,} rows, step {prof['step']}, "
              f"{prof['first']:%Y-%m-%d %H:%M} -> {prof['last']:%Y-%m-%d %H:%M}")
        print(f"duplicate timestamps: {prof['duplicate_timestamps']}, missing timestamps: {prof['missing_timestamps']}"
              + (f" on {', '.join(map(str, prof['missing_days'][:10]))}" if prof["missing_days"] else ""))
        print(profile_columns(df).round(2).to_string())

    print("\n=== SUMMARY")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
