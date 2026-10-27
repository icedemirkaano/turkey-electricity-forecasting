"""Build data/processed/ptf_hourly.parquet from the raw tables and print what each step changed.

Usage (from the repository root):
    python -m scripts.build_dataset
"""

from src.dataset import OUTPUT_PATH, build_ptf_hourly


def main():
    table, report = build_ptf_hourly()
    for key, value in report.items():
        print(f"{key}: {value}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT_PATH)
    print(f"\nsaved {OUTPUT_PATH}  ({table.shape[0]:,} rows x {table.shape[1]} columns, "
          f"{table.index.min():%Y-%m-%d %H:%M} -> {table.index.max():%Y-%m-%d %H:%M})")


if __name__ == "__main__":
    main()
