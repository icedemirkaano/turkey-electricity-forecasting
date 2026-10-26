"""Download or update the raw EPİAŞ tables.

Usage (from the repository root):
    python -m scripts.fetch_data                          # all tables, START_MONTH -> current month
    python -m scripts.fetch_data --start 2026-10           # only from October 2026
    python -m scripts.fetch_data --tables mcp rt-cons      # only some tables

Requires EPTR_USERNAME and EPTR_PASSWORD in a .env file (see .env.example).
Safe to re-run: complete months are skipped, failed or incomplete ones are retried.
"""

import argparse

import pandas as pd
from eptr2 import EPTR2

from src.config import START_MONTH, TABLES, TIMEZONE
from src.fetch import fetch_table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=START_MONTH, help="first month, YYYY-MM")
    parser.add_argument("--end", default=pd.Timestamp.now(tz=TIMEZONE).strftime("%Y-%m"), help="last month, YYYY-MM")
    parser.add_argument("--tables", nargs="+", default=list(TABLES), choices=list(TABLES))
    args = parser.parse_args()

    eptr = EPTR2(use_dotenv=True, recycle_tgt=True)
    print(f"Fetching {len(args.tables)} tables, {args.start} -> {args.end}")

    failures = {}
    for key in args.tables:
        print(f"\n[{key}] {TABLES[key]}")
        failed = fetch_table(eptr, key, args.start, args.end)
        if failed:
            failures[key] = failed

    if failures:
        print("\nSome months failed (run the command again to retry):")
        for key, months in failures.items():
            print(f"  {key}: {', '.join(months)}")
    else:
        print("\nAll months complete.")


if __name__ == "__main__":
    main()
