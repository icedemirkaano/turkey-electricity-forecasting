"""Check the sign of rt-gen 'importExport' against published daily/weekly import-export figures.

Published references (net = import - export):
  25 Oct 2024 (TEİAŞ data, reported by Dünya):   export 15,632 MWh, import 485 MWh      -> net -15,147
  10-16 Jun 2024 (EPİAŞ weekly report, week 24): export 37,419 MWh, import 103,598 MWh  -> net +66,179
If positive = import and negative = export, our sums should match these nets. Read-only.

Usage (from the repository root):
    python -m scripts.check_import_export_sign
"""

from src.load import load_raw

REFERENCES = [
    ("2024-10-25", "2024-10-25", -15_147),
    ("2024-06-10", "2024-06-16", 66_179),
]


def main():
    flow = load_raw("rt-gen")["importExport"]
    for start, end, published_net in REFERENCES:
        ours = flow.loc[start:end].sum()
        print(f"{start} .. {end}: our sum {ours:>10,.0f} MWh | published net {published_net:>10,} MWh "
              f"| same sign: {(ours > 0) == (published_net > 0)} | ratio {ours / published_net:.2f}")


if __name__ == "__main__":
    main()
