"""Project-wide settings: paths, date range and the EPİAŞ tables we use."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

TIMEZONE = "Europe/Istanbul"
START_MONTH = "2023-01"  # first month of the dataset

# Maximum price limit (cap) of the day-ahead and balancing power markets, TL/MWh, by first delivery day.
# Source: EPİAŞ announcements of EPDK Board decisions; minimum limit is 0 TL/MWh throughout.
# Checked against the data (scripts/check_price_caps.py): every monthly maximum equals the cap, no hour above it.
PRICE_CAPS = [
    ("2023-01-01", 4200),  # Kurul Kararı 11499, 29.12.2022
    ("2023-02-01", 3650),  # Kurul Kararı 11588, 26.01.2023
    ("2023-03-01", 3050),  # Kurul Kararı 11682, 28.02.2023
    ("2023-04-01", 2600),  # Kurul Kararı 11772, 30.03.2023
    ("2023-07-04", 2700),  # Kurul Kararı 11933, 03.07.2023
    ("2024-07-01", 3000),  # Kurul Kararı 12716, 27.06.2024
    ("2025-04-05", 3400),  # Kurul Kararı 13423, 27.03.2025
    ("2026-04-04", 4500),  # Kurul Kararı 14459, 02.04.2026
]
PRICE_FLOOR = 0

# EPİAŞ Transparency Platform tables (eptr2 call keys) used in the price (PTF) phase.
# Descriptions follow EPİAŞ's own definitions.
TABLES = {
    "mcp": "Market Clearing Price (PTF), day-ahead market, hourly",
    "smp": "System Marginal Price (SMF), balancing power market, hourly",
    "wap": "Intraday market weighted average price, hourly",
    "dam-clearing": "Day-ahead market matched quantity, hourly",
    "rt-cons": "Real-time consumption, hourly",
    "load-plan": "Load forecast plan for the next day, hourly",
    "rt-gen": "Real-time generation by source, hourly",
    "wind-forecast": "Wind generation and forecast with quantile bands, 15-min",
}
