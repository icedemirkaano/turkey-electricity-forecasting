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
