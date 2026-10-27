"""Build the hourly table for the price (PTF) phase from the raw EPİAŞ tables.

One row per hour on the price table's time axis. Every cleaning decision lives in src/clean.py;
this module only applies them in order and joins the results.

COLUMN_AVAILABILITY records, for every column, what is known when tomorrow's forecast is made
(before the 12:30 day-ahead offer deadline). Feature building must respect it:
  target          - what we forecast (never an input for the same hour)
  known_ahead     - published for tomorrow before the forecast -> may be used for the target hour
  past_only       - only known up to "now" -> may only be used with a lag
  evaluation_only - never a model input (used to evaluate other inputs)
"""

import pandas as pd

from src.clean import (LOAD_PLAN_FILL_LAGS_DAYS, WIND_FILL_LAGS_DAYS, clip_negative_generation,
                       fill_from_past_same_hour, to_hourly_wind)
from src.config import PRICE_CAPS, PRICE_FLOOR, PROCESSED_DIR, TIMEZONE
from src.load import load_raw

OUTPUT_PATH = PROCESSED_DIR / "ptf_hourly.parquet"

# rt-gen: columns that are always 0 (no information) and English names for the rest
GEN_DROP = ["naphta", "lng"]
GEN_RENAME = {
    "total": "gen_total", "naturalGas": "gen_natural_gas", "dammedHydro": "gen_dammed_hydro",
    "lignite": "gen_lignite", "river": "gen_river", "importCoal": "gen_import_coal", "wind": "gen_wind",
    "sun": "gen_solar", "fueloil": "gen_fuel_oil", "geothermal": "gen_geothermal",
    "asphaltiteCoal": "gen_asphaltite_coal", "blackCoal": "gen_black_coal", "biomass": "gen_biomass",
    "wasteheat": "gen_waste_heat", "importExport": "net_import",  # + = net import, - = net export
}

COLUMN_AVAILABILITY = {
    "ptf_usd": "target",
    "ptf_try": "past_only", "ptf_eur": "past_only",
    "is_price_floor": "past_only", "is_price_cap": "past_only", "price_cap_try": "known_ahead",
    "smf_try": "past_only", "idm_wap_try": "past_only",
    "dam_matched_bids": "past_only", "dam_matched_offers": "past_only",
    "consumption": "past_only",
    "load_forecast": "known_ahead", "load_forecast_is_filled": "known_ahead",
    "wind_forecast": "known_ahead", "wind_forecast_is_filled": "known_ahead", "wind_n_obs": "known_ahead",
    "wind_generation_monitored": "evaluation_only",
    **{name: "past_only" for name in GEN_RENAME.values()},
}


def price_cap_per_hour(index: pd.DatetimeIndex) -> pd.Series:
    """Cap valid in each hour, from the PRICE_CAPS schedule (first delivery day -> cap)."""
    starts = pd.DatetimeIndex([pd.Timestamp(day, tz=TIMEZONE) for day, _ in PRICE_CAPS])
    caps = pd.Series([cap for _, cap in PRICE_CAPS], index=starts, dtype=float)
    return caps.reindex(index.union(starts)).ffill().reindex(index).rename("price_cap_try")


def build_ptf_hourly() -> tuple[pd.DataFrame, dict]:
    """Return (hourly table, report of what each cleaning step changed)."""
    report = {}
    mcp = load_raw("mcp")
    index = mcp.index

    # Prices (target in USD; TRY/EUR kept for analysis)
    prices = mcp.rename(columns={"price": "ptf_try", "priceUsd": "ptf_usd", "priceEur": "ptf_eur"})
    cap = price_cap_per_hour(index)
    prices["price_cap_try"] = cap
    prices["is_price_floor"] = prices["ptf_try"] == PRICE_FLOOR
    prices["is_price_cap"] = prices["ptf_try"] == cap

    other_prices = pd.concat([
        load_raw("smp")["systemMarginalPrice"].rename("smf_try"),
        load_raw("wap")["wap"].rename("idm_wap_try"),
    ], axis=1)
    matched = load_raw("dam-clearing").rename(columns={"matchedBids": "dam_matched_bids",
                                                        "matchedOffers": "dam_matched_offers"})
    consumption = load_raw("rt-cons")["consumption"]

    # Load forecast: missing day(s) filled from the same weekday/hour of the previous 4 weeks
    load_plan = load_raw("load-plan")["lep"].reindex(index)
    load_forecast, load_filled = fill_from_past_same_hour(load_plan, LOAD_PLAN_FILL_LAGS_DAYS)
    report["load_forecast_filled_hours"] = int(load_filled.sum())

    # Generation: drop empty sources, negative generation -> 0 (net_import keeps its sign)
    gen = load_raw("rt-gen").drop(columns=GEN_DROP)
    gen, clipped = clip_negative_generation(gen)
    report["generation_clipped_values"] = clipped.to_dict()
    gen = gen.rename(columns=GEN_RENAME)

    # Wind forecast: hourly, negatives of the monitored generation -> 0, missing hours filled (previous 7 days)
    wind = to_hourly_wind(load_raw("wind-forecast")).reindex(index)
    wind, wind_clipped = clip_negative_generation(wind)
    report["wind_clipped_values"] = wind_clipped.to_dict()
    wind["wind_forecast"], wind_filled = fill_from_past_same_hour(wind["wind_forecast"], WIND_FILL_LAGS_DAYS)
    wind["wind_n_obs"] = wind["wind_n_obs"].fillna(0).astype(int)
    report["wind_forecast_filled_hours"] = int(wind_filled.sum())

    table = pd.concat([
        prices, other_prices, matched, consumption.rename("consumption"),
        load_forecast.rename("load_forecast"), load_filled.rename("load_forecast_is_filled"),
        gen, wind, wind_filled.rename("wind_forecast_is_filled"),
    ], axis=1).reindex(index)
    table.index.name = "timestamp"

    report["rows"] = len(table)
    report["missing_values"] = {c: int(n) for c, n in table.isna().sum().items() if n}
    return table[list(COLUMN_AVAILABILITY)], report
