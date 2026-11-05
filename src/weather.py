"""Temperature forecasts for the price model (round 2, review request).

Source: Open-Meteo Historical Forecast API (https://open-meteo.com/en/docs/historical-forecast-api), archived
weather forecasts from 2022 on, free for non-commercial use, no account needed. The archive stitches the first
hours of each forecast run into one hourly series, so it is somewhat more accurate than a true day-ahead
forecast made on day D for D+1: results with it are slightly optimistic (noted in the README).
National temperature = plain mean of the five largest cities. Equal weights, because official population
figures could not be verified; the cities' temperatures move together, so the weighting matters little.
"""

import pandas as pd
import requests

from src.config import RAW_DIR, TIMEZONE

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_DIR = RAW_DIR / "weather"
CITIES = {  # city centre, latitude / longitude
    "istanbul": (41.01, 28.98),
    "ankara": (39.93, 32.86),
    "izmir": (38.42, 27.14),
    "bursa": (40.19, 29.06),
    "antalya": (36.90, 30.70),
}


def fetch_city(name: str, start: str, end: str) -> pd.Series:
    """Hourly 2 m temperature (°C) for one city, local time."""
    latitude, longitude = CITIES[name]
    response = requests.get(URL, params={"latitude": latitude, "longitude": longitude, "start_date": start,
                                         "end_date": end, "hourly": "temperature_2m", "timezone": TIMEZONE},
                            timeout=120)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"])).tz_localize(TIMEZONE)   # no DST in Turkey since 2016
    return pd.Series(hourly["temperature_2m"], index=index, name=name, dtype=float)


def load_weather() -> pd.DataFrame:
    """Saved city temperatures (data/raw/weather/<city>.csv), one column per city."""
    frames = []
    for name in CITIES:
        frame = pd.read_csv(WEATHER_DIR / f"{name}.csv", index_col="time")
        frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(TIMEZONE)
        frames.append(frame[name])
    return pd.concat(frames, axis=1)


def national_temperature(cities: pd.DataFrame) -> pd.Series:
    return cities.mean(axis=1, skipna=False).rename("temperature_forecast")
