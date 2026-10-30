# Turkey Electricity Forecasting

Forecasting next-day **hourly electricity prices (PTF)** and **consumption** in the Turkish electricity market, using public data from the [EPİAŞ Transparency Platform](https://seffaflik.epias.com.tr).

> 🚧 Work in progress — capstone project of the Workintech Data Science bootcamp.

## Problem

An electricity supplier has to answer two questions every day, before the day-ahead market closes:

1. **How much** electricity will its customers consume tomorrow, hour by hour?
2. **At what price** will it buy that electricity? In the day-ahead market, the price of each hour of the next day is set separately.

Offers for the next day are due by 12:30, and the next day's 24 hourly prices are announced at 14:00. A useful forecast must therefore be made before 12:30, using only the information available at that moment.

## Data

Hourly data from January 2023 onwards, downloaded with the EPİAŞ web service (via the [`eptr2`](https://pypi.org/project/eptr2/) package). Raw data is not stored in this repository; it can be downloaded with the fetch script below.

| Table | Content |
|---|---|
| `mcp` | Market clearing price (PTF), day-ahead market |
| `smp` | System marginal price (SMF), balancing power market |
| `wap` | Intraday market weighted average price |
| `dam-clearing` | Day-ahead market matched quantity |
| `rt-cons` | Real-time consumption |
| `load-plan` | Load forecast plan for the next day |
| `rt-gen` | Real-time generation by source |
| `wind-forecast` | Wind generation forecast with quantile bands |

## Project structure

```
src/            reusable code (config, data fetching, cleaning)
scripts/        command-line entry points
notebooks/      analysis notebooks, one per phase
tests/          unit tests for src/
reports/figures charts used in this README
app/            demo application (planned)
data/           raw / interim / processed data (not committed)
```

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env            # then fill in your EPİAŞ account
python -m scripts.fetch_data    # downloads all tables into data/raw/
python -m scripts.build_dataset # cleans and joins them into data/processed/ptf_hourly.parquet
python -m pytest                # runs the tests
```

Fetching requires a free EPİAŞ Transparency Platform account. The script can be re-run safely: complete months are skipped and incomplete ones are downloaded again.

## Roadmap

- [x] Data inventory and download pipeline
- [x] Price (PTF) data preparation: cleaning decisions and hourly dataset
- [x] Price (PTF) exploratory analysis — [`notebooks/01_ptf_eda.ipynb`](notebooks/01_ptf_eda.ipynb)
- [ ] Consumption analysis
- [ ] Forecasting models (classical ML and deep learning)
- [ ] Demo application

## Data source

EPİAŞ Transparency Platform — Enerji Piyasaları İşletme A.Ş. Data used with attribution, as permitted by the platform.
