# Turkey Electricity Forecasting

Forecasting next-day **hourly electricity prices (PTF)** in the Turkish day-ahead electricity market, using public data from the [EPİAŞ Transparency Platform](https://seffaflik.epias.com.tr).

> 🚧 Work in progress — capstone project of the Workintech Data Science bootcamp.

## Problem

An electricity supplier buys most of tomorrow's electricity in the day-ahead market, where the price of each hour of the next day is set separately. Knowing **at what price** each hour is likely to clear — and which hours will be cheap or expensive — drives its buying cost and risk. This project forecasts those 24 hourly prices. Expected demand enters as an input through EPİAŞ's own next-day load forecast.

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

## Forecasting setup

**Target.** The 24 hourly prices of day D+1 in USD/MWh, forecast on day D before the 12:30 offer deadline.

**Inputs (16), only what is known at forecast time.** Every input is checked against an availability rule in code (`src/features.py`), so a feature that would leak future information raises an error:

| Group | Inputs | Why it is known in time |
|---|---|---|
| Past prices | same hour yesterday, 2 days ago and last week; yesterday's daily mean | Day D's prices were published on D-1 at 14:00 |
| Next-day forecasts | EPİAŞ load forecast, wind forecast, net load (load − wind), price cap | Published for D+1 before the deadline |
| Past solar generation | same hour 2 days ago and last week | Realised generation is known up to D-1 |
| Calendar | hour, day of week, weekend, month, official holidays, holiday-eve afternoon | Known in advance |

**Evaluation.** All models are scored on the same periods, with the same metric:

- **Validation (model choice and tuning):** quarterly walk-forward over 2025. Training always starts in January 2024 and ends the day before each validated quarter.
- **Test:** January–September 2026, scored **once**, after the model was fixed.
- **Demo:** October 2026 onwards, never used for training or model choice.
- **Metric:** mean absolute error (MAE, USD/MWh), the average size of an hourly price error. RMSE and MAE on price-floor hours (price = 0) and price-cap hours are reported alongside.
- **Baseline to beat:** *same hour yesterday*, a forecast anyone has for free.

## Results

### Model choice (validation, 2025)

Mean MAE over the four 2025 quarters (USD/MWh, lower is better):

| Model | Default settings | Tuned (Optuna) |
|---|---|---|
| Dummy (training median) | 13.98 | – |
| Same hour last week | 10.28 | – |
| **Same hour yesterday (baseline)** | **10.14** | – |
| Ridge regression | 8.97 | – |
| Random forest | 7.27 | 7.12 |
| HistGradientBoosting | 7.15 | 6.96 |
| **XGBoost** | 7.19 | **6.88** |
| GRU (last 168 hours → next 24 hours) | 10.47 | – |
| GRU, learning the deviation from the baseline | 8.92 | – |

- **Chosen model: tuned XGBoost**, 32% below the baseline's error.
- The three tree-based models were close. XGBoost had the lowest mean error and was the fastest of the three to train.
- The GRU did not get close. One sample is one forecast day, so the network had only 366–639 training days. Learning the deviation from the baseline instead of the price itself helped, but did not close the gap.

The model relies mostly on past prices (about 59% of the total gain) and on the next-day load and wind forecasts (about 20%):

![Feature importance](reports/figures/models/01_xgb_feature_importance.png)

### Test (January–September 2026, scored once)

| Forecast | MAE | RMSE | vs. baseline |
|---|---|---|---|
| **XGBoost, trained once on 2024–2025** | **18.57** | 24.62 | **52% worse** |
| Same hour yesterday | 12.22 | 18.55 | – |
| Same hour last week | 14.08 | 21.10 | 15% worse |

**The model failed on the test period.** It beat the baseline in five of nine months: January, February and July–September. From March to June it was far worse. From February to June it forecast too high every month, by 14–33 USD/MWh on average.

The cause is a market situation the model had never seen:

- In spring 2026 prices stayed low for weeks, with **395 price-floor hours** (price = 0) in nine months. The whole training period (2024–2025) had **46**.
- In 2024–2025 low prices did not last, and the model had learned that a low price is followed by a higher one.

![Daily mean price, actual vs forecasts](reports/figures/models/02_test_daily_actual_vs_forecasts.png)

The failure is specific to low prices. In the 60–80 USD/MWh range, the most common price level in the test period, the model's error was about half the baseline's (5.3 vs 11.0). Below 20 USD/MWh it was far worse (37.3 vs 9.8):

![MAE by actual price level](reports/figures/models/05_test_mae_by_price_level.png)

### Post-hoc experiment: retraining frequency

> ⚠️ This experiment was decided **after** the test result was known, so it is not the project's test result. Every retraining frequency that was tried is reported here.

Same features, same tuned settings, no new tuning. The model is retrained regularly on all data up to the day before each period it forecasts, as a model in daily use would be:

| Forecast | MAE | vs. baseline | Months better than baseline |
|---|---|---|---|
| Trained once (reported test result) | 18.57 | 52% worse | 5 of 9 |
| Retrained every month | 10.89 | 11% better | 7 of 9 |
| Retrained every week | 9.64 | 21% better | 9 of 9 |
| Same hour yesterday | 12.22 | – | – |

Under weekly retraining, April and May were only marginally better than the baseline (11.55 vs 12.04, 7.50 vs 7.54).

**Practical conclusion:** the model is the same in all three rows; only the update frequency changes. A price model like this must be retrained at least weekly. A single training run is not enough when the market changes.

### Post-hoc experiment: classical time series models and GRU tuning

> ⚠️ Also added **after** the test result was known, to compare more of the methods covered in the bootcamp. The model choice above is unchanged. Every variant that was tried is reported here.

Classical models forecast the price series from its own past. SARIMAX and Prophet also use the next-day net load forecast. All of them forecast D+1 from data known on day D, like the models above:

- **Holt-Winters** (exponential smoothing, additive daily or weekly season). Parameters are estimated at each training. Each day the forecast is updated with the latest prices.
- **SARIMAX** with a daily (24-hour) season. Estimated on the last 8 weeks before the forecast period, because memory grows with the length of the series. Each day its state is updated with the latest prices.
- **Prophet** (daily and weekly season, Turkish holidays). Refitted every day, because it has no cheap way to add new data.

Validation (mean MAE over the four 2025 quarters):

| Model | MAE |
|---|---|
| SARIMAX (1,0,1)(1,0,1,24) | 9.12 |
| SARIMAX (1,0,1)(0,1,1,24) | 9.60 |
| Prophet, with yearly season | 10.00 |
| Same hour yesterday (baseline) | 10.14 |
| Prophet | 10.30 |
| Holt-Winters, daily season | 12.74 |
| Holt-Winters, weekly season | 14.47 |
| *Tuned XGBoost, for comparison* | *6.88* |

On the test period, the best variant of each family was scored under the two protocols above:

| Forecast | Trained once | Retrained every week |
|---|---|---|
| XGBoost | 18.57 | **9.64** |
| SARIMAX (1,0,1)(1,0,1,24) | 15.08 | 10.80 |
| Prophet, with yearly season | 13.48 (refitted daily) | – |
| Holt-Winters, daily season | 16.04 | 16.03 |
| Same hour yesterday | 12.22 | 12.22 |

- **The ranking is the same as in validation:** XGBoost, then SARIMAX, then the baseline.
- **Trained once,** every classical model was better than the frozen XGBoost, because each one updates its forecast with the latest prices every day. None of them beat the baseline.
- **Retrained weekly,** SARIMAX was 12% better than the baseline. It beat weekly XGBoost in February and August.

**Information or model?** XGBoost sees 16 inputs, the classical models one or none. To separate the two reasons, both kinds of model were given the same information (2025 validation, mean MAE):

| | Own past prices only | All inputs |
|---|---|---|
| Time series (SARIMA / SARIMAX) | 9.89 | 10.00 |
| XGBoost | **8.34** | **6.88** |

- **With the same information, XGBoost is still clearly better:** given only past prices and the hour, it reaches 8.34, better than every time series model, and better than ridge with all inputs (8.97).
- **The extra inputs did not help the time series model.** SARIMAX with all inputs (load and wind forecasts, price cap, past solar, holidays, day of week) reached 10.00, worse than with the net load forecast alone (9.12). With the default 50 iterations its estimation did not converge (9.62); with 200 iterations it converged and the result got worse, so the problem is not the estimation. A linear model with a fixed pattern cannot use these inputs the way the price reacts to them (thresholds, the floor and the cap, a different effect of net load at noon and in the evening).
- Thresholds set before the run: time series counted as competitive below 7.38, and "the gap is mostly information" if XGBoost with past prices only were above 8.6. Neither was reached. XGBoost kept its tuned settings with fewer inputs, which slightly favours the full model. Lasso-type autoregressive models from the price forecasting literature were not tried.

**GRU tuning (Optuna).** The rule was set before the search: if the tuned GRU reached a mean validation MAE below 7.5, it would be added to the test table. The search covered history length (3, 7 or 14 days), layers, units, dropout, learning rate and batch size. It was planned for 25 trials and stopped after 15, because the best result had stopped improving and trials took up to an hour. The best settings, trained with three random seeds, reached **8.61** (untuned: 8.92). That is above the threshold, so the GRU was not scored on the test period.

### Business value: shifting flexible load

> Designed after the test result was known. No model was retrained; the saved test forecasts were reused.

A consumer has a flexible job that needs 4 (or 8) hours of 1 MW every day, in any hours. On day D it runs the job in the hours of D+1 that the forecast says are cheapest, and pays the **actual** prices. Two references: no shifting (the day's average price), and perfect foresight (the truly cheapest hours). *Captured share* is the part of the possible saving that the forecast achieved.

Test period, 4 flexible hours per day (average price paid, USD/MWh):

| Forecast | Price paid | Saving vs. no shifting | Captured share | Days worse than no shifting |
|---|---|---|---|---|
| No shifting | 43.09 | – | 0% | – |
| **XGBoost, trained once (reported)** | **24.25** | **18.83** | **85%** | 3 of 273 |
| Same hour last week | 26.01 | 17.08 | 77% | 8 |
| Same hour yesterday | 27.14 | 15.94 | 72% | 13 |
| *XGBoost, retrained weekly (post-hoc)* | *24.32* | *18.77* | *85%* | *2* |
| *SARIMAX, retrained weekly (post-hoc)* | *25.83* | *17.26* | *78%* | *13* |
| Perfect foresight | 20.98 | 22.11 | 100% | – |

With 8 flexible hours the picture is the same (XGBoost trained once 85%, retrained weekly 86%, same hour yesterday 72%).

**The model that failed on MAE was the best at this decision.** For choosing hours only the order of the prices within a day matters, not their level. The frozen model forecast too high in spring, but it still ranked the hours well. It captured more of the possible saving than the baseline in every month except June (92.5% vs 94.3%). The gap was largest in January–March (23–28 percentage points). In April–May every forecast captured 88–97%, because with many hours at the price floor (95 and 214 hours) the cheap hours were easy to find.

### Demo period (October 2026)

The demo uses days that no training run or model choice had seen. Following the retraining result, the tuned XGBoost is retrained every Monday on all data up to the day before:

| Forecast, 1–29 October 2026 | MAE | Days better than the baseline | 4-hour job: captured share of the possible saving |
|---|---|---|---|
| XGBoost, retrained weekly | 7.84 | 19 of 29 | 85% |
| Same hour yesterday | 11.56 | – | 82% |

In October the model was clearly better at the price level (MAE), but only slightly better at picking the cheapest hours: yesterday's daily pattern was already a good guide to today's.

The demo app is built for one user and one decision: a business with a flexible job (a few hours a day) planning tomorrow's consumption.

- **Tomorrow's plan:** the forecast as it looked on the day before, with the hours to run the job (green) and to avoid (red) and the expected saving. What actually happened is revealed only on request.
- **Can I trust it?:** the same forecasts checked against the actual prices over all demo days, against the same-hour-yesterday baseline.

### Limitations

- The validation year (2025) had only 42 price-floor hours, so validation could not reveal the weakness at long low-price periods.
- No weather or hydro reservoir data is used. Hydro had an unusually large share of generation in spring 2026.
- Prices are modelled in USD using EPİAŞ's own USD price column; the exchange rate source behind it is not documented here.
- Solar generation in the real-time generation table is a small share of the total (about 1–4%). Whether it covers all unlicensed solar plants has not been verified.

## Project structure

```
src/              reusable code (data, features, evaluation, models)
scripts/          command-line entry points
notebooks/        analysis notebooks, one per phase
tests/            unit tests for src/
reports/figures/  charts used in this README
reports/results/  scores and predictions (CSV)
app/              demo application (Streamlit)
data/             raw / interim / processed data (not committed)
models/           trained models (not committed)
```

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env            # then fill in your EPİAŞ account
python -m scripts.fetch_data    # downloads all tables into data/raw/
python -m scripts.build_dataset # cleans and joins them into data/processed/ptf_hourly.parquet
python -m pytest                # runs the tests
```

Models and results, in order (each step writes its results to `reports/results/`):

```bash
python -m scripts.evaluate_baselines           # baseline scores on the 2025 validation quarters
python -m scripts.compare_models               # all models with default settings
python -m scripts.tune_models                  # Optuna search (about 25 minutes)
python -m scripts.evaluate_gru                 # GRU variants
python -m scripts.train_final_xgboost          # final model on 2024-2025 -> models/
python -m scripts.evaluate_test                # the one-time test score
python -m scripts.posthoc_retrain --every week # post-hoc retraining experiment (also: --every month)
python -m scripts.analyse_errors               # error analysis figures
python -m scripts.evaluate_timeseries          # classical time series models, validation
python -m scripts.evaluate_timeseries_test     # classical time series models, test (post-hoc)
python -m scripts.information_vs_model         # same information for both model kinds (post-hoc, validation)
python -m scripts.tune_gru                     # Optuna search for the GRU (several hours)
python -m scripts.evaluate_load_shifting       # business layer: load shifting with the test forecasts
```

Demo (October 2026 onwards):

```bash
python -m scripts.make_demo_forecasts          # weekly retrained forecasts -> app/data/demo_forecasts.csv
streamlit run app/streamlit_app.py             # opens the app at http://localhost:8501
```

Fetching requires a free EPİAŞ Transparency Platform account. The script can be re-run safely: complete months are skipped and incomplete ones are downloaded again.

## Roadmap

- [x] Data inventory and download pipeline
- [x] Price (PTF) data preparation: cleaning decisions and hourly dataset
- [x] Price (PTF) exploratory analysis — [`notebooks/01_ptf_eda.ipynb`](notebooks/01_ptf_eda.ipynb)
- [x] Price forecasting models (classical ML, deep learning, classical time series), test and error analysis — [`notebooks/02_ptf_models.ipynb`](notebooks/02_ptf_models.ipynb)
- [x] Business layer: load shifting (imbalance cost left out: it depends on quantity forecast errors, and this project forecasts prices only)
- [x] Demo application (local Streamlit app, October 2026)
- [ ] Publish the demo online
- [ ] _Future work:_ daily retraining; learning the deviation from the baseline with XGBoost; combining XGBoost and SARIMAX; hydro reservoir inputs
- [ ] _Future work:_ consumption forecasting (national hourly demand)

## Data source

EPİAŞ Transparency Platform — Enerji Piyasaları İşletme A.Ş. Data used with attribution, as permitted by the platform.
