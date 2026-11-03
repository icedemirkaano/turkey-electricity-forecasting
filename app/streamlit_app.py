"""Demo: next-day hourly electricity price forecast for Turkey (PTF, USD/MWh).

Run from the repository root:
    streamlit run app/streamlit_app.py
Reads app/data/demo_forecasts.csv (python -m scripts.make_demo_forecasts); trains nothing itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `src` importable

import plotly.graph_objects as go
import streamlit as st

from src.demo import (BASELINE, FORECAST, buying_cost, day_errors, daily_errors, days, extreme_hours, load_demo,
                      one_day, shifting_result)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"   # readable on light and dark themes

st.set_page_config(page_title="Turkey electricity price forecast", layout="wide")


@st.cache_data
def data():
    return load_demo()


demo = data()
available = days(demo)

st.title("Next-day electricity price forecast, Turkey")
st.caption(f"Day-ahead market price (PTF) in USD/MWh. Each forecast is made on the day before, before the 12:30 "
           f"offer deadline, by an XGBoost model retrained every Monday. The days shown "
           f"({available[0]:%d %b} – {available[-1]:%d %b %Y}) were never used for training or model choice. "
           f"Data: EPİAŞ Transparency Platform.")

with st.sidebar:
    day = st.date_input("Delivery day", value=available[-1], min_value=available[0], max_value=available[-1])
    mwh = st.number_input("Purchase, MWh in every hour", min_value=0.0, value=10.0, step=1.0)
    n_hours = st.slider("Flexible job: hours needed per day", min_value=1, max_value=12, value=4)

if day not in available:
    st.warning("No data for this day.")
    st.stop()
today = one_day(demo, day)
errors = day_errors(today)

# Day at a glance
a, b, c = st.columns(3)
a.metric("Model error, MAE (USD/MWh)", f"{errors[FORECAST]:.1f}")
b.metric("Same hour yesterday, MAE (USD/MWh)", f"{errors[BASELINE]:.1f}",
         delta=f"{errors[BASELINE] - errors[FORECAST]:+.1f} for the model", delta_color="normal")
c.metric("Actual mean price (USD/MWh)", f"{today['actual'].mean():.1f}")

fig = go.Figure()
fig.add_scatter(x=today.index, y=today["actual"], name="actual", line=dict(color=BLUE, width=2.5))
fig.add_scatter(x=today.index, y=today[FORECAST], name="forecast (made the day before)", line=dict(color=ORANGE))
fig.add_scatter(x=today.index, y=today[BASELINE], name="same hour yesterday", line=dict(color=GRAY, dash="dot"))
fig.update_layout(title=f"{day:%A %d %B %Y}: 24 hourly prices", xaxis_title="hour", yaxis_title="USD/MWh",
                  hovermode="x unified", height=420, margin=dict(t=50, b=40),
                  legend=dict(orientation="h", y=-0.2))
fig.update_xaxes(dtick=2)
st.plotly_chart(fig, width="stretch")

# Cheapest and most expensive hours
left, right = st.columns(2)
for column, cheapest, label in [(left, True, "Cheapest"), (right, False, "Most expensive")]:
    hours = extreme_hours(today, n_hours, cheapest).sort_index()
    column.subheader(f"{label} {n_hours} hours (forecast)")
    column.dataframe(hours.style.format("{:.1f}"), width="stretch")

# Cost of buying and of a flexible job
st.subheader("What it means in money")
cost = buying_cost(today, mwh)
shift = shifting_result(today, n_hours)
left, right = st.columns(2)
with left:
    st.markdown(f"**Buying {mwh:g} MWh in every hour of the day**")
    st.metric("Cost expected the day before", f"{cost['forecast']:,.0f} USD")
    st.metric("Actual cost", f"{cost['actual']:,.0f} USD",
              delta=f"{cost['difference']:+,.0f} USD forecast minus actual", delta_color="off")
with right:
    st.markdown(f"**Running a {n_hours}-hour job in the cheapest hours (average price paid)**")
    st.metric("Following the forecast", f"{shift['following the forecast']:.1f} USD/MWh")
    st.metric("Without shifting (day average)", f"{shift['no shifting']:.1f} USD/MWh")
    st.metric("With perfect foresight", f"{shift['perfect foresight']:.1f} USD/MWh")

# Whole demo period
with st.expander("All demo days"):
    period = daily_errors(demo)
    overall = demo[[FORECAST, BASELINE]].sub(demo["actual"], axis=0).abs().mean()
    st.markdown(f"Over {len(period)} days the model's MAE was **{overall[FORECAST]:.2f}** USD/MWh, "
                f"the baseline's **{overall[BASELINE]:.2f}**. The model was better on "
                f"**{int((period[FORECAST] < period[BASELINE]).sum())} of {len(period)}** days.")
    trend = go.Figure()
    trend.add_bar(x=list(period.index), y=period[FORECAST], name="model", marker_color=ORANGE)
    trend.add_bar(x=list(period.index), y=period[BASELINE], name="same hour yesterday", marker_color=GRAY)
    trend.update_layout(barmode="group", yaxis_title="MAE (USD/MWh)", height=320, margin=dict(t=20, b=40),
                        legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(trend, width="stretch")
