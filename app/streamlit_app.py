"""Demo: plan tomorrow's flexible electricity use from a next-day price forecast (Turkey, PTF, USD/MWh).

Two tabs:
  - Tomorrow's plan: the forecast as it was on the day before (cheapest / most expensive hours, a recommendation);
    what actually happened is revealed only on request
  - Can I trust it?: the same forecasts checked against the actual prices over the whole demo period
Run from the repository root:
    streamlit run app/streamlit_app.py
Reads app/data/demo_forecasts.csv (python -m scripts.make_demo_forecasts); trains nothing itself.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `src` importable

import plotly.graph_objects as go
import streamlit as st

from src.demo import BASELINE, FORECAST, days, load_demo, one_day, outcome, period_accuracy, period_shifting, plan

BLUE, ORANGE, GRAY, GREEN, RED = "#2a78d6", "#eb6834", "#a3a29c", "#1baf7a", "#d64545"  # readable on both themes

st.set_page_config(page_title="Tomorrow's electricity plan", layout="wide")


@st.cache_data
def data():
    return load_demo()


def hours_text(hours: list[int]) -> str:
    return ", ".join(f"{h:02d}:00" for h in hours)


demo = data()
available = days(demo)

st.title("Plan tomorrow's electricity use")
st.caption("For a business that can move part of its consumption (a flexible job of a few hours a day). "
           "Prices: Turkish day-ahead market (PTF), USD/MWh. Forecasts by an XGBoost model retrained every Monday, "
           f"for days it never saw ({available[0]:%d %b} – {available[-1]:%d %b %Y}). Data: EPİAŞ Transparency Platform.")

with st.sidebar:
    st.header("Your flexible job")
    n_hours = st.slider("Hours needed per day", min_value=1, max_value=12, value=4)
    power = st.number_input("Power, MW", min_value=0.1, value=5.0, step=0.5)

plan_tab, trust_tab = st.tabs(["Tomorrow's plan", "Can I trust it?"])

# --- Tab 1: the plan, as it looked on the day before ---------------------------------------------------------
with plan_tab:
    day = st.selectbox("Delivery day", available[::-1], format_func=lambda d: f"{d:%A %d %B %Y}")
    today = one_day(demo, day)
    p = plan(today, n_hours, power)
    st.markdown(f"Forecast made on **{day - timedelta(days=1):%A %d %B}**, before the 12:30 offer deadline.")

    colors = [GREEN if h in p["run"] else RED if h in p["avoid"] else GRAY for h in today.index]
    fig = go.Figure()
    fig.add_bar(x=today.index, y=today[FORECAST], marker_color=colors, name="forecast",
                hovertemplate="%{x}:00 — %{y:.1f} USD/MWh<extra></extra>")
    show_actual = st.toggle("Show what actually happened")
    if show_actual:
        fig.add_scatter(x=today.index, y=today["actual"], name="actual", mode="lines+markers",
                        line=dict(color=BLUE, width=2.5))
    fig.update_layout(title="Forecast price for each hour (green: run the job, red: avoid)", xaxis_title="hour",
                      yaxis_title="USD/MWh", height=400, margin=dict(t=50, b=40), showlegend=show_actual,
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

    st.success(f"**Run your {n_hours}-hour job at {hours_text(p['run'])}.** Expected average price "
               f"{p['expected_price']:.1f} USD/MWh, against {p['day_mean']:.1f} for the day as a whole: "
               f"about **{p['expected_saving']:,.0f} USD** less for a {power:g} MW job.")
    st.error(f"**Avoid {hours_text(p['avoid'])}**, the most expensive hours of the day.")

    if show_actual:
        o = outcome(today, n_hours, power)
        st.subheader("How the plan turned out")
        a, b, c = st.columns(3)
        a.metric("Price paid following the plan", f"{o['paid']:.1f} USD/MWh")
        b.metric("Day average (no shifting)", f"{o['day_mean']:.1f} USD/MWh")
        c.metric("Best possible (perfect foresight)", f"{o['perfect']:.1f} USD/MWh")
        captured = 100 * o["saving"] / o["possible_saving"] if o["possible_saving"] > 0 else float("nan")
        st.markdown(f"The plan saved **{o['saving']:,.0f} USD** against not shifting; knowing the prices in advance "
                    f"would have saved {o['possible_saving']:,.0f} USD (**{captured:.0f}%** captured). "
                    f"The forecast's average error on this day: {o['mae']:.1f} USD/MWh.")

# --- Tab 2: evidence over the whole demo period --------------------------------------------------------------
with trust_tab:
    acc = period_accuracy(demo)
    shift = period_shifting(demo, n_hours)
    st.markdown(f"All {acc['days']} demo days, each forecast made on the day before and compared with the actual "
                f"prices afterwards. Reference: *same hour yesterday*, a forecast anyone has for free.")

    a, b, c = st.columns(3)
    a.metric("Model error (MAE, USD/MWh)", f"{acc['mae'][FORECAST]:.2f}")
    b.metric("Same hour yesterday (MAE)", f"{acc['mae'][BASELINE]:.2f}")
    c.metric("Days the model was better", f"{acc['days_better']} of {acc['days']}")

    st.subheader(f"Does following the forecast pay off? ({n_hours}-hour job every day)")
    a, b, c = st.columns(3)
    a.metric("Following the model", f"{shift.loc[FORECAST, 'price_forecast']:.1f} USD/MWh",
             f"{shift.loc[FORECAST, 'captured_%']:.0f}% of the possible saving", delta_color="off")
    b.metric("Following same hour yesterday", f"{shift.loc[BASELINE, 'price_forecast']:.1f} USD/MWh",
             f"{shift.loc[BASELINE, 'captured_%']:.0f}% of the possible saving", delta_color="off")
    c.metric("No shifting (day average)", f"{shift.loc[FORECAST, 'price_no_shifting']:.1f} USD/MWh")
    saving = shift.loc[FORECAST, "saving_per_mwh"] * n_hours * power * acc["days"]
    st.markdown(f"Over {acc['days']} days, a {power:g} MW job of {n_hours} hours a day would have saved about "
                f"**{saving:,.0f} USD** by following the model instead of not shifting. Perfect foresight would have "
                f"paid {shift.loc[FORECAST, 'price_perfect']:.1f} USD/MWh.")

    daily = acc["daily"]
    trend = go.Figure()
    trend.add_bar(x=list(daily.index), y=daily[FORECAST], name="model", marker_color=ORANGE)
    trend.add_bar(x=list(daily.index), y=daily[BASELINE], name="same hour yesterday", marker_color=GRAY)
    trend.update_layout(title="Forecast error by day", barmode="group", yaxis_title="MAE (USD/MWh)", height=340,
                        margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(trend, width="stretch")
