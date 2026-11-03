"""Demo: should a power plant prepare to run tomorrow? Decided from a next-day price forecast (Turkey, PTF, USD/MWh).

The day-ahead auction decides hour by hour whether an offer is accepted; what the plant must fix BEFORE the prices
are known (which block to offer, gas, crew, start-up) needs a forecast. Two tabs:
  - Tomorrow's plan: the forecast as it was on the day before, the block to offer (or stay off) and the expected
    profit; what actually happened is revealed only on request
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

from src.demo import BASELINE, FORECAST, days, load_demo, one_day, outcome, period_accuracy, plan, profit_by_cost

BLUE, ORANGE, GRAY, GREEN, RED = "#2a78d6", "#eb6834", "#a3a29c", "#1baf7a", "#d64545"  # readable on both themes
COSTS = list(range(20, 95, 5))

st.set_page_config(page_title="Should the plant run tomorrow?", layout="wide")


@st.cache_data
def data():
    return load_demo()


@st.cache_data
def cost_sweep(min_hours, start_cost):
    return profit_by_cost(data(), COSTS, min_hours, start_cost)


def block_text(block) -> str:
    return f"{block[0]:02d}:00–{block[1] + 1:02d}:00 ({block[1] - block[0] + 1} hours)" if block else "stay off"


demo = data()
available = days(demo)

st.title("Should the plant run tomorrow?")
st.caption("For the operator of a power plant (1 MW, e.g. a gas unit). The auction decides hour by hour whether an "
           "offer is accepted, but the block to offer, the gas to nominate and the crew must be fixed before 12:30, "
           "without knowing the prices. Prices: Turkish day-ahead market (PTF), USD/MWh. Forecasts by a model "
           f"retrained every Monday, for days no model choice used ({available[0]:%d %b} – {available[-1]:%d %b %Y}). "
           "Data: EPİAŞ Transparency Platform; weather data by [Open-Meteo.com](https://open-meteo.com/).")

with st.sidebar:
    st.header("Your plant")
    cost = st.slider("Variable cost (fuel), USD/MWh", min_value=20, max_value=90, value=50, step=5)
    min_hours = st.slider("Minimum run time, hours", min_value=1, max_value=12, value=4)
    start_cost = st.number_input("Cost of one start, USD", min_value=0, max_value=1000, value=50, step=10)
    st.caption("Assumptions, not a real plant's figures. The plant runs at most one block of consecutive hours a "
               "day and does not move the price.")

plan_tab, trust_tab = st.tabs(["Tomorrow's plan", "Can I trust it?"])

# --- Tab 1: the plan, as it looked on the day before ---------------------------------------------------------
with plan_tab:
    day = st.selectbox("Delivery day", available[::-1], format_func=lambda d: f"{d:%A %d %B %Y}")
    today = one_day(demo, day)
    p = plan(today, cost, min_hours, start_cost)
    st.markdown(f"Forecast made on **{day - timedelta(days=1):%A %d %B}**, before the 12:30 offer deadline.")

    colors = [GREEN if h in p["hours"] else (GRAY if today.loc[h, FORECAST] > cost else RED) for h in today.index]
    fig = go.Figure()
    fig.add_bar(x=today.index, y=today[FORECAST], marker_color=colors, name="forecast",
                hovertemplate="%{x}:00 — %{y:.1f} USD/MWh<extra></extra>")
    fig.add_hline(y=cost, line_dash="dash", line_color=ORANGE, annotation_text=f"your cost {cost}",
                  annotation_position="top left")
    show_actual = st.toggle("Show what actually happened")
    if show_actual:
        fig.add_scatter(x=today.index, y=today["actual"], name="actual", mode="lines+markers",
                        line=dict(color=BLUE, width=2.5))
    fig.update_layout(title="Forecast price per hour (green: block to offer, gray: above cost but outside it, red: below cost)",
                      xaxis_title="hour", yaxis_title="USD/MWh", height=400, margin=dict(t=50, b=40),
                      showlegend=show_actual, legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")

    if p["block"]:
        st.success(f"**Prepare to run: offer {block_text(p['block'])}.** Expected profit after the start cost: "
                   f"about **{p['expected_profit']:,.0f} USD** per MW.")
    else:
        st.info(f"**Stay off tomorrow.** No block of at least {min_hours} hours is expected to cover the "
                f"{start_cost} USD start cost at a fuel cost of {cost} USD/MWh.")

    if show_actual:
        o = outcome(today, cost, min_hours, start_cost)
        st.subheader("How the decision turned out (profit at the actual prices, USD per MW)")
        a, b, c = st.columns(3)
        a.metric("Following the model", f"{o['model']['profit']:,.0f}", block_text(o["model"]["block"]),
                 delta_color="off")
        b.metric("Following same hour yesterday", f"{o['same hour yesterday']['profit']:,.0f}",
                 block_text(o["same hour yesterday"]["block"]), delta_color="off")
        c.metric("Best possible (perfect foresight)", f"{o['perfect foresight']['profit']:,.0f}",
                 block_text(o["perfect foresight"]["block"]), delta_color="off")
        error = (today[FORECAST] - today["actual"]).abs().mean()
        st.markdown(f"The forecast's average error on this day: {error:.1f} USD/MWh.")

# --- Tab 2: evidence over the whole demo period --------------------------------------------------------------
with trust_tab:
    acc = period_accuracy(demo)
    sweep = cost_sweep(min_hours, start_cost)
    here = sweep.loc[cost]
    st.markdown(f"All {acc['days']} demo days, each forecast made on the day before and compared with the actual "
                f"prices afterwards. Reference: *same hour yesterday*, a forecast anyone has for free.")

    a, b, c = st.columns(3)
    a.metric("Model error (MAE, USD/MWh)", f"{acc['mae'][FORECAST]:.2f}")
    b.metric("Same hour yesterday (MAE)", f"{acc['mae'][BASELINE]:.2f}")
    c.metric("Days the model was better", f"{acc['days_better']} of {acc['days']}")

    st.subheader(f"Does it pay off for your plant? ({acc['days']} days, USD per MW)")
    a, b, c, d = st.columns(4)
    a.metric("Following the model", f"{here['model']:,.0f}")
    b.metric("Following same hour yesterday", f"{here['same hour yesterday']:,.0f}")
    c.metric("Always run", f"{here['always run']:,.0f}")
    d.metric("Perfect foresight", f"{here['perfect foresight']:,.0f}")
    gain = here["model"] - here["same hour yesterday"]
    st.markdown(f"At a fuel cost of {cost} USD/MWh, following the model instead of yesterday's prices "
                f"{'earned' if gain >= 0 else 'lost'} **{abs(gain):,.0f} USD per MW** over {acc['days']} days.")

    diff = sweep["model"] - sweep["same hour yesterday"]
    bars = go.Figure()
    bars.add_bar(x=sweep.index, y=diff, marker_color=[ORANGE if c_ == cost else GRAY for c_ in sweep.index],
                 hovertemplate="cost %{x}: %{y:,.0f} USD<extra></extra>")
    bars.update_layout(title="Extra profit from the model vs. same hour yesterday, by plant cost (USD per MW)",
                       xaxis_title="variable cost, USD/MWh", yaxis_title="USD", height=340, margin=dict(t=50, b=40))
    bars.update_xaxes(dtick=5)
    st.plotly_chart(bars, width="stretch")
    st.caption("The forecast matters least at the ends: a very cheap plant runs anyway, a very expensive one never "
               "does. In October the gain was largest for plants just below the typical price.")

    daily = acc["daily"]
    trend = go.Figure()
    trend.add_bar(x=list(daily.index), y=daily[FORECAST], name="model", marker_color=ORANGE)
    trend.add_bar(x=list(daily.index), y=daily[BASELINE], name="same hour yesterday", marker_color=GRAY)
    trend.update_layout(title="Price forecast error by day", barmode="group", yaxis_title="MAE (USD/MWh)", height=340,
                        margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(trend, width="stretch")
