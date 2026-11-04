"""Demo: should a power plant prepare to run tomorrow? Decided from a next-day price forecast (Turkey, PTF, USD/MWh).

The day-ahead auction decides hour by hour whether an offer is accepted; what the plant must fix BEFORE the prices
are known (which block to offer, gas, crew, start-up) needs a forecast. Two tabs:
  - Tomorrow's plan: the forecast as it was on the day before, the block to offer (or stay off) and the expected
    profit; what actually happened is revealed only on request
  - Can I trust it?: the same forecasts checked against the actual prices over the whole demo period
English or Turkish, chosen in the sidebar (texts: src/demo_texts.py).
Run from the repository root:
    streamlit run app/streamlit_app.py
Deep link to a day with the actual prices shown, in Turkish: http://localhost:8501/?day=2026-10-11&reveal=1&lang=tr
Reads app/data/demo_forecasts.csv (python -m scripts.make_demo_forecasts); trains nothing itself.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `src` importable

import plotly.graph_objects as go
import streamlit as st

from src.demo import BASELINE, FORECAST, days, load_demo, one_day, outcome, period_accuracy, plan, profit_by_cost
from src.demo_texts import LANGUAGES, long_date, number, short_date, text

BLUE, ORANGE, GRAY, GREEN, RED = "#2a78d6", "#eb6834", "#a3a29c", "#1baf7a", "#d64545"  # readable on both themes
COSTS = list(range(20, 95, 5))


@st.cache_data
def data():
    return load_demo()


@st.cache_data
def cost_sweep(run_hours, start_price):
    return profit_by_cost(data(), COSTS, run_hours, start_price)


# --- Language: sidebar choice, starting from ?lang=tr / ?lang=en (default English) ---------------------------
codes = list(LANGUAGES)
asked_lang = st.query_params.get("lang", "en")
asked_lang = st.session_state.get("lang", asked_lang if asked_lang in codes else "en")   # widget value wins
st.set_page_config(page_title=text(asked_lang, "page_title"), layout="wide")   # must be the first Streamlit call
lang = st.sidebar.radio("Language / Dil", codes, index=codes.index(asked_lang), format_func=LANGUAGES.get,
                        horizontal=True, key="lang")
st.query_params["lang"] = lang   # keep the choice in the address, so a reload or a shared link keeps it
t = lambda key, **values: text(lang, key, **values)


def block_text(block) -> str:
    if not block:
        return t("off")
    return t("block", first=block[0], end=block[1] + 1, hours=block[1] - block[0] + 1)


demo = data()
available = days(demo)

st.title(t("title"))
st.caption(t("intro", first=short_date(available[0], lang), last=short_date(available[-1], lang, year=True)))

with st.sidebar:
    st.header(t("your_plant"))
    cost = st.slider(t("cost"), min_value=20, max_value=90, value=50, step=5)
    min_hours = st.slider(t("min_hours"), min_value=1, max_value=12, value=4)
    start_cost = st.number_input(t("start_cost"), min_value=0, max_value=1000, value=50, step=10)
    st.caption(t("assumptions"))

plan_tab, trust_tab = st.tabs([t("tab_plan"), t("tab_trust")])

# --- Tab 1: the plan, as it looked on the day before ---------------------------------------------------------
with plan_tab:
    choices = available[::-1]
    asked = st.query_params.get("day")   # deep link, e.g. ?day=2026-10-11&reveal=1
    index = next((i for i, d in enumerate(choices) if f"{d:%Y-%m-%d}" == asked), 0)
    day = st.selectbox(t("delivery_day"), choices, index=index, format_func=lambda d: long_date(d, lang))
    today = one_day(demo, day)
    p = plan(today, cost, min_hours, start_cost)
    st.markdown(t("forecast_made", day=long_date(day - timedelta(days=1), lang, year=False)))

    colors = [GREEN if h in p["hours"] else (GRAY if today.loc[h, FORECAST] > cost else RED) for h in today.index]
    fig = go.Figure()
    fig.add_bar(x=today.index, y=today[FORECAST], marker_color=colors, name=t("forecast"),
                hovertemplate=t("hover_hour") + "<extra></extra>")
    fig.add_hline(y=cost, line_dash="dash", line_color=ORANGE, annotation_text=t("your_cost", cost=cost),
                  annotation_position="top left")
    show_actual = st.toggle(t("show_actual"), value=st.query_params.get("reveal") == "1")
    if show_actual:
        fig.add_scatter(x=today.index, y=today["actual"], name=t("actual"), mode="lines+markers",
                        line=dict(color=BLUE, width=2.5))
    fig.update_layout(title=t("plan_chart"), xaxis_title=t("hour"), yaxis_title="USD/MWh", height=400,
                      margin=dict(t=50, b=40), showlegend=show_actual, legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(dtick=1, range=[-0.6, 23.6])
    st.plotly_chart(fig, width="stretch")

    if p["block"]:
        st.success(t("run", block=block_text(p["block"]), profit=number(p["expected_profit"], lang)))
    else:
        st.info(t("stay_off", hours=min_hours, start=start_cost, cost=cost))

    if show_actual:
        o = outcome(today, cost, min_hours, start_cost)
        st.subheader(t("outcome_title"))
        sources = [("follow_model", "model"), ("follow_baseline", "same hour yesterday"),
                   ("perfect_best", "perfect foresight")]
        for column, (label, name) in zip(st.columns(3), sources):
            column.metric(t(label), number(o[name]["profit"], lang))
            column.caption(t("runs", block=block_text(o[name]["block"])))
        error = (today[FORECAST] - today["actual"]).abs().mean()
        st.markdown(t("day_error", error=number(error, lang, decimals=1)))

# --- Tab 2: evidence over the whole demo period --------------------------------------------------------------
with trust_tab:
    acc = period_accuracy(demo)
    sweep = cost_sweep(min_hours, start_cost)
    here = sweep.loc[cost]
    st.markdown(t("trust_intro", days=acc["days"]))

    a, b, c = st.columns(3)
    a.metric(t("model_mae"), number(acc["mae"][FORECAST], lang, decimals=2))
    b.metric(t("baseline_mae"), number(acc["mae"][BASELINE], lang, decimals=2))
    c.metric(t("days_better"), t("of", wins=acc["days_better"], days=acc["days"]))

    st.subheader(t("pays_off", days=acc["days"]))
    for column, (label, name) in zip(st.columns(4), [("follow_model", "model"),
                                                     ("follow_baseline", "same hour yesterday"),
                                                     ("always_run", "always run"),
                                                     ("perfect", "perfect foresight")]):
        column.metric(t(label), number(here[name], lang))
    gain = here["model"] - here["same hour yesterday"]
    st.markdown(t("gain_earned" if gain >= 0 else "gain_lost", cost=cost, amount=number(abs(gain), lang),
                  days=acc["days"]))

    diff = sweep["model"] - sweep["same hour yesterday"]
    bars = go.Figure()
    bars.add_bar(x=sweep.index, y=diff, marker_color=[ORANGE if c_ == cost else GRAY for c_ in sweep.index],
                 hovertemplate=t("hover_cost") + "<extra></extra>")
    bars.update_layout(title=t("gain_chart"), xaxis_title=t("cost_axis"), yaxis_title="USD", height=340,
                       margin=dict(t=50, b=40))
    bars.update_xaxes(dtick=5)
    st.plotly_chart(bars, width="stretch")
    st.caption(t("gain_note"))

    daily = acc["daily"]
    labels = [short_date(d, lang) for d in daily.index]   # dates in the chosen language (Plotly's would be English)
    trend = go.Figure()
    trend.add_bar(x=labels, y=daily[FORECAST], name=t("model"), marker_color=ORANGE)
    trend.add_bar(x=labels, y=daily[BASELINE], name=t("baseline"), marker_color=GRAY)
    trend.update_layout(title=t("error_chart"), barmode="group", yaxis_title="MAE (USD/MWh)", height=340,
                        margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(trend, width="stretch")
