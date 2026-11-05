"""Demo: tomorrow's electricity price in Turkey (PTF, USD/MWh), read on the day before, and what it is worth.

Three tabs (calculations: src/demo.py, texts in English and Turkish: src/demo_texts.py):
  - Tomorrow's price: the forecast as it was on the day before, as a short bulletin, the tariff-period means, the
    cheapest and priciest hours, the day type and the band where the price usually falls; what actually happened
    is revealed only on request
  - What the cheap hours are worth: a flexible job of n hours a day placed in the cheapest hours by each forecast,
    over October 2026 and the 638 validation days
  - Can I trust it?: the same forecasts checked against the actual prices over the whole demo period
Run from the repository root:
    streamlit run app/streamlit_app.py
Deep link to a day with the actual prices shown, in Turkish: http://localhost:8501/?day=2026-10-18&reveal=1&lang=tr
Reads app/data/ (scripts/make_demo_forecasts.py, scripts/ptf_analysis.py); trains nothing itself.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `src` importable

import plotly.graph_objects as go
import streamlit as st

from src.demo import (BASELINE, FORECAST, day_check, days, hour_ranges, input_change, load_band, load_demo,
                      load_inputs, load_validation, one_day, outcome, outlook, period_accuracy, period_shifting,
                      validation_shifting)
from src.demo_texts import LANGUAGES, hour_text, long_date, number, percent, short_date, text

BLUE, ORANGE, GRAY, GREEN, RED = "#2a78d6", "#eb6834", "#a3a29c", "#1baf7a", "#d64545"  # readable on both themes


@st.cache_data
def data():
    return load_demo(), load_band(), load_inputs(), load_validation()


# --- Language: sidebar choice, starting from ?lang=tr / ?lang=en (default English) ---------------------------
codes = list(LANGUAGES)
asked_lang = st.query_params.get("lang", "en")
asked_lang = st.session_state.get("lang", asked_lang if asked_lang in codes else "en")   # widget value wins
st.set_page_config(page_title=text(asked_lang, "page_title"), layout="wide")   # must be the first Streamlit call
lang = st.sidebar.radio("Language / Dil", codes, index=codes.index(asked_lang), format_func=LANGUAGES.get,
                        horizontal=True, key="lang")
st.query_params["lang"] = lang   # keep the choice in the address, so a reload or a shared link keeps it
t = lambda key, **values: text(lang, key, **values)
usd = lambda value, decimals=1: number(value, lang, decimals=decimals)
hours_of = lambda hours: hour_text(hour_ranges(hours))

demo, band, inputs, validation = data()
available = days(demo)

st.title(t("title"))
st.caption(t("intro", first=short_date(available[0], lang), last=short_date(available[-1], lang, year=True)))

with st.sidebar:
    st.header(t("your_job"))
    n_hours = st.slider(t("n_hours"), min_value=1, max_value=12, value=4)
    st.caption(t("job_note"))

price_tab, value_tab, trust_tab = st.tabs([t("tab_price"), t("tab_value"), t("tab_trust")])

# --- Tab 1: tomorrow's price, as it looked on the day before -------------------------------------------------
with price_tab:
    choices = available[::-1]
    asked = st.query_params.get("day")   # deep link, e.g. ?day=2026-10-18&reveal=1
    index = next((i for i, d in enumerate(choices) if f"{d:%Y-%m-%d}" == asked), 0)
    day = st.selectbox(t("delivery_day"), choices, index=index, format_func=lambda d: long_date(d, lang))
    today = one_day(demo, day)
    view = outlook(today, band, n_hours)
    st.markdown(t("forecast_made", day=long_date(day - timedelta(days=1), lang, year=False)))
    st.info(t("bulletin", day=long_date(day, lang, year=False), mean=usd(view["mean"]), peak=usd(view["peak"]),
              n=n_hours, cheap=hours_of(view["cheapest"]), pricey=hours_of(view["priciest"]), type=t(view["type"])))

    for column, (label, key) in zip(st.columns(5), [("m_mean", "mean"), ("m_day", "day"), ("m_peak", "peak"),
                                                     ("m_night", "night"), ("m_spread", "spread")]):
        column.metric(t(label), usd(view[key]))

    fig = go.Figure()
    fig.add_scatter(x=today.index, y=view["high"], mode="lines", line=dict(width=0), hoverinfo="skip",
                    showlegend=False)
    fig.add_scatter(x=today.index, y=view["low"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(42,120,214,0.15)", name=t("band"), hoverinfo="skip")
    # one bar trace per colour, so the legend says what green, red and gray mean
    groups = [(view["cheapest"], GREEN, t("bars_cheap", n=n_hours)),
              (view["priciest"], RED, t("bars_pricey")),
              ([h for h in today.index if h not in view["cheapest"] + view["priciest"]], GRAY, t("bars_other"))]
    for hours, color, label in groups:
        fig.add_bar(x=hours, y=today.loc[hours, FORECAST], marker_color=color, name=label,
                    hovertemplate=t("hover_hour") + "<extra></extra>")
    show_actual = st.toggle(t("show_actual"), value=st.query_params.get("reveal") == "1")
    if show_actual:
        fig.add_scatter(x=today.index, y=today["actual"], name=t("actual"), mode="lines+markers",
                        line=dict(color=BLUE, width=2.5), hovertemplate=t("hover_hour") + "<extra></extra>")
    fig.update_layout(title=t("price_chart", n=n_hours), xaxis_title=t("hour"), yaxis_title="USD/MWh", height=420,
                      barmode="overlay",   # the three bar traces never share an hour; keep full-width bars
                      margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.28, traceorder="normal"))
    fig.update_xaxes(dtick=1, range=[-0.6, 23.6])
    st.plotly_chart(fig, width="stretch")

    if show_actual:
        result = outcome(today, n_hours)
        st.subheader(t("outcome_title", n=n_hours))
        sources = [("src_model", "model"), ("src_baseline", "same hour yesterday"),
                   ("src_perfect", "perfect foresight"), ("src_none", "no shifting")]
        for column, (label, name) in zip(st.columns(4), sources):
            column.metric(t(label), f"{usd(result[name]['paid'])} USD")
            if result[name]["hours"]:
                column.caption(t("hours_used", hours=hours_of(result[name]["hours"])))
        check = day_check(today, band)
        st.markdown(t("day_check", error=usd(check["error"]), inside=check["inside"],
                      forecast_type=t(check["type_forecast"]), actual_type=t(check["type_actual"])))

    with st.expander(t("changes_title")):
        change = input_change(inputs, day)
        inputs_shown = [("inp_load", "load_forecast", 0, "MWh"), ("inp_wind", "wind_forecast", 0, "MWh"),
                        ("inp_temp", "temperature_forecast", 1, "°C")]
        for column, (label, name, decimals, unit) in zip(st.columns(3), inputs_shown):
            row = change.loc[name]
            column.metric(t(label), f"{number(row['tomorrow'], lang, decimals)} {unit}",
                          f"{number(row['tomorrow'] - row['today'], lang, decimals)} {unit}", delta_color="off")
        st.caption(t("changes_note"))

# --- Tab 2: what the cheap hours are worth -------------------------------------------------------------------
with value_tab:
    st.markdown(t("value_intro", n=n_hours))
    test = period_shifting(demo, n_hours)
    val = validation_shifting(validation, n_hours)
    model_s, base_s = test["summary"]["model"], test["summary"]["same hour yesterday"]
    periods = [
        (t("value_test", days=model_s["days"]),
         [model_s["price_no_shifting"], model_s["price_forecast"], base_s["price_forecast"], model_s["price_perfect"]],
         model_s["captured_%"], base_s["captured_%"], test, test["gap_per_day"]),
        (t("value_validation", days=val["days"]),
         [val["no_shifting"], val["paid"]["model"], val["paid"]["same hour yesterday"], val["perfect"]],
         val["captured"]["model"], val["captured"]["same hour yesterday"], val, val["gap_per_day"]),
    ]
    for title, prices, captured_model, captured_base, counts, gap in periods:
        st.subheader(title)
        st.caption(t("avg_paid"))
        for column, label, value in zip(st.columns(4), ["src_none", "src_model", "src_baseline", "src_perfect"],
                                        prices):
            column.metric(t(label), usd(value))
        st.markdown(t("captured", model=percent(captured_model / 100, lang),
                      baseline=percent(captured_base / 100, lang)))
        st.markdown(t("day_count", model=counts["model_cheaper"], baseline=counts["baseline_cheaper"],
                      equal=counts["equal"]))
    st.markdown(t("per_year", amount=number(val["gap_per_day"] * 365, lang)))

    curve = validation.pivot(index="n_hours", columns="forecast", values="captured_%")
    chart = go.Figure()
    chart.add_scatter(x=curve.index, y=curve["model"], name=t("model"), mode="lines+markers",
                      line=dict(color=ORANGE, width=3), hovertemplate=t("hover_captured") + "<extra></extra>")
    chart.add_scatter(x=curve.index, y=curve["same hour yesterday"], name=t("baseline"), mode="lines+markers",
                      line=dict(color=GRAY, width=3), hovertemplate=t("hover_captured") + "<extra></extra>")
    chart.add_vline(x=n_hours, line_dash="dot", line_color=BLUE)
    chart.update_layout(title=t("captured_chart", days=val["days"]), xaxis_title=t("hours_axis"), yaxis_title="%",
                        height=340, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25),
                        yaxis=dict(range=[50, 100]))
    chart.update_xaxes(dtick=1)
    st.plotly_chart(chart, width="stretch")

# --- Tab 3: evidence over the whole demo period --------------------------------------------------------------
with trust_tab:
    acc = period_accuracy(demo, band, n_hours)
    st.markdown(t("trust_intro", days=acc["days"]))
    a, b, c = st.columns(3)
    a.metric(t("model_mae"), number(acc["mae"][FORECAST], lang, decimals=2))
    b.metric(t("baseline_mae"), number(acc["mae"][BASELINE], lang, decimals=2))
    c.metric(t("days_better"), t("of", wins=acc["days_better"], days=acc["days"]))

    d, e, f = st.columns(3)
    d.metric(t("coverage"), percent(acc["coverage"], lang, decimals=1))
    d.caption(t("coverage_note"))
    e.metric(t("hits", n=n_hours), percent(acc["hits"]["model"], lang))
    e.caption(t("hits_note", baseline=percent(acc["hits"]["baseline"], lang)))
    f.metric(t("types"), percent(acc["types"]["model"], lang))
    f.caption(t("types_note", baseline=percent(acc["types"]["baseline"], lang)))

    by_hour = acc["by_hour"]
    hours_fig = go.Figure()
    hours_fig.add_bar(x=by_hour.index, y=by_hour[FORECAST], name=t("model"), marker_color=ORANGE,
                      hovertemplate=t("hover_hour") + "<extra></extra>")
    hours_fig.add_bar(x=by_hour.index, y=by_hour[BASELINE], name=t("baseline"), marker_color=GRAY,
                      hovertemplate=t("hover_hour") + "<extra></extra>")
    hours_fig.update_layout(title=t("by_hour_chart"), barmode="group", xaxis_title=t("hour"),
                            yaxis_title="MAE (USD/MWh)", height=360, margin=dict(t=50, b=40),
                            legend=dict(orientation="h", y=-0.25))
    hours_fig.update_xaxes(dtick=1)
    st.plotly_chart(hours_fig, width="stretch")
