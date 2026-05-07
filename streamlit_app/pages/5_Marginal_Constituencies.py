"""
Page 5 — Marginal Constituencies.

Shows the 20 tightest seats, majority-margin bar chart, tactical swing
probability calculator, and a Scotland region map placeholder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models.marginals import _demo_marginals, tactical_swing_probability, CONSTITUENCIES

PARTY_COLOURS = {
    "SNP": "#D4A017",
    "Reform": "#1ABC9C",
    "Labour": "#C0392B",
    "Conservative": "#2471A3",
    "LibDem": "#F39C12",
    "Green": "#27AE60",
}

st.set_page_config(page_title="Marginal Constituencies", page_icon="⚖️", layout="wide")
st.title("⚖️ Marginal Constituencies")
st.markdown(
    "The 20 tightest seats by predicted majority margin. "
    "A **marginal** is defined as a majority of < 5 percentage points."
)

# ── Load marginal seat data ───────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_marginals() -> pd.DataFrame:
    seats = _demo_marginals(threshold=5.0)
    rows = []
    for s in seats[:20]:
        rows.append(
            {
                "Constituency": s.constituency,
                "Region": s.region,
                "Leading party": s.leading_party,
                "Second party": s.second_party,
                "Margin (pp)": s.majority_margin_pp,
                "Swing needed (pp)": s.swing_needed_pp,
                "Marginal?": "Yes" if s.is_marginal else "No",
                "Tactical rec.": s.tactical_vote_recommendation,
            }
        )
    return pd.DataFrame(rows)


df = load_marginals()

# ── KPI row ───────────────────────────────────────────────────────────────────
n_marginal = (df["Marginal?"] == "Yes").sum()
avg_margin = df["Margin (pp)"].mean()
tightest = df.iloc[0]["Constituency"]
tightest_margin = df.iloc[0]["Margin (pp)"]

col1, col2, col3 = st.columns(3)
col1.metric("Marginal seats (< 5pp)", n_marginal)
col2.metric("Avg margin (top 20)", f"{avg_margin:.1f} pp")
col3.metric("Tightest seat", tightest, f"{tightest_margin:.1f} pp")

st.divider()

# ── Colour-coded table ────────────────────────────────────────────────────────
st.markdown("### Top 20 tightest seats")


def _colour_row(row: pd.Series) -> list[str]:
    colour = PARTY_COLOURS.get(row["Leading party"], "#AAAAAA")
    bg = colour + "30"
    return [f"background-color: {bg}"] * len(row)


styled = (
    df.style.apply(_colour_row, axis=1)
    .format({"Margin (pp)": "{:.2f}", "Swing needed (pp)": "{:.2f}"})
    .set_properties(**{"font-size": "13px"})
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ── Majority margin bar chart ─────────────────────────────────────────────────
st.markdown("### Majority margin by constituency")

bar_colours = [PARTY_COLOURS.get(p, "#AAAAAA") for p in df["Leading party"]]
fig_bar = go.Figure(
    go.Bar(
        x=df["Constituency"],
        y=df["Margin (pp)"],
        marker_color=bar_colours,
        text=df["Leading party"],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Leading: %{text}<br>"
            "Margin: %{y:.1f} pp<extra></extra>"
        ),
    )
)
fig_bar.add_hline(
    y=5.0,
    line_dash="dash",
    line_color="red",
    annotation_text="5pp marginal threshold",
    annotation_position="top right",
)
fig_bar.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Majority margin (pp)",
    height=420,
    margin=dict(t=20, b=140),
    showlegend=False,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Tactical swing calculator ─────────────────────────────────────────────────
st.markdown("### Tactical swing probability")
st.markdown(
    "Select a constituency to see the probability that coordinated tactical "
    "voting would flip the seat from the current leader to the challenger."
)

constituencies = df["Constituency"].tolist()
selected = st.selectbox("Choose constituency", constituencies)

row = df[df["Constituency"] == selected].iloc[0]
from_party = row["Leading party"]
to_party = row["Second party"]
margin = row["Margin (pp)"]

prob = tactical_swing_probability(from_party, to_party, margin)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Current leader", from_party)
col_b.metric("Challenger", to_party)
col_c.metric("Swing to flip", f"{row['Swing needed (pp)']:.1f} pp")

prob_pct = prob * 100
colour = "#27AE60" if prob_pct >= 50 else "#E67E22" if prob_pct >= 25 else "#C0392B"
st.markdown(
    f"**Probability that tactical voting flips {selected}:** "
    f"<span style='color:{colour}; font-size:1.4em; font-weight:bold'>{prob_pct:.1f}%</span>",
    unsafe_allow_html=True,
)

# Gauge chart
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colour},
            "steps": [
                {"range": [0, 25], "color": "#FADBD8"},
                {"range": [25, 50], "color": "#FDEBD0"},
                {"range": [50, 100], "color": "#D5F5E3"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Flip probability"},
    )
)
fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=30, r=30))
st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ── Scotland region map (static SVG placeholder) ─────────────────────────────
st.markdown("### Scotland — seats by region")
st.markdown("_Interactive constituency map coming in Phase 3 (GeoPandas + Folium)._")

region_counts = df.groupby("Region")["Constituency"].count().reset_index()
region_counts.columns = ["Region", "Marginal seats in top 20"]

fig_region = px.bar(
    region_counts.sort_values("Marginal seats in top 20", ascending=True),
    x="Marginal seats in top 20",
    y="Region",
    orientation="h",
    color="Marginal seats in top 20",
    color_continuous_scale="Reds",
    title="Top-20 marginals by region",
)
fig_region.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig_region, use_container_width=True)
