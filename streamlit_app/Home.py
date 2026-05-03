"""
Scotland 2026 Scottish Parliament Election Forecast — Home Dashboard.

Displays KPI tiles, constituency & regional list polling bars, MRP seat
projection chart, and independence breakdown pie.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1]))

# ── party colour palette ──────────────────────────────────────────────────────
PARTY_COLOURS = {
    "SNP": "#D4A017",
    "Reform": "#1ABC9C",
    "Labour": "#C0392B",
    "Conservative": "#2471A3",
    "LibDem": "#F39C12",
    "Green": "#27AE60",
}

# ── YouGov MRP priors (April 2026) ────────────────────────────────────────────
CONSTITUENCY = {
    "SNP": 35.6, "Reform": 17.9, "Labour": 16.3,
    "Conservative": 10.3, "LibDem": 8.0, "Green": 6.0,
}
REGIONAL_LIST = {
    "SNP": 29.1, "Reform": 18.0, "Labour": 15.3,
    "Conservative": 11.1, "LibDem": 9.0, "Green": 8.5,
}
SEATS_2026 = {
    "SNP": 67, "Reform": 20, "Labour": 15,
    "Green": 11, "Conservative": 9, "LibDem": 7,
}
SEATS_2021 = {
    "SNP": 64, "Conservative": 31, "Labour": 22,
    "Green": 8, "LibDem": 4, "Reform": 0,
}
INDEPENDENCE = {"Yes": 46.6, "No": 44.6, "Undecided": 8.9}

st.set_page_config(
    page_title="Scotland 2026 Forecast",
    page_icon="🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    layout="wide",
)

st.title("🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland 2026 Scottish Parliament Election Forecast")
st.caption("YouGov MRP polling priors · April 2026 · n = 3,925")

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("### Key Metrics")
kpi_cols = st.columns(5)

kpi_data = [
    ("SNP seats (proj.)", "67", "↑3 vs 2021"),
    ("Reform seats (proj.)", "20", "New entrant"),
    ("Yes (independence)", "46.6%", "2pp lead over No"),
    ("SNP majority", "65 needed", "67 projected"),
    ("Tactical voters", "~38%", "Split-ticket estimate"),
]
for col, (label, value, delta) in zip(kpi_cols, kpi_data):
    col.metric(label, value, delta)

st.divider()

# ── Polling bars row ──────────────────────────────────────────────────────────
col_con, col_list = st.columns(2)

parties_sorted = sorted(CONSTITUENCY, key=lambda p: -CONSTITUENCY[p])
colours = [PARTY_COLOURS[p] for p in parties_sorted]


def horizontal_bar(data: dict, title: str) -> go.Figure:
    ps = sorted(data, key=lambda p: -data[p])
    fig = go.Figure(go.Bar(
        x=[data[p] for p in ps],
        y=ps,
        orientation="h",
        marker_color=[PARTY_COLOURS[p] for p in ps],
        text=[f"{data[p]}%" for p in ps],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Vote share (%)",
        xaxis=dict(range=[0, 45]),
        height=320,
        margin=dict(l=0, r=40, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


with col_con:
    st.plotly_chart(horizontal_bar(CONSTITUENCY, "Constituency vote (%)"), use_container_width=True)

with col_list:
    st.plotly_chart(horizontal_bar(REGIONAL_LIST, "Regional list vote (%)"), use_container_width=True)

st.divider()

# ── Seat projection chart ─────────────────────────────────────────────────────
st.markdown("### MRP Seat Projection — 2021 vs 2026")

parties_all = sorted(set(SEATS_2026) | set(SEATS_2021), key=lambda p: -SEATS_2026.get(p, 0))
fig_seats = go.Figure()
fig_seats.add_trace(go.Bar(
    name="2021 result",
    x=parties_all,
    y=[SEATS_2021.get(p, 0) for p in parties_all],
    marker_color=[PARTY_COLOURS[p] for p in parties_all],
    opacity=0.45,
    text=[SEATS_2021.get(p, 0) for p in parties_all],
    textposition="outside",
))
fig_seats.add_trace(go.Bar(
    name="2026 projection",
    x=parties_all,
    y=[SEATS_2026.get(p, 0) for p in parties_all],
    marker_color=[PARTY_COLOURS[p] for p in parties_all],
    text=[SEATS_2026.get(p, 0) for p in parties_all],
    textposition="outside",
))
fig_seats.add_hline(
    y=65, line_dash="dash", line_color="red",
    annotation_text="Majority (65)", annotation_position="top right",
)
fig_seats.update_layout(
    barmode="group",
    xaxis_title="Party",
    yaxis_title="MSP seats",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=0, r=0, t=40, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_seats, use_container_width=True)

st.divider()

# ── Independence pie ──────────────────────────────────────────────────────────
col_pie, col_text = st.columns([1, 2])
with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=list(INDEPENDENCE.keys()),
        values=list(INDEPENDENCE.values()),
        marker_colors=["#27AE60", "#2471A3", "#95A5A6"],
        hole=0.35,
    ))
    fig_pie.update_layout(
        title="Scottish independence polling (%)",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_text:
    st.markdown("""
#### Independence outlook

- **Yes** leads No by **2 percentage points** (46.6% vs 44.6%)
- 8.9% remain undecided — outcome hinges on their break
- Independence ranks **4th** in voter priorities (12%), behind Economy (48%), Health (44%) and Immigration (36%)
- SNP majority (67 seats projected) would renew indyref2 mandate
- YouGov MRP · April 2026 · n = 3,925
    """)

st.divider()
st.caption("Model: Stacking ensemble (XGBoost + LightGBM + CatBoost + RF) → LR meta-learner  |  "
           "Seat allocation: D'Hondt AMS  |  "
           "Navigate the sidebar for interactive tools.")
