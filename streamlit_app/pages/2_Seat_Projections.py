"""
Page 2 — Seat Projections.

Shows the D'Hondt seat allocation table, a 2021 vs 2026 grouped bar chart,
and a regional list breakdown heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))

PARTY_COLOURS = {
    "SNP": "#D4A017", "Reform": "#1ABC9C", "Labour": "#C0392B",
    "Conservative": "#2471A3", "LibDem": "#F39C12", "Green": "#27AE60",
}
PARTIES = list(PARTY_COLOURS.keys())

SEATS_2026 = {"SNP": 67, "Reform": 20, "Labour": 15, "Green": 11, "Conservative": 9, "LibDem": 7}
SEATS_2021 = {"SNP": 64, "Conservative": 31, "Labour": 22, "Green": 8, "LibDem": 4, "Reform": 0}
CON_2026 = {"SNP": 50, "Reform": 0, "Labour": 10, "Conservative": 8, "LibDem": 5, "Green": 0}
REG_2026 = {"SNP": 17, "Reform": 20, "Labour": 5, "Conservative": 1, "LibDem": 2, "Green": 11}

# Regional list breakdown (derived from D'Hondt model, 8 regions × 7 seats = 56)
REGIONAL_BREAKDOWN = {
    "Central Scotland": {"SNP": 2, "Reform": 2, "Labour": 2, "Conservative": 1, "LibDem": 0, "Green": 0},
    "Glasgow": {"SNP": 2, "Reform": 1, "Labour": 2, "Conservative": 0, "LibDem": 0, "Green": 2},
    "Highlands and Islands": {"SNP": 3, "Reform": 1, "Labour": 1, "Conservative": 1, "LibDem": 1, "Green": 0},
    "Lothian": {"SNP": 1, "Reform": 2, "Labour": 1, "Conservative": 0, "LibDem": 1, "Green": 2},
    "Mid Scotland and Fife": {"SNP": 2, "Reform": 2, "Labour": 1, "Conservative": 1, "LibDem": 1, "Green": 0},
    "North East Scotland": {"SNP": 2, "Reform": 3, "Labour": 1, "Conservative": 1, "LibDem": 0, "Green": 0},
    "South Scotland": {"SNP": 1, "Reform": 3, "Labour": 1, "Conservative": 1, "LibDem": 1, "Green": 0},
    "West Scotland": {"SNP": 4, "Reform": 2, "Labour": 1, "Conservative": 0, "LibDem": 0, "Green": 0},
}

st.set_page_config(page_title="Seat Projections", page_icon="🪑", layout="wide")
st.title("🪑 Seat Projections")
st.caption("📊 YouGov MRP polling priors · April 2026 · Not the ML model output · Not the actual result")
st.markdown("D'Hondt AMS allocation — 73 constituency + 56 regional list = **129 MSPs**")

# ── summary table ─────────────────────────────────────────────────────────────
st.markdown("### Projected seat totals")
st.info(
    "These projections are based on YouGov MRP polling priors (April 2026, n=3,925) — the "
    "pre-election prior used to generate synthetic training data. The ML model projected SNP 73 seats. "
    "Actual certified result (8 May 2026): SNP 58 seats — minority government. "
    "See **Actual Scottish Election Results 2026** for full results."
)
rows = []
for p in sorted(SEATS_2026, key=lambda x: -SEATS_2026[x]):
    change = SEATS_2026[p] - SEATS_2021.get(p, 0)
    rows.append({
        "Party": p,
        "Constituency": CON_2026.get(p, 0),
        "Regional list": REG_2026.get(p, 0),
        "Total 2026": SEATS_2026[p],
        "Total 2021": SEATS_2021.get(p, 0),
        "Change": f"{'+'if change>=0 else ''}{change}",
    })
df_seats = pd.DataFrame(rows)

def colour_row(row):
    c = PARTY_COLOURS.get(row["Party"], "#ccc")
    return [f"background-color:{c}22" for _ in row]

st.dataframe(
    df_seats.style.apply(colour_row, axis=1),
    use_container_width=True,
    hide_index=True,
)

# ── majority line ─────────────────────────────────────────────────────────────
snp_total = SEATS_2026["SNP"]
st.warning(
    f"⚠️ YouGov prior projected SNP **{snp_total} seats** (majority). "
    "ML model projected **73**. Actual result: SNP **58** — MINORITY government. Threshold: 65 seats."
)

st.divider()

# ── Three-way comparison table ─────────────────────────────────────────────────
st.subheader("📊 Three-Way Comparison: Prior vs Model vs Actual")
st.caption(
    "YouGov MRP prior (pre-election polling) · "
    "ML model (stacking ensemble trained on synthetic data) · "
    "Actual certified result (8 May 2026)"
)

_CMP_ROWS = [
    {"Party": "SNP",          "YouGov Prior": 67,  "ML Model (v1.5.0)": 73,  "Actual Result": 58,  "Model Error": -15, "Prior Error": -9},
    {"Party": "Labour",       "YouGov Prior": 15,  "ML Model (v1.5.0)": 16,  "Actual Result": 17,  "Model Error":   1, "Prior Error":  2},
    {"Party": "Reform",       "YouGov Prior": 20,  "ML Model (v1.5.0)": 16,  "Actual Result": 17,  "Model Error":   1, "Prior Error": -3},
    {"Party": "Green",        "YouGov Prior": 11,  "ML Model (v1.5.0)":  8,  "Actual Result": 15,  "Model Error":   7, "Prior Error":  4},
    {"Party": "Conservative", "YouGov Prior":  9,  "ML Model (v1.5.0)":  8,  "Actual Result": 12,  "Model Error":   4, "Prior Error":  3},
    {"Party": "LibDem",       "YouGov Prior":  7,  "ML Model (v1.5.0)":  8,  "Actual Result": 10,  "Model Error":   2, "Prior Error":  3},
    {"Party": "Total",        "YouGov Prior": 129, "ML Model (v1.5.0)": 129, "Actual Result": 129, "Model Error":   0, "Prior Error":  0},
]
df_cmp = pd.DataFrame(_CMP_ROWS)


def _colour_error(val: int) -> str:
    try:
        a = abs(int(val))
    except (ValueError, TypeError):
        return ""
    if a > 5:
        return "background-color:#FADBD8; color:#C0392B; font-weight:bold"
    if a >= 2:
        return "background-color:#FDEBD0; color:#E67E22; font-weight:bold"
    return "background-color:#D5F5E3; color:#1E8449"


def _colour_cmp_row(row: pd.Series) -> list[str]:
    c = PARTY_COLOURS.get(row["Party"], "#aaaaaa")
    return [f"background-color:{c}22"] + [""] * (len(row) - 1)


cmp_styled = (
    df_cmp.style
    .apply(_colour_cmp_row, axis=1)
    .map(_colour_error, subset=["Model Error", "Prior Error"])
    .format({
        "YouGov Prior": "{:d}",
        "ML Model (v1.5.0)": "{:d}",
        "Actual Result": "{:d}",
        "Model Error": "{:+d}",
        "Prior Error": "{:+d}",
    })
    .set_properties(**{"font-size": "13px"})
)
st.dataframe(cmp_styled, use_container_width=True, hide_index=True)
st.info(
    "Model Error = ML Model minus Actual. Prior Error = YouGov Prior minus Actual. "
    "The ML model's primary failure was SNP overestimation (−15) due to no incumbency modelling. "
    "The YouGov prior was closer on SNP (−9) but underestimated Green (+4) and LibDem (+3)."
)

st.divider()

# ── 2021 vs 2026 grouped bar ───────────────────────────────────────────────────
st.markdown("### 2021 Result vs YouGov Prior vs ML Model vs Actual 2026")

ML_MODEL_2026 = {"SNP": 73, "Reform": 16, "Labour": 16, "Green": 8, "Conservative": 8, "LibDem": 8}
ACTUAL_2026 = {"SNP": 58, "Reform": 17, "Labour": 17, "Green": 15, "Conservative": 12, "LibDem": 10}
ACTUAL_COLOURS = {
    "SNP": "#D4A017", "Reform": "#00B4B4", "Labour": "#C0392B",
    "Green": "#27AE60", "Conservative": "#1A5276", "LibDem": "#E67E22",
}

all_parties = sorted(set(SEATS_2026) | set(SEATS_2021), key=lambda p: -SEATS_2026.get(p, 0))
fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    name="2021 Result",
    x=all_parties,
    y=[SEATS_2021.get(p, 0) for p in all_parties],
    marker_color=[PARTY_COLOURS[p] for p in all_parties],
    opacity=0.4,
    text=[SEATS_2021.get(p, 0) for p in all_parties],
    textposition="outside",
))
fig_comp.add_trace(go.Bar(
    name="YouGov Prior",
    x=all_parties,
    y=[SEATS_2026.get(p, 0) for p in all_parties],
    marker_color=[PARTY_COLOURS[p] for p in all_parties],
    text=[SEATS_2026.get(p, 0) for p in all_parties],
    textposition="outside",
))
fig_comp.add_trace(go.Bar(
    name="ML Model",
    x=all_parties,
    y=[ML_MODEL_2026.get(p, 0) for p in all_parties],
    marker_color=[PARTY_COLOURS[p] for p in all_parties],
    opacity=0.7,
    text=[ML_MODEL_2026.get(p, 0) for p in all_parties],
    textposition="outside",
))
fig_comp.add_trace(go.Bar(
    name="Actual 2026",
    x=all_parties,
    y=[ACTUAL_2026.get(p, 0) for p in all_parties],
    marker=dict(
        color=[ACTUAL_COLOURS[p] for p in all_parties],
        line=dict(color="black", width=1.5),
    ),
    text=[ACTUAL_2026.get(p, 0) for p in all_parties],
    textposition="outside",
))
fig_comp.add_hline(y=65, line_dash="dash", line_color="red",
                   annotation_text="Majority (65)")
fig_comp.update_layout(
    barmode="group", height=420,
    yaxis_title="MSP seats",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_comp, use_container_width=True)
st.caption(
    "Bars left to right per party: 2021 Actual | YouGov Prior (April 2026) | "
    "ML Model v1.5.0 | Actual 2026 Result. "
    "Red dashed line = 65 seat majority threshold."
)

st.divider()

# ── regional list breakdown ────────────────────────────────────────────────────
st.markdown("### Regional list seats by region")
st.caption("📊 YouGov MRP Prior projection — pre-election estimate. See actual results below.")
heat_df = pd.DataFrame(REGIONAL_BREAKDOWN).T[PARTIES]
fig_heat = px.imshow(
    heat_df,
    color_continuous_scale="YlOrRd",
    text_auto=True,
    aspect="auto",
    title="Regional list seats won per party per region",
)
fig_heat.update_layout(height=380, margin=dict(l=0, r=0, t=50, b=20))
st.plotly_chart(fig_heat, use_container_width=True)

st.caption(
    "Constituency seats: FPTP winner per constituency.  "
    "Regional seats: D'Hondt, adjusted for constituency wins."
)

st.divider()
st.subheader("🗳️ Actual 2026 Regional List Results")
st.caption("Certified D'Hondt allocation — 8 May 2026")

# Verified totals: SNP=1, Reform=17, Labour=14, Green=13, Con=8, LD=3 = 56
# Each region sums to 7 (8 regions × 7 seats)
ACTUAL_REGIONAL = {
    "Central Scotland":      {"SNP": 0, "Reform": 3, "Labour": 2, "Conservative": 1, "LibDem": 0, "Green": 1},
    "Glasgow":               {"SNP": 0, "Reform": 2, "Labour": 3, "Conservative": 0, "LibDem": 0, "Green": 2},
    "Highlands and Islands": {"SNP": 1, "Reform": 1, "Labour": 1, "Conservative": 1, "LibDem": 1, "Green": 2},
    "Lothian":               {"SNP": 0, "Reform": 2, "Labour": 2, "Conservative": 1, "LibDem": 0, "Green": 2},
    "Mid Scotland and Fife": {"SNP": 0, "Reform": 2, "Labour": 2, "Conservative": 1, "LibDem": 0, "Green": 2},
    "North East Scotland":   {"SNP": 0, "Reform": 3, "Labour": 2, "Conservative": 1, "LibDem": 1, "Green": 0},
    "South Scotland":        {"SNP": 0, "Reform": 3, "Labour": 1, "Conservative": 2, "LibDem": 1, "Green": 0},
    "West Scotland":         {"SNP": 0, "Reform": 1, "Labour": 1, "Conservative": 1, "LibDem": 0, "Green": 4},
}
actual_heat_df = pd.DataFrame(ACTUAL_REGIONAL).T[PARTIES]
fig_actual_heat = px.imshow(
    actual_heat_df,
    color_continuous_scale="YlOrRd",
    text_auto=True,
    aspect="auto",
    title="Actual 2026 regional list seats — certified result (8 May 2026)",
)
fig_actual_heat.update_layout(height=380, margin=dict(l=0, r=0, t=50, b=20))
st.plotly_chart(fig_actual_heat, use_container_width=True)
st.caption(
    "Actual: SNP 1 · Reform 17 · Labour 14 · Green 13 · Conservative 8 · Lib Dem 3 = 56 regional list seats"
)

st.divider()
st.caption("Built by **Debabrata Mishra** · Data Scientist / ML Engineer · "
           "[GitHub](https://github.com/dmishra27) · "
           "[LinkedIn](https://linkedin.com/in/debabrata-mishra) · "
           "[Portfolio](https://bit.ly/m/Debabrata_Mishra_Data_Science_Bio)")
