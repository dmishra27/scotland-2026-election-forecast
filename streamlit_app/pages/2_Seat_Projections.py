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
st.markdown("D'Hondt AMS allocation — 73 constituency + 56 regional list = **129 MSPs**")

# ── summary table ─────────────────────────────────────────────────────────────
st.markdown("### Projected seat totals")
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
if snp_total >= 65:
    st.success(f"✅ SNP projected at **{snp_total} seats** — majority government (65 needed)")
else:
    st.warning(f"⚠️ SNP projected at **{snp_total} seats** — minority government")

st.divider()

# ── 2021 vs 2026 grouped bar ───────────────────────────────────────────────────
st.markdown("### 2021 result vs 2026 projection")
all_parties = sorted(set(SEATS_2026) | set(SEATS_2021), key=lambda p: -SEATS_2026.get(p, 0))
fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    name="2021 result",
    x=all_parties,
    y=[SEATS_2021.get(p, 0) for p in all_parties],
    marker_color=[PARTY_COLOURS[p] for p in all_parties],
    opacity=0.4,
    text=[SEATS_2021.get(p, 0) for p in all_parties],
    textposition="outside",
))
fig_comp.add_trace(go.Bar(
    name="2026 projection",
    x=all_parties,
    y=[SEATS_2026.get(p, 0) for p in all_parties],
    marker_color=[PARTY_COLOURS[p] for p in all_parties],
    text=[SEATS_2026.get(p, 0) for p in all_parties],
    textposition="outside",
))
fig_comp.add_hline(y=65, line_dash="dash", line_color="red",
                   annotation_text="Majority (65)")
fig_comp.update_layout(
    barmode="group", height=380,
    yaxis_title="MSP seats",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_comp, use_container_width=True)

st.divider()

# ── regional list breakdown ────────────────────────────────────────────────────
st.markdown("### Regional list seats by region")
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
