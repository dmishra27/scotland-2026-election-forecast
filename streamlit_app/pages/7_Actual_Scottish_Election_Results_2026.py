"""
Page 7 — Actual Scottish (Holyrood) Election Results 2026.

Certified constituency results for all 73 seats declared 8 May 2026.
Includes model accuracy scorecard, region/party filters, and gains table.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

PARTY_COLOURS = {
    "SNP": "#D4A017",
    "Liberal Democrat": "#F39C12",
    "Labour": "#C0392B",
    "Conservative": "#2471A3",
    "Green": "#27AE60",
    "Reform": "#1ABC9C",
}

# ── Certified results — all 73 constituency seats, 8 May 2026 ─────────────────
RESULTS = [
    {"constituency": "Aberdeen Central", "region": "North East Scotland", "winner": "SNP", "majority_pp": 25.67, "turnout_pct": 46.88, "swing": "SNP hold"},
    {"constituency": "Aberdeen Deeside & North Kincardine", "region": "North East Scotland", "winner": "SNP", "majority_pp": 3.60, "turnout_pct": 54.87, "swing": "SNP hold"},
    {"constituency": "Aberdeen Donside", "region": "North East Scotland", "winner": "SNP", "majority_pp": 15.51, "turnout_pct": 47.44, "swing": "SNP hold"},
    {"constituency": "Aberdeenshire East", "region": "North East Scotland", "winner": "SNP", "majority_pp": 2.71, "turnout_pct": 55.63, "swing": "SNP hold"},
    {"constituency": "Aberdeenshire West", "region": "North East Scotland", "winner": "Conservative", "majority_pp": 15.62, "turnout_pct": 60.47, "swing": "C hold"},
    {"constituency": "Airdrie", "region": "Central Scotland", "winner": "SNP", "majority_pp": 17.70, "turnout_pct": 46.76, "swing": "SNP hold"},
    {"constituency": "Almond Valley", "region": "Lothian", "winner": "SNP", "majority_pp": 24.33, "turnout_pct": 50.33, "swing": "SNP hold"},
    {"constituency": "Angus North & Mearns", "region": "North East Scotland", "winner": "SNP", "majority_pp": 11.21, "turnout_pct": 52.66, "swing": "SNP hold"},
    {"constituency": "Angus South", "region": "North East Scotland", "winner": "SNP", "majority_pp": 21.05, "turnout_pct": 53.22, "swing": "SNP hold"},
    {"constituency": "Argyll & Bute", "region": "Highlands and Islands", "winner": "SNP", "majority_pp": 9.26, "turnout_pct": 55.34, "swing": "SNP hold"},
    {"constituency": "Ayr", "region": "South Scotland", "winner": "SNP", "majority_pp": 12.47, "turnout_pct": 56.02, "swing": "SNP hold"},
    {"constituency": "Banffshire & Buchan Coast", "region": "North East Scotland", "winner": "SNP", "majority_pp": 1.23, "turnout_pct": 47.94, "swing": "SNP gain from C"},
    {"constituency": "Bathgate", "region": "Lothian", "winner": "SNP", "majority_pp": 16.74, "turnout_pct": 49.03, "swing": "SNP hold"},
    {"constituency": "Caithness, Sutherland & Ross", "region": "Highlands and Islands", "winner": "Liberal Democrat", "majority_pp": 16.68, "turnout_pct": 53.99, "swing": "LD gain from SNP"},
    {"constituency": "Carrick, Cumnock & Doon Valley", "region": "South Scotland", "winner": "SNP", "majority_pp": 9.05, "turnout_pct": 48.74, "swing": "SNP hold"},
    {"constituency": "Clackmannanshire & Dunblane", "region": "Mid Scotland and Fife", "winner": "SNP", "majority_pp": 13.17, "turnout_pct": 56.67, "swing": "SNP hold"},
    {"constituency": "Clydebank & Milngavie", "region": "West Scotland", "winner": "SNP", "majority_pp": 13.34, "turnout_pct": 57.73, "swing": "SNP hold"},
    {"constituency": "Clydesdale", "region": "South Scotland", "winner": "SNP", "majority_pp": 12.25, "turnout_pct": 56.55, "swing": "SNP hold"},
    {"constituency": "Coatbridge & Chryston", "region": "Central Scotland", "winner": "SNP", "majority_pp": 23.01, "turnout_pct": 49.88, "swing": "SNP hold"},
    {"constituency": "Cowdenbeath", "region": "Mid Scotland and Fife", "winner": "SNP", "majority_pp": 21.02, "turnout_pct": 47.29, "swing": "SNP hold"},
    {"constituency": "Cumbernauld & Kilsyth", "region": "Central Scotland", "winner": "SNP", "majority_pp": 26.95, "turnout_pct": 53.40, "swing": "SNP hold"},
    {"constituency": "Cunninghame North", "region": "West Scotland", "winner": "SNP", "majority_pp": 19.21, "turnout_pct": 53.25, "swing": "SNP hold"},
    {"constituency": "Cunninghame South", "region": "West Scotland", "winner": "SNP", "majority_pp": 14.22, "turnout_pct": 48.62, "swing": "SNP hold"},
    {"constituency": "Dumbarton", "region": "West Scotland", "winner": "Labour", "majority_pp": 5.58, "turnout_pct": 56.84, "swing": "Lab hold"},
    {"constituency": "Dumfriesshire", "region": "South Scotland", "winner": "Conservative", "majority_pp": 3.39, "turnout_pct": 51.55, "swing": "C hold"},
    {"constituency": "Dundee City East", "region": "North East Scotland", "winner": "SNP", "majority_pp": 30.80, "turnout_pct": 46.50, "swing": "SNP hold"},
    {"constituency": "Dundee City West", "region": "North East Scotland", "winner": "SNP", "majority_pp": 24.53, "turnout_pct": 46.19, "swing": "SNP hold"},
    {"constituency": "Dunfermline", "region": "Mid Scotland and Fife", "winner": "SNP", "majority_pp": 15.93, "turnout_pct": 53.12, "swing": "SNP hold"},
    {"constituency": "East Kilbride", "region": "South Scotland", "winner": "SNP", "majority_pp": 14.62, "turnout_pct": 55.40, "swing": "SNP hold"},
    {"constituency": "East Lothian Coast & Lammermuirs", "region": "Lothian", "winner": "SNP", "majority_pp": 1.17, "turnout_pct": 59.66, "swing": "SNP hold"},
    {"constituency": "Eastwood", "region": "West Scotland", "winner": "SNP", "majority_pp": 1.91, "turnout_pct": 65.59, "swing": "SNP gain from C"},
    {"constituency": "Edinburgh Central", "region": "Lothian", "winner": "Green", "majority_pp": 13.00, "turnout_pct": 54.54, "swing": "Green gain from SNP"},
    {"constituency": "Edinburgh Eastern, Musselburgh & Tranent", "region": "Lothian", "winner": "SNP", "majority_pp": 15.84, "turnout_pct": 51.86, "swing": "SNP hold"},
    {"constituency": "Edinburgh North Eastern & Leith", "region": "Lothian", "winner": "SNP", "majority_pp": 8.08, "turnout_pct": 56.97, "swing": "SNP hold"},
    {"constituency": "Edinburgh Northern", "region": "Lothian", "winner": "Liberal Democrat", "majority_pp": 6.65, "turnout_pct": 56.79, "swing": "LD gain from SNP"},
    {"constituency": "Edinburgh North Western", "region": "Lothian", "winner": "Liberal Democrat", "majority_pp": 32.43, "turnout_pct": 62.49, "swing": "LD hold"},
    {"constituency": "Edinburgh Southern", "region": "Lothian", "winner": "Labour", "majority_pp": 12.55, "turnout_pct": 60.98, "swing": "Lab gain from SNP"},
    {"constituency": "Edinburgh South Western", "region": "Lothian", "winner": "SNP", "majority_pp": 10.15, "turnout_pct": 53.28, "swing": "SNP hold"},
    {"constituency": "Ettrick, Roxburgh & Berwickshire", "region": "South Scotland", "winner": "Conservative", "majority_pp": 17.53, "turnout_pct": 52.76, "swing": "C hold"},
    {"constituency": "Falkirk East & Linlithgow", "region": "Lothian", "winner": "SNP", "majority_pp": 14.48, "turnout_pct": 55.08, "swing": "SNP hold"},
    {"constituency": "Falkirk West", "region": "Central Scotland", "winner": "SNP", "majority_pp": 18.65, "turnout_pct": 52.07, "swing": "SNP hold"},
    {"constituency": "Fife Mid & Glenrothes", "region": "Mid Scotland and Fife", "winner": "SNP", "majority_pp": 29.68, "turnout_pct": 46.41, "swing": "SNP hold"},
    {"constituency": "Fife North East", "region": "Mid Scotland and Fife", "winner": "Liberal Democrat", "majority_pp": 40.22, "turnout_pct": 58.42, "swing": "LD hold"},
    {"constituency": "Galloway & Dumfries West", "region": "South Scotland", "winner": "Conservative", "majority_pp": 5.33, "turnout_pct": 51.68, "swing": "C hold"},
    {"constituency": "Glasgow Anniesland", "region": "Glasgow", "winner": "SNP", "majority_pp": 15.07, "turnout_pct": 51.95, "swing": "SNP hold"},
    {"constituency": "Glasgow Baillieston & Shettleston", "region": "Glasgow", "winner": "SNP", "majority_pp": 18.89, "turnout_pct": 46.95, "swing": "SNP hold"},
    {"constituency": "Glasgow Cathcart & Pollok", "region": "Glasgow", "winner": "SNP", "majority_pp": 16.05, "turnout_pct": 49.38, "swing": "SNP hold"},
    {"constituency": "Glasgow Central", "region": "Glasgow", "winner": "SNP", "majority_pp": 38.04, "turnout_pct": 44.86, "swing": "SNP hold"},
    {"constituency": "Glasgow Easterhouse & Springburn", "region": "Glasgow", "winner": "SNP", "majority_pp": 19.98, "turnout_pct": 43.33, "swing": "SNP hold"},
    {"constituency": "Glasgow Kelvin & Maryhill", "region": "Glasgow", "winner": "SNP", "majority_pp": 6.37, "turnout_pct": 55.38, "swing": "SNP hold"},
    {"constituency": "Glasgow Southside", "region": "Glasgow", "winner": "Green", "majority_pp": 8.05, "turnout_pct": 59.09, "swing": "Green gain from SNP"},
    {"constituency": "Hamilton, Larkhall & Stonehouse", "region": "Central Scotland", "winner": "SNP", "majority_pp": 8.65, "turnout_pct": 50.18, "swing": "SNP gain from Lab"},
    {"constituency": "Inverclyde", "region": "West Scotland", "winner": "SNP", "majority_pp": 16.60, "turnout_pct": 51.55, "swing": "SNP hold"},
    {"constituency": "Inverness and Nairn", "region": "Highlands and Islands", "winner": "SNP", "majority_pp": 1.16, "turnout_pct": 54.54, "swing": "SNP gain from Ind"},
    {"constituency": "Kilmarnock & Irvine Valley", "region": "South Scotland", "winner": "SNP", "majority_pp": 15.29, "turnout_pct": 50.35, "swing": "SNP hold"},
    {"constituency": "Kirkcaldy", "region": "Mid Scotland and Fife", "winner": "SNP", "majority_pp": 16.80, "turnout_pct": 45.27, "swing": "SNP hold"},
    {"constituency": "Midlothian North", "region": "Lothian", "winner": "SNP", "majority_pp": 8.63, "turnout_pct": 50.98, "swing": "SNP hold"},
    {"constituency": "Midlothian South, Tweeddale & Lauderdale", "region": "South Scotland", "winner": "SNP", "majority_pp": 20.77, "turnout_pct": 55.20, "swing": "SNP hold"},
    {"constituency": "Moray", "region": "Highlands and Islands", "winner": "SNP", "majority_pp": 8.15, "turnout_pct": 50.81, "swing": "SNP hold"},
    {"constituency": "Motherwell & Wishaw", "region": "Central Scotland", "winner": "SNP", "majority_pp": 20.17, "turnout_pct": 47.27, "swing": "SNP hold"},
    {"constituency": "Na h-Eileanan an Iar", "region": "Highlands and Islands", "winner": "Labour", "majority_pp": 1.25, "turnout_pct": 56.70, "swing": "Lab gain from SNP"},
    {"constituency": "Orkney Islands", "region": "Highlands and Islands", "winner": "Liberal Democrat", "majority_pp": 54.07, "turnout_pct": 57.64, "swing": "LD hold"},
    {"constituency": "Paisley", "region": "West Scotland", "winner": "SNP", "majority_pp": 9.80, "turnout_pct": 50.33, "swing": "SNP hold"},
    {"constituency": "Perthshire North", "region": "Mid Scotland and Fife", "winner": "SNP", "majority_pp": 17.25, "turnout_pct": 58.12, "swing": "SNP gain from C"},
    {"constituency": "Perthshire South & Kinross-shire", "region": "Mid Scotland and Fife", "winner": "SNP", "majority_pp": 13.90, "turnout_pct": 58.91, "swing": "SNP hold"},
    {"constituency": "Renfrewshire North & Cardonald", "region": "West Scotland", "winner": "SNP", "majority_pp": 13.94, "turnout_pct": 53.51, "swing": "SNP hold"},
    {"constituency": "Renfrewshire West & Levern Valley", "region": "West Scotland", "winner": "SNP", "majority_pp": 9.56, "turnout_pct": 53.83, "swing": "SNP hold"},
    {"constituency": "Rutherglen & Cambuslang", "region": "Glasgow", "winner": "SNP", "majority_pp": 17.25, "turnout_pct": 52.21, "swing": "SNP hold"},
    {"constituency": "Shetland Islands", "region": "Highlands and Islands", "winner": "SNP", "majority_pp": 13.21, "turnout_pct": 63.87, "swing": "SNP gain from LD"},
    {"constituency": "Skye, Lochaber & Badenoch", "region": "Highlands and Islands", "winner": "Liberal Democrat", "majority_pp": 2.43, "turnout_pct": 57.47, "swing": "LD gain from SNP"},
    {"constituency": "Stirling", "region": "Mid Scotland and Fife", "winner": "SNP", "majority_pp": 23.16, "turnout_pct": 58.06, "swing": "SNP hold"},
    {"constituency": "Strathkelvin & Bearsden", "region": "West Scotland", "winner": "Liberal Democrat", "majority_pp": 6.47, "turnout_pct": 62.72, "swing": "LD gain from SNP"},
    {"constituency": "Uddingston & Bellshill", "region": "Central Scotland", "winner": "SNP", "majority_pp": 12.43, "turnout_pct": 48.67, "swing": "SNP hold"},
]

# ── Seats that changed hands ──────────────────────────────────────────────────
GAINS = [
    {"constituency": "Caithness, Sutherland & Ross", "from_party": "SNP", "to_party": "Liberal Democrat", "majority_pp": 16.68},
    {"constituency": "Edinburgh Central", "from_party": "SNP", "to_party": "Green", "majority_pp": 13.00},
    {"constituency": "Edinburgh Northern", "from_party": "SNP", "to_party": "Liberal Democrat", "majority_pp": 6.65},
    {"constituency": "Edinburgh Southern", "from_party": "SNP", "to_party": "Labour", "majority_pp": 12.55},
    {"constituency": "Eastwood", "from_party": "Conservative", "to_party": "SNP", "majority_pp": 1.91},
    {"constituency": "Banffshire & Buchan Coast", "from_party": "Conservative", "to_party": "SNP", "majority_pp": 1.23},
    {"constituency": "Hamilton, Larkhall & Stonehouse", "from_party": "Labour", "to_party": "SNP", "majority_pp": 8.65},
    {"constituency": "Perthshire North", "from_party": "Conservative", "to_party": "SNP", "majority_pp": 17.25},
    {"constituency": "Skye, Lochaber & Badenoch", "from_party": "SNP", "to_party": "Liberal Democrat", "majority_pp": 2.43},
    {"constituency": "Shetland Islands", "from_party": "Liberal Democrat", "to_party": "SNP", "majority_pp": 13.21},
    {"constituency": "Strathkelvin & Bearsden", "from_party": "SNP", "to_party": "Liberal Democrat", "majority_pp": 6.47},
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Actual Scottish Election Results 2026",
    page_icon="🗳️",
    layout="wide",
)
st.title("🗳️ Actual Scottish (Holyrood) Election Results 2026")
st.markdown(
    "Certified results — 8 May 2026 · All 129 seats declared · "
    "73 constituency (FPTP) + 56 regional list (D'Hondt)"
)

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("SNP", "58 seats", "57 con + 1 list")
m2.metric("Labour", "17 seats", "3 con + 14 list")
m3.metric("Reform", "17 seats", "0 con + 17 list")
m4.metric("Green", "15 seats", "2 con + 13 list")
m5.metric("Conservative", "12 seats", "4 con + 8 list")
m6.metric("Lib Dem", "10 seats", "7 con + 3 list")

st.info(
    "SNP fell short of majority — 58 total seats vs 65 threshold. "
    "SNP will govern as a minority administration.\n\n"
    "**Constituency:** SNP 57 · LD 7 · Conservative 4 · Labour 3 · Green 2 · Reform 0\n\n"
    "**Regional list:** SNP 1 · Labour 14 · Reform 17 · Green 13 · Conservative 8 · LD 3"
)

st.divider()

# ── Model accuracy scorecard ──────────────────────────────────────────────────
st.subheader("🎯 Model Prediction Accuracy")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**What the model got RIGHT ✅**")
    st.metric("Largest party", "SNP ✅", "Correctly predicted")
    st.metric("Reform constituency seats", "0 ✅", "Correctly predicted")
    st.metric("Conservative regional seats", "8 ✅", "Exact match")
    st.metric("Labour regional seats", "17 ✅", "Model: 16, Actual: 17 (+1)")
    st.metric("Reform regional seats", "17 ✅", "Model: 16, Actual: 17 (+1)")

with col2:
    st.markdown("**What the model got WRONG ❌**")
    st.metric("SNP total seats", "58 ❌", "Model: 73, Actual: 58 (−15)")
    st.metric("SNP majority prediction", "WRONG ❌", "Minority govt, not majority")
    st.metric("LD constituency seats", "7 ❌", "Model: 0, Actual: 7 (+7)")
    st.metric("Green total seats", "15 ❌", "Model: 8, Actual: 15 (+7)")
    st.metric("Conservative total seats", "12 ❌", "Model: 8, Actual: 12 (+4)")

acc_df = pd.DataFrame({
    "Party": ["SNP", "Labour", "Reform", "Green", "Conservative", "LibDem", "TOTAL"],
    "Model Total": [73, 16, 16, 8, 8, 8, 129],
    "Actual Total": [58, 17, 17, 15, 12, 10, 129],
    "Delta": [-15, 1, 1, 7, 4, 2, 0],
    "Accurate?": ["❌", "✅", "✅", "❌", "❌", "✅", "—"],
})


def _colour_delta(val) -> str:
    try:
        if abs(val) > 3:
            return "background-color:#FADBD8; color:#C0392B"
        if abs(val) <= 2:
            return "background-color:#D5F5E3; color:#1E8449"
        return "background-color:#FDEBD0; color:#E67E22"
    except (TypeError, ValueError):
        return ""


acc_styled = (
    acc_df.style
    .map(_colour_delta, subset=["Delta"])
    .format({"Delta": "{:+d}"})
    .set_properties(**{"font-size": "13px"})
)
st.dataframe(acc_styled, use_container_width=True, hide_index=True)
st.caption(
    "Model trained on synthetic data from YouGov MRP April 2026 priors. "
    "Primary failure: no incumbency modelling — LD won 7 constituency seats "
    "the model assigned to SNP. Root cause documented in ELECTION_RESULTS.md"
)

st.divider()

# ── Filter controls ───────────────────────────────────────────────────────────
df_all = pd.DataFrame(RESULTS)
regions = sorted(df_all["region"].unique().tolist())

fc1, fc2 = st.columns(2)
with fc1:
    region_filter = st.selectbox("Filter by region", ["All regions"] + regions)
with fc2:
    party_filter = st.selectbox(
        "Filter by winning party",
        ["All parties", "SNP", "Liberal Democrat", "Labour", "Conservative", "Green", "Reform"],
    )

df_filtered = df_all.copy()
if region_filter != "All regions":
    df_filtered = df_filtered[df_filtered["region"] == region_filter]
if party_filter != "All parties":
    df_filtered = df_filtered[df_filtered["winner"] == party_filter]

df_filtered = df_filtered.sort_values(
    ["region", "majority_pp"], ascending=[True, False]
).reset_index(drop=True)

# ── Results table ─────────────────────────────────────────────────────────────
display_df = df_filtered.rename(columns={
    "constituency": "Constituency",
    "region": "Region",
    "winner": "Winner",
    "majority_pp": "Majority (pp)",
    "turnout_pct": "Turnout %",
    "swing": "Result",
})


def _colour_winner_cell(val: str) -> str:
    colour = PARTY_COLOURS.get(val, "#888888")
    return f"background-color: {colour}33; color: {colour}; font-weight: bold"


table_styled = (
    display_df.style
    .map(_colour_winner_cell, subset=["Winner"])
    .format({"Majority (pp)": "{:.2f}", "Turnout %": "{:.2f}"})
    .set_properties(**{"font-size": "13px"})
)

st.markdown("### Results")
st.dataframe(table_styled, use_container_width=True, hide_index=True)
st.caption(f"Showing {len(df_filtered)} of 73 constituencies")

st.divider()

# ── Gains summary ─────────────────────────────────────────────────────────────
st.subheader("🔄 Seats that changed hands (11 seats)")

gains_df = pd.DataFrame(GAINS).rename(columns={
    "constituency": "Constituency",
    "from_party": "From",
    "to_party": "To",
    "majority_pp": "Majority (pp)",
})


def _colour_party_cell(val: str) -> str:
    colour = PARTY_COLOURS.get(val, "#888888")
    return f"background-color: {colour}33; color: {colour}; font-weight: bold"


gains_styled = (
    gains_df.style
    .map(_colour_party_cell, subset=["From", "To"])
    .format({"Majority (pp)": "{:.2f}"})
    .set_properties(**{"font-size": "13px"})
)
st.dataframe(gains_styled, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Built by **Debabrata Mishra** · Data Scientist / ML Engineer · "
    "[GitHub](https://github.com/dmishra27) · "
    "[LinkedIn](https://linkedin.com/in/debabrata-mishra) · "
    "[Portfolio](https://bit.ly/m/Debabrata_Mishra_Data_Science_Bio)"
)
