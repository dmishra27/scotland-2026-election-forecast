"""
Page 1 — Voter Simulator.

Build a voter profile with sidebar inputs, call FastAPI /predict,
fall back to a rule-based prior-weighted model when the API is offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))

API_URL = "http://localhost:8000"
PARTIES = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"]
PARTY_COLOURS = {
    "SNP": "#D4A017", "Reform": "#1ABC9C", "Labour": "#C0392B",
    "Conservative": "#2471A3", "LibDem": "#F39C12", "Green": "#27AE60",
}
PRIOR_PROBS = {
    "SNP": 0.356, "Reform": 0.179, "Labour": 0.163,
    "Conservative": 0.103, "LibDem": 0.080, "Green": 0.060,
}
REGIONS = [
    "Central Scotland", "Glasgow", "Highlands and Islands", "Lothian",
    "Mid Scotland and Fife", "North East Scotland", "South Scotland", "West Scotland",
]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
EDUCATION = ["No qualifications", "Standard grades", "Highers", "HNC/HND", "Degree", "Postgraduate"]
PREV_PARTIES = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green", "Did not vote"]


# ── rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_predict(voter: dict) -> dict:
    probs = dict(PRIOR_PROBS)
    stance = voter["independence_stance"]
    if stance == "Strong Yes":
        probs["SNP"] = min(probs["SNP"] * 1.5, 0.85)
        probs["Green"] = min(probs["Green"] * 1.2, 0.85)
    elif stance == "Lean Yes":
        probs["SNP"] = min(probs["SNP"] * 1.25, 0.85)
    elif stance == "Lean No":
        probs["SNP"] *= 0.75
        probs["Conservative"] = min(probs["Conservative"] * 1.3, 0.85)
    elif stance == "Strong No":
        probs["SNP"] *= 0.55
        probs["Conservative"] = min(probs["Conservative"] * 1.6, 0.85)
    boost = {"Economy": "Reform", "Health": "Labour", "Immigration": "Reform", "Independence": "SNP", "Housing": "Green"}
    b = boost.get(voter["top_priority"])
    if b:
        probs[b] *= 1.20
    total = sum(probs.values())
    probs = {p: round(v / total, 4) for p, v in probs.items()}
    return {"predicted_party": max(probs, key=lambda p: probs[p]), "probabilities": probs}


def _call_api(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Voter Simulator", page_icon="🗳️", layout="wide")
st.title("🗳️ Voter Simulator")
st.markdown("Configure a voter profile and see who they are predicted to vote for.")

with st.sidebar:
    st.header("Voter Profile")
    region = st.selectbox("Region", REGIONS, index=1)
    age = st.slider("Age", 18, 85, 38)
    age_group = AGE_GROUPS[min(max((age - 18) // 10, 0), 5)]
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    education = st.selectbox("Education", EDUCATION, index=4)
    urban_rural = st.selectbox("Urban/Rural", ["Urban", "Suburban", "Rural"])
    indep_stance = st.selectbox("Independence stance", ["Strong Yes", "Lean Yes", "Undecided", "Lean No", "Strong No"])
    top_priority = st.selectbox("Top priority", ["Economy", "Health", "Immigration", "Independence", "Housing"])
    prev_vote = st.selectbox("2021 vote", PREV_PARTIES)
    party_id = st.slider("Party attachment (0=none, 3=strong)", 0, 3, 1)
    econ_concern = st.slider("Economic concern (0–10)", 0.0, 10.0, 6.5, 0.1)
    health_concern = st.slider("Health/NHS concern (0–10)", 0.0, 10.0, 6.8, 0.1)
    immig_concern = st.slider("Immigration concern (0–10)", 0.0, 10.0, 5.2, 0.1)
    nhs_sat = st.slider("NHS satisfaction (1=very poor, 5=excellent)", 1, 5, 3)
    col_impact = st.slider("Cost-of-living impact (1=none, 5=severe)", 1, 5, 3)
    run_btn = st.button("Predict vote", type="primary")

if run_btn:
    payload = {
        "region": region,
        "age": age,
        "age_group": age_group,
        "gender": gender,
        "education": education,
        "urban_rural": urban_rural,
        "economic_concern": econ_concern,
        "health_concern": health_concern,
        "immigration_concern": immig_concern,
        "top_priority": top_priority,
        "independence_stance": indep_stance,
        "previous_vote": prev_vote,
        "party_id_strength": party_id,
        "nhs_satisfaction": nhs_sat,
        "cost_of_living_impact": col_impact,
    }

    with st.spinner("Predicting..."):
        result = _call_api(payload)

    api_used = result is not None
    if not api_used:
        result = _rule_based_predict(payload)
        st.info("API offline — using rule-based fallback prediction.", icon="⚠️")

    predicted = result["predicted_party"]
    probs = result["probabilities"]

    # ── result header ─────────────────────────────────────────────────────
    col_res, col_eng = st.columns([2, 1])
    with col_res:
        colour = PARTY_COLOURS.get(predicted, "#888")
        st.markdown(
            f"<div style='background:{colour};padding:20px;border-radius:8px;"
            f"text-align:center;'>"
            f"<h2 style='color:white;margin:0'>Predicted vote: {predicted}</h2></div>",
            unsafe_allow_html=True,
        )

    with col_eng:
        if api_used:
            st.metric("Tactical Swing Index", f"{result.get('tactical_swing_index', 0):.3f}")
            st.metric("Indep × Economy", f"{result.get('indep_economy_interaction', 0):.3f}")
            st.metric("NHS Dissatisfaction", f"{result.get('nhs_dissatisfaction', 0):.3f}")

    st.markdown("#### Vote probability breakdown")
    sorted_parties = sorted(probs, key=lambda p: -probs[p])
    fig = go.Figure(go.Bar(
        x=[probs[p] * 100 for p in sorted_parties],
        y=sorted_parties,
        orientation="h",
        marker_color=[PARTY_COLOURS[p] for p in sorted_parties],
        text=[f"{probs[p]*100:.1f}%" for p in sorted_parties],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Probability (%)",
        xaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=0, r=60, t=10, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Source: {'FastAPI model endpoint' if api_used else 'Rule-based prior fallback'}")

else:
    st.info("Configure a voter profile in the sidebar and click **Predict vote**.")
