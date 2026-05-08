"""
Page 6 — Constituency Explorer.

Live search across all 73 constituency seats via the forecast API,
with a styled results table and a vote-share detail card.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")

PARTY_COLOURS = {
    "SNP": "#D4A017",
    "Reform": "#1ABC9C",
    "Labour": "#C0392B",
    "Conservative": "#2471A3",
    "LibDem": "#F39C12",
    "Green": "#27AE60",
}

st.set_page_config(page_title="Constituency Explorer", page_icon="🗺️", layout="wide")
st.title("🗺️ Constituency Explorer")
st.markdown(
    "Search any of the 73 Scottish Parliament constituency seats. "
    "Results update as you type."
)

# ── Search input ──────────────────────────────────────────────────────────────
search_term = st.text_input("Search constituency name...", placeholder="e.g. Glasgow, Orkney, Edinburgh")

# ── Fetch from API ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Fetching constituency data…")
def fetch_constituencies(search: str) -> dict | None:
    params = {}
    if search.strip():
        params["search"] = search.strip()
    try:
        resp = requests.get(
            f"{API_BASE}/seats/constituencies",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Could not connect to API at {API_BASE}. Is the API container running?"}
    except requests.exceptions.Timeout:
        return {"error": "API request timed out."}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"API returned {exc.response.status_code}: {exc.response.text}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


data = fetch_constituencies(search_term)

if data is None or "error" in data:
    st.error(data["error"] if data else "Unknown error fetching data.")
    st.stop()

constituencies = data.get("constituencies", [])
n_results = data.get("n_results", len(constituencies))

# ── Summary metrics ───────────────────────────────────────────────────────────
n_marginal = sum(1 for c in constituencies if c["is_marginal"])

col1, col2 = st.columns(2)
col1.metric("Results", n_results)
col2.metric("Marginal seats in results", n_marginal)

st.divider()

if not constituencies:
    st.info("No constituencies match your search.")
    st.stop()

# ── Build display dataframe ───────────────────────────────────────────────────
rows = [
    {
        "Constituency": c["constituency"],
        "Region": c["region"],
        "Predicted Winner": c["predicted_winner"],
        "Majority Margin (pp)": c["majority_margin_pp"],
        "Marginal?": "Yes" if c["is_marginal"] else "No",
        "Tactical Rec.": c["tactical_vote_recommendation"] or "—",
    }
    for c in constituencies
]
df = pd.DataFrame(rows)


def _style_row(row: pd.Series) -> list[str]:
    if row["Marginal?"] == "Yes":
        return ["background-color: #FEFBD0"] * len(row)
    return [""] * len(row)


def _colour_winner(val: str) -> str:
    colour = PARTY_COLOURS.get(val, "#888888")
    return f"background-color: {colour}33; color: {colour}; font-weight: bold"


styled = (
    df.style.apply(_style_row, axis=1)
    .map(_colour_winner, subset=["Predicted Winner"])
    .format({"Majority Margin (pp)": "{:.2f}"})
    .set_properties(**{"font-size": "13px"})
)

st.markdown("### Results")
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Detail card (single result) ───────────────────────────────────────────────
if n_results == 1:
    c = constituencies[0]
    st.divider()
    st.markdown(f"### 📌 {c['constituency']}")

    info_col, chart_col = st.columns([1, 2])
    with info_col:
        st.markdown(f"**Region:** {c['region']}")
        st.markdown(
            f"**Predicted winner:** "
            f"<span style='color:{PARTY_COLOURS.get(c['predicted_winner'], '#888')}; "
            f"font-weight:bold'>{c['predicted_winner']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Majority margin:** {c['majority_margin_pp']:.2f} pp")
        st.markdown(f"**Marginal seat:** {'Yes ⚠️' if c['is_marginal'] else 'No'}")
        if c["tactical_vote_recommendation"]:
            st.markdown(f"**Tactical rec.:** {c['tactical_vote_recommendation']}")

    with chart_col:
        vote_shares: dict[str, float] = c["vote_shares"]
        sorted_items = sorted(vote_shares.items(), key=lambda x: x[1], reverse=True)
        parties = [p for p, _ in sorted_items]
        shares = [v * 100 for _, v in sorted_items]
        colours = [PARTY_COLOURS.get(p, "#888888") for p in parties]

        fig = go.Figure(
            go.Bar(
                x=parties,
                y=shares,
                marker_color=colours,
                text=[f"{s:.1f}%" for s in shares],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Vote share: %{y:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Predicted vote shares",
            yaxis_title="Vote share (%)",
            yaxis_range=[0, max(shares) * 1.2],
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
