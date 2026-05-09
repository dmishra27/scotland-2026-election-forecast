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

# Certified Electoral Management Board results — 8 May 2026, all 129 seats declared.
ACTUAL_RESULTS: dict[str, dict] = {
    "Orkney Islands": {
        "actual_winner": "LibDem",
        "actual_majority": 5560,
        "actual_majority_pp": 54.07,
        "turnout_pct": 57.64,
        "actual_shares": {
            "LibDem": 70.22, "SNP": 16.15, "Reform": 8.21,
            "Conservative": 3.48, "Labour": 1.94,
        },
    },
    "Inverclyde": {
        "actual_winner": "SNP",
        "actual_majority": 5317,
        "actual_majority_pp": 16.60,
        "turnout_pct": 51.55,
        "actual_shares": {
            "SNP": 44.32, "Labour": 27.72, "Reform": 17.64,
            "LibDem": 6.10, "Conservative": 4.22,
        },
    },
    "Inverness and Nairn": {
        "actual_winner": "SNP",
        "actual_majority": 427,
        "actual_majority_pp": 1.16,
        "turnout_pct": 54.54,
        "actual_shares": {
            "SNP": 30.39, "LibDem": 29.22, "Independent": 21.34,
            "Reform": 10.32, "Labour": 4.69, "Conservative": 3.74,
        },
    },
    "Strathkelvin and Bearsden": {
        "actual_winner": "LibDem",
        "actual_majority": 2572,
        "actual_majority_pp": 6.47,
        "turnout_pct": 62.72,
        "actual_shares": {
            "LibDem": 39.46, "SNP": 33.00, "Labour": 11.76,
            "Reform": 10.44, "Conservative": 5.33,
        },
    },
    "Edinburgh Central": {
        "actual_winner": "Green",
        "actual_majority": 4582,
        "actual_majority_pp": 13.00,
        "turnout_pct": 54.54,
        "actual_shares": {
            "Green": 35.98, "Labour": 22.98, "SNP": 21.86,
            "Conservative": 6.42, "LibDem": 6.15, "Reform": 5.32,
        },
    },
    "Na h-Eileanan an Iar": {
        "actual_winner": "Labour",
        "actual_majority": 154,
        "actual_majority_pp": 1.25,
        "turnout_pct": 56.70,
        "actual_shares": {
            "Labour": 37.72, "SNP": 36.48, "Reform": 13.14,
            "LibDem": 6.57, "Conservative": 4.80,
        },
    },
    "Glasgow Southside": {
        "actual_winner": "Green",
        "actual_majority": 3101,
        "actual_majority_pp": 8.05,
        "turnout_pct": 59.09,
        "actual_shares": {
            "Green": 36.49, "SNP": 28.43, "Labour": 18.96,
            "Reform": 7.84, "Conservative": 3.59, "LibDem": 2.97,
        },
    },
    "Fife North East": {
        "actual_winner": "LibDem",
        "actual_majority": 13474,
        "actual_majority_pp": 40.22,
        "turnout_pct": 58.42,
        "actual_shares": {
            "LibDem": 63.72, "SNP": 23.51, "Reform": 7.53,
            "Conservative": 2.79, "Labour": 2.44,
        },
    },
    "Edinburgh Southern": {
        "actual_winner": "Labour",
        "actual_majority": 4963,
        "actual_majority_pp": 12.55,
        "turnout_pct": 60.98,
        "actual_shares": {
            "Labour": 42.88, "SNP": 30.33, "Conservative": 8.65,
            "LibDem": 8.43, "Reform": 8.38,
        },
    },
    "Aberdeenshire West": {
        "actual_winner": "Conservative",
        "actual_majority": 5784,
        "actual_majority_pp": 15.62,
        "turnout_pct": 60.47,
        "actual_shares": {
            "Conservative": 42.92, "SNP": 27.30, "Reform": 14.76,
            "LibDem": 10.79, "Labour": 4.24,
        },
    },
    "Dumbarton": {
        "actual_winner": "Labour",
        "actual_majority": 1786,
        "actual_majority_pp": 5.58,
        "turnout_pct": 56.84,
        "actual_shares": {
            "Labour": 39.81, "SNP": 34.23, "Reform": 15.74,
            "Conservative": 4.27, "LibDem": 3.73,
        },
    },
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
def _top_dev_str(api_row: dict, actual: dict | None) -> str:
    if not actual:
        return "—"
    pred_pct = {p: v * 100 for p, v in api_row["vote_shares"].items()}
    all_parties = set(pred_pct) | set(actual["actual_shares"])
    max_dev = max(
        (pred_pct.get(p, 0.0) - actual["actual_shares"].get(p, 0.0) for p in all_parties),
        key=abs,
    )
    return f"{max_dev:+.1f}"


rows = []
for c in constituencies:
    actual = ACTUAL_RESULTS.get(c["constituency"])
    rows.append({
        "Constituency": c["constituency"],
        "Region": c["region"],
        "Predicted Winner": c["predicted_winner"],
        "Majority Margin (pp)": c["majority_margin_pp"],
        "Marginal?": "Yes" if c["is_marginal"] else "No",
        "Tactical Rec.": c["tactical_vote_recommendation"] or "—",
        "Actual Winner": actual["actual_winner"] if actual else "—",
        "Actual Majority (pp)": f"{actual['actual_majority_pp']:.2f}" if actual else "—",
        "Turnout %": f"{actual['turnout_pct']:.1f}" if actual else "—",
        "Model Correct?": (
            "✅" if actual and actual["actual_winner"] == c["predicted_winner"]
            else ("❌" if actual else "—")
        ),
        "Top Deviation (pp)": _top_dev_str(c, actual),
    })
df = pd.DataFrame(rows)


def _style_row(row: pd.Series) -> list[str]:
    if row["Marginal?"] == "Yes":
        return ["background-color: #FEFBD0"] * len(row)
    return [""] * len(row)


def _colour_winner(val: str) -> str:
    colour = PARTY_COLOURS.get(val, "#888888")
    return f"background-color: {colour}33; color: {colour}; font-weight: bold"


def _colour_correct(val: str) -> str:
    if val == "✅":
        return "background-color:#D5F5E3; color:#1E8449; font-weight:bold"
    if val == "❌":
        return "background-color:#FADBD8; color:#C0392B; font-weight:bold"
    return ""


def _colour_top_dev(val: str) -> str:
    try:
        if abs(float(val)) > 10:
            return "background-color:#FADBD8; color:#C0392B; font-weight:bold"
    except (ValueError, TypeError):
        pass
    return ""


def _render_detail_card(c: dict) -> None:
    """Render the full detail card for a single constituency API result dict."""
    st.divider()
    st.markdown(f"### 📌 {c['constituency']}")

    info_col, chart_col = st.columns([1, 2])
    with info_col:
        st.markdown(f"**Region:** {c['region']}")
        st.markdown(
            f"**Predicted winner:** "
            f"<span style='color:{PARTY_COLOURS.get(c[\"predicted_winner\"], \"#888\")}; "
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

    # ── Actual 2026 results ───────────────────────────────────────────────────
    actual = ACTUAL_RESULTS.get(c["constituency"])
    if actual:
        st.divider()
        predicted_winner = c["predicted_winner"]
        actual_winner = actual["actual_winner"]

        if predicted_winner != actual_winner:
            st.warning(
                f"⚠️ Model predicted **{predicted_winner}** — "
                f"actual winner was **{actual_winner}**"
            )

        st.subheader("🗳️ Actual Result (8 May 2026)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Actual Winner", actual_winner)
        m2.metric("Actual Majority", f"{actual['actual_majority_pp']:.2f} pp")
        m3.metric("Turnout", f"{actual['turnout_pct']:.1f}%")

        act_items = sorted(actual["actual_shares"].items(), key=lambda x: x[1], reverse=True)
        act_parties = [p for p, _ in act_items]
        act_shares_vals = [v for _, v in act_items]
        act_colours = [PARTY_COLOURS.get(p, "#888888") for p in act_parties]

        fig_act = go.Figure(
            go.Bar(
                x=act_parties,
                y=act_shares_vals,
                marker_color=act_colours,
                text=[f"{v:.1f}%" for v in act_shares_vals],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Vote share: %{y:.1f}%<extra></extra>",
            )
        )
        fig_act.update_layout(
            title="Actual vote shares (%)",
            yaxis_title="Vote share (%)",
            yaxis_range=[0, max(act_shares_vals) * 1.2],
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_act, use_container_width=True)

        st.subheader("📊 Model vs Actual Deviation")
        pred_shares_pct: dict[str, float] = {p: v * 100 for p, v in c["vote_shares"].items()}
        all_parties_dev = sorted(
            set(pred_shares_pct) | set(actual["actual_shares"]),
            key=lambda p: abs(
                pred_shares_pct.get(p, 0.0) - actual["actual_shares"].get(p, 0.0)
            ),
            reverse=True,
        )
        dev_rows = [
            {
                "Party": p,
                "Predicted %": pred_shares_pct.get(p, 0.0),
                "Actual %": actual["actual_shares"].get(p, 0.0),
                "Deviation (pp)": (
                    pred_shares_pct.get(p, 0.0) - actual["actual_shares"].get(p, 0.0)
                ),
            }
            for p in all_parties_dev
        ]
        dev_df = pd.DataFrame(dev_rows)

        def _colour_deviation(val: float) -> str:
            if abs(val) > 5:
                return "background-color:#FADBD8; color:#C0392B"
            if abs(val) >= 2:
                return "background-color:#FDEBD0; color:#E67E22"
            return "background-color:#D5F5E3; color:#1E8449"

        dev_styled = (
            dev_df.style
            .map(_colour_deviation, subset=["Deviation (pp)"])
            .format({"Predicted %": "{:.1f}", "Actual %": "{:.1f}", "Deviation (pp)": "{:+.2f}"})
            .set_properties(**{"font-size": "13px"})
        )
        st.dataframe(dev_styled, use_container_width=True, hide_index=True)
        st.caption("Positive = model overestimated; Negative = model underestimated")
    else:
        st.divider()
        st.info(
            "Actual 2026 results not yet loaded for this constituency. "
            "Full results: [ELECTION_RESULTS.md](ELECTION_RESULTS.md)"
        )


styled = (
    df.style.apply(_style_row, axis=1)
    .map(_colour_winner, subset=["Predicted Winner"])
    .map(_colour_winner, subset=["Actual Winner"])
    .map(_colour_correct, subset=["Model Correct?"])
    .map(_colour_top_dev, subset=["Top Deviation (pp)"])
    .format({"Majority Margin (pp)": "{:.2f}"})
    .set_properties(**{"font-size": "13px"})
)

st.markdown("### Results")
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Detail card (single result only) ─────────────────────────────────────────
if n_results == 1:
    _render_detail_card(constituencies[0])

st.divider()
st.caption("Built by **Debabrata Mishra** · Data Scientist / ML Engineer · "
           "[GitHub](https://github.com/dmishra27) · "
           "[LinkedIn](https://linkedin.com/in/debabrata-mishra) · "
           "[Portfolio](https://bit.ly/m/Debabrata_Mishra_Data_Science_Bio)")
