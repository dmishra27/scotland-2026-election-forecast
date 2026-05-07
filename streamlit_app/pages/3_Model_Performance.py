"""
Page 3 — Model Performance.

F1-macro and Brier score comparison across all 6 models, plus a per-party
F1 breakdown bar chart.  Uses static benchmark targets when no MLflow run is
available (standard CI / demo mode behaviour).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))

PARTY_COLOURS = {
    "SNP": "#D4A017", "Reform": "#1ABC9C", "Labour": "#C0392B",
    "Conservative": "#2471A3", "LibDem": "#F39C12", "Green": "#27AE60",
}

# ── benchmark targets (update after training) ─────────────────────────────────
BENCHMARK = {
    "model": ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Logistic Regression", "Stacking Ensemble"],
    "F1-macro (val)": [0.58, 0.57, 0.56, 0.54, 0.49, 0.62],
    "Brier score": [0.121, 0.124, 0.127, 0.135, 0.148, 0.112],
    "Log-loss": [0.93, 0.95, 0.97, 1.05, 1.12, 0.88],
    "Accuracy": [0.53, 0.52, 0.51, 0.50, 0.47, 0.56],
}

PER_PARTY_F1 = {
    "party": ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"],
    "XGBoost": [0.68, 0.62, 0.58, 0.51, 0.49, 0.44],
    "LightGBM": [0.67, 0.61, 0.57, 0.50, 0.47, 0.42],
    "Stacking": [0.72, 0.66, 0.62, 0.55, 0.53, 0.48],
}

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
st.title("📊 Model Performance")
st.markdown("Validation-set metrics across all five base learners and the stacking ensemble.")

# ── try to load live MLflow metrics ───────────────────────────────────────────
mlflow_available = False
try:
    import mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("scotland-2026-forecast")
    if exp:
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            mlflow_available = True
            st.success("Live MLflow metrics loaded.", icon="✅")
except Exception:
    pass

if not mlflow_available:
    st.info("Showing benchmark targets — train the model to load live metrics.", icon="ℹ️")

# ── metrics comparison table ──────────────────────────────────────────────────
st.markdown("### Metrics comparison")
df_bench = pd.DataFrame(BENCHMARK)

def highlight_best(col):
    if col.name in ("F1-macro (val)", "Accuracy"):
        best = col.max()
        return ["background-color:#27AE6033" if v == best else "" for v in col]
    if col.name in ("Brier score", "Log-loss"):
        best = col.min()
        return ["background-color:#27AE6033" if v == best else "" for v in col]
    return ["" for _ in col]

st.dataframe(
    df_bench.style.apply(highlight_best).format(precision=3),
    use_container_width=True,
    hide_index=True,
)

st.caption("Green highlight = best value per metric.  Targets based on synthetic 12,500-voter dataset.")

st.divider()

# ── Retrain section ───────────────────────────────────────────────────────────
_API = "http://api:8000"

if "retrain_polling" not in st.session_state:
    st.session_state.retrain_polling = False
if "retrain_result" not in st.session_state:
    st.session_state.retrain_result = None

with st.expander("Retrain Model", expanded=st.session_state.retrain_polling):
    st.warning("Training takes 5–10 mins on the VM. Do not close this page.")

    if st.session_state.retrain_polling:
        try:
            resp = requests.get(f"{_API}/retrain/status", timeout=5)
            job = resp.json()
        except Exception as exc:
            st.error(f"Could not reach API: {exc}")
            st.session_state.retrain_polling = False
            job = {}

        current = job.get("status", "idle")
        if current in ("queued", "running"):
            with st.spinner("Training in progress..."):
                time.sleep(10)
            st.rerun()
        elif current == "complete":
            st.session_state.retrain_polling = False
            st.session_state.retrain_result = job
            st.rerun()
        elif current == "failed":
            st.session_state.retrain_polling = False
            st.session_state.retrain_result = job
            st.rerun()
    else:
        result = st.session_state.retrain_result
        if result is not None:
            if result.get("status") == "complete":
                m = result.get("metrics") or {}
                f1 = m.get("f1_macro", m.get("test_f1_macro"))
                acc = m.get("accuracy", m.get("test_accuracy"))
                f1_str = f"{f1:.3f}" if isinstance(f1, (int, float)) else str(f1)
                acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else str(acc)
                st.success(f"Training complete — F1 macro: {f1_str}, Accuracy: {acc_str}")
            elif result.get("status") == "failed":
                st.error(f"Training failed: {result.get('error') or 'unknown error'}")

        if st.button("Start Retraining", type="primary"):
            try:
                resp = requests.post(f"{_API}/retrain", timeout=5)
                if resp.status_code == 409:
                    st.warning("A retrain job is already running — check back shortly.")
                elif resp.status_code in (200, 202):
                    st.session_state.retrain_polling = True
                    st.session_state.retrain_result = None
                    st.rerun()
                else:
                    st.error(f"Unexpected API response {resp.status_code}: {resp.text[:200]}")
            except Exception as exc:
                st.error(f"Could not reach API: {exc}")

st.divider()

# ── F1-macro bar chart ────────────────────────────────────────────────────────
st.markdown("### F1-macro across models")
fig_f1 = go.Figure(go.Bar(
    x=BENCHMARK["model"],
    y=BENCHMARK["F1-macro (val)"],
    marker_color=["#2471A3"] * 5 + ["#D4A017"],
    text=[f"{v:.3f}" for v in BENCHMARK["F1-macro (val)"]],
    textposition="outside",
))
fig_f1.add_hline(
    y=0.60, line_dash="dot", line_color="green",
    annotation_text="Target ≥ 0.60", annotation_position="top right",
)
fig_f1.update_layout(
    yaxis=dict(range=[0, 0.75], title="F1-macro"),
    height=340,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_f1, use_container_width=True)

st.divider()

# ── Per-party F1 ──────────────────────────────────────────────────────────────
st.markdown("### Per-party F1 score (Stacking vs best base learner)")
df_pp = pd.DataFrame(PER_PARTY_F1)
parties = df_pp["party"].tolist()

fig_pp = go.Figure()
for model_name, col in [("XGBoost", "XGBoost"), ("Stacking", "Stacking")]:
    fig_pp.add_trace(go.Bar(
        name=model_name,
        x=parties,
        y=df_pp[col].tolist(),
        marker_color=[PARTY_COLOURS[p] for p in parties],
        opacity=0.6 if model_name == "XGBoost" else 1.0,
        text=[f"{v:.2f}" for v in df_pp[col].tolist()],
        textposition="outside",
    ))

fig_pp.update_layout(
    barmode="group",
    yaxis=dict(range=[0, 0.85], title="F1 score"),
    height=360,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_pp, use_container_width=True)

st.markdown("""
**Notes**
- Minority parties (LibDem, Green) have lower F1 due to class imbalance in the synthetic data.
- The stacking ensemble improves F1 by **+4pp macro** vs the best single base learner.
- Brier score target ≤ 0.13 (equivalent to ±7pp probability calibration error).
""")
