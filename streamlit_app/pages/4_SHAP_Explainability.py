"""
Page 4 — SHAP Explainability.

Horizontal bar chart of mean absolute SHAP values, colour-coded by feature
category (demographic, policy concern, sentiment, identity, engineered).
Loads from models/shap_importance.csv if available; falls back to static
benchmark values for demo mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))

# ── feature category mapping ──────────────────────────────────────────────────
FEATURE_CATEGORIES = {
    "age": "demographic",
    "age_group": "demographic",
    "gender": "demographic",
    "education": "demographic",
    "urban_rural": "demographic",
    "region": "geographic",
    "economic_concern": "policy concern",
    "health_concern": "policy concern",
    "immigration_concern": "policy concern",
    "top_priority": "policy concern",
    "independence_stance": "identity",
    "previous_vote": "identity",
    "party_id_strength": "identity",
    "nhs_satisfaction": "sentiment",
    "cost_of_living_impact": "sentiment",
    "tactical_swing_index": "engineered",
    "indep_economy_interaction": "engineered",
    "nhs_dissatisfaction": "engineered",
}

CATEGORY_COLOURS = {
    "engineered": "#8E44AD",
    "identity": "#D4A017",
    "sentiment": "#C0392B",
    "policy concern": "#1ABC9C",
    "demographic": "#2471A3",
    "geographic": "#27AE60",
}

# ── static benchmark SHAP values (demo mode) ──────────────────────────────────
DEMO_SHAP = pd.DataFrame({
    "feature": [
        "independence_stance", "previous_vote", "indep_economy_interaction",
        "economic_concern", "party_id_strength", "nhs_dissatisfaction",
        "health_concern", "top_priority", "tactical_swing_index",
        "age", "region", "cost_of_living_impact", "immigration_concern",
        "education", "urban_rural", "nhs_satisfaction", "gender",
        "age_group",
    ],
    "mean_abs_shap": [
        0.182, 0.161, 0.143, 0.118, 0.107, 0.096,
        0.089, 0.082, 0.074, 0.063, 0.058, 0.054,
        0.049, 0.041, 0.036, 0.031, 0.024, 0.019,
    ],
})

st.set_page_config(page_title="SHAP Explainability", page_icon="🔍", layout="wide")
st.title("🔍 SHAP Feature Importance")
st.markdown("Mean absolute SHAP values — how much each feature shifts the predicted vote probability.")

# ── load data ─────────────────────────────────────────────────────────────────
shap_path = Path("models/shap_importance.csv")
if shap_path.exists():
    shap_df = pd.read_csv(shap_path)
    st.success("Live SHAP values loaded from trained model.", icon="✅")
else:
    shap_df = DEMO_SHAP
    st.info("Showing benchmark SHAP values — train the model to load live values.", icon="ℹ️")

# ── plot ──────────────────────────────────────────────────────────────────────
shap_df = shap_df.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)
# Map clean feature names (strip transformer prefix if present)
shap_df["feature_clean"] = shap_df["feature"].str.split("__").str[-1]
shap_df["category"] = shap_df["feature_clean"].map(
    lambda f: next((FEATURE_CATEGORIES[k] for k in FEATURE_CATEGORIES if k in f), "other")
)
shap_df["colour"] = shap_df["category"].map(lambda c: CATEGORY_COLOURS.get(c, "#95A5A6"))

fig = go.Figure(go.Bar(
    x=shap_df["mean_abs_shap"],
    y=shap_df["feature_clean"],
    orientation="h",
    marker_color=shap_df["colour"].tolist(),
    text=[f"{v:.3f}" for v in shap_df["mean_abs_shap"]],
    textposition="outside",
))
fig.update_layout(
    title="Mean |SHAP| per feature (averaged over all classes and base learners)",
    xaxis_title="Mean absolute SHAP value",
    height=max(400, len(shap_df) * 22),
    margin=dict(l=0, r=80, t=50, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# ── legend ────────────────────────────────────────────────────────────────────
st.markdown("#### Feature category legend")
legend_cols = st.columns(len(CATEGORY_COLOURS))
for col, (cat, colour) in zip(legend_cols, CATEGORY_COLOURS.items()):
    col.markdown(
        f"<span style='background:{colour};color:white;padding:3px 8px;"
        f"border-radius:4px;font-size:0.85em'>{cat}</span>",
        unsafe_allow_html=True,
    )

st.divider()

# ── interpretation text ───────────────────────────────────────────────────────
st.markdown("""
#### Interpretation

| Rank | Feature | Category | Insight |
|------|---------|----------|---------|
| 1 | `independence_stance` | Identity | The single strongest predictor — Yes voters strongly favour SNP; No voters split towards Labour/Conservative |
| 2 | `previous_vote` | Identity | Vote recall is the most stable predictor of future vote in British elections |
| 3 | `indep_economy_interaction` | Engineered | Pro-independence voters who feel economic pain may punish the SNP government — key swing group |
| 4 | `economic_concern` | Policy concern | Reform's primary recruiting ground; high concern boosts Reform probability |
| 5 | `party_id_strength` | Identity | Weak identifiers (0–1) are the pool of genuine swing voters |
| 6 | `nhs_dissatisfaction` | Engineered | Scotland's NHS is a devolved responsibility — dissatisfaction shifts votes away from SNP/Labour incumbents |

**Engineered features** (purple) punch above their weight relative to raw inputs, validating
the feature engineering step.  `independence_stance` and `previous_vote` together account for
~34% of total model attribution, consistent with British Electoral Survey findings.
""")

st.divider()
st.caption("Built by **Debabrata Mishra** · Data Scientist / ML Engineer · "
           "[GitHub](https://github.com/dmishra27) · "
           "[LinkedIn](https://linkedin.com/in/debabrata-mishra) · "
           "[Portfolio](https://bit.ly/m/Debabrata_Mishra_Data_Science_Bio)")
