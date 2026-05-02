"""
Feature engineering pipeline for Scotland 2026 election forecast.

Produces a scikit-learn Pipeline-compatible transformer that:
  1. Engineers three domain-specific interaction features.
  2. Encodes categoricals (ordinal + one-hot).
  3. Scales continuous features.

Engineered features
-------------------
tactical_swing_index
    Captures how likely a voter is to switch parties strategically.
    Higher when concern levels diverge strongly AND party attachment is weak.

indep_economy_interaction
    Cross-term between independence stance and economic anxiety.
    Voters who back independence AND feel economic pain may punish the SNP government.

nhs_dissatisfaction
    Transforms the 1–5 NHS satisfaction scale into a 0–10 dissatisfaction score
    weighted by health salience, reflecting the Scottish Government's NHS record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

# ── column definitions ───────────────────────────────────────────────────────
TARGET_COL = "constituency_vote"

CONTINUOUS_COLS = [
    "age",
    "economic_concern",
    "health_concern",
    "immigration_concern",
    "tactical_swing_index",
    "indep_economy_interaction",
    "nhs_dissatisfaction",
]

ORDINAL_COLS = {
    "age_group": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "party_id_strength": [0, 1, 2, 3],
    "nhs_satisfaction": [1, 2, 3, 4, 5],
    "cost_of_living_impact": [1, 2, 3, 4, 5],
    "education": [
        "No qualifications",
        "Standard grades",
        "Highers",
        "HNC/HND",
        "Degree",
        "Postgraduate",
    ],
}

NOMINAL_COLS = ["region", "gender", "urban_rural", "top_priority", "independence_stance", "previous_vote"]

FEATURE_COLS = CONTINUOUS_COLS + list(ORDINAL_COLS.keys()) + NOMINAL_COLS

# ── engineering helpers ──────────────────────────────────────────────────────

_INDEP_MAP = {"Yes": 1.0, "No": -1.0, "Undecided": 0.0}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three engineered features to *df* in-place and return it.

    Does not modify the input if it is a copy; callers should pass df.copy()
    if they need to preserve the original.
    """
    # tactical_swing_index
    # |economic_concern – health_concern| normalised by max possible range (10),
    # amplified when party identity is weak (4 – strength → higher ↔ weaker ID).
    concern_divergence = (df["economic_concern"] - df["health_concern"]).abs() / 10.0
    id_weakness = (4 - df["party_id_strength"]) / 4.0
    df["tactical_swing_index"] = (concern_divergence * id_weakness).round(4)

    # indep_economy_interaction
    # Maps independence stance to {Yes: 1, Undecided: 0, No: -1} then multiplies
    # by scaled economic concern.  Positive when pro-independence AND economically anxious.
    indep_numeric = df["independence_stance"].map(_INDEP_MAP).fillna(0.0)
    df["indep_economy_interaction"] = (indep_numeric * df["economic_concern"] / 10.0).round(4)

    # nhs_dissatisfaction
    # Inverts the 1–5 NHS satisfaction scale → 0–8 dissatisfaction score,
    # then weights by health salience (0–10 scaled to 0–1).
    raw_dissatisfaction = (6 - df["nhs_satisfaction"]).clip(0, 5) * (8 / 5)
    health_weight = df["health_concern"] / 10.0
    df["nhs_dissatisfaction"] = (raw_dissatisfaction * health_weight).round(4)

    return df


# ── pipeline factory ─────────────────────────────────────────────────────────

def build_feature_pipeline() -> Pipeline:
    """
    Return a fitted-ready sklearn Pipeline that engineers and transforms features.

    The pipeline first adds engineered features (via a FunctionTransformer-style
    step) then applies ColumnTransformer encoding.
    """
    ordinal_transformers = [
        (
            f"ord_{col}",
            OrdinalEncoder(
                categories=[cats],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            [col],
        )
        for col, cats in ORDINAL_COLS.items()
    ]

    nominal_transformer = (
        "ohe",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        NOMINAL_COLS,
    )

    continuous_transformer = (
        "scaler",
        StandardScaler(),
        CONTINUOUS_COLS,
    )

    preprocessor = ColumnTransformer(
        transformers=[continuous_transformer, *ordinal_transformers, nominal_transformer],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def prepare_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer features on a copy of df and return (X, y)."""
    df_eng = engineer_features(df.copy())
    X = df_eng[FEATURE_COLS]
    y = df_eng[TARGET_COL]
    return X, y
