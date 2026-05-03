"""
Synthetic voter data generator for Scotland 2026 Scottish Parliament election.

Each voter's vote probability vector is derived from YouGov MRP regional priors
(April 2026) then adjusted multiplicatively by their personal attributes:
independence stance, top policy priority, previous vote loyalty, and age cohort.

This produces realistic feature→label correlations so the downstream ML model
has genuine signal to learn from (target F1-macro ~0.55–0.65).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── party ordering ────────────────────────────────────────────────────────────
PARTIES: list[str] = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"]
_PI: dict[str, int] = {p: i for i, p in enumerate(PARTIES)}

# ── YouGov MRP priors (April 2026, n=3,925) ──────────────────────────────────
CONSTITUENCY_PRIORS: dict[str, float] = {
    "SNP": 0.356, "Reform": 0.179, "Labour": 0.163,
    "Conservative": 0.103, "LibDem": 0.080, "Green": 0.060,
}
REGIONAL_LIST_PRIORS: dict[str, float] = {
    "SNP": 0.291, "Reform": 0.180, "Labour": 0.153,
    "Conservative": 0.111, "LibDem": 0.090, "Green": 0.085,
}

# ── independence polling split into 4 ordered stances + Undecided ─────────────
# Preserves overall Yes 46.6 / No 44.6 / Undecided 8.9 from MRP
INDEPENDENCE_PRIORS: dict[str, float] = {
    "Strong Yes": 0.230,
    "Lean Yes":   0.236,
    "Undecided":  0.089,
    "Lean No":    0.223,
    "Strong No":  0.222,
}

# ── regions & electorate weights ─────────────────────────────────────────────
REGIONS: list[str] = [
    "Central Scotland", "Glasgow", "Highlands and Islands", "Lothian",
    "Mid Scotland and Fife", "North East Scotland", "South Scotland", "West Scotland",
]
REGION_WEIGHTS = np.array([0.135, 0.160, 0.085, 0.155, 0.135, 0.120, 0.115, 0.095])

REGION_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "Central Scotland":       {"SNP": 1.05, "Labour": 1.10, "Reform": 0.95},
    "Glasgow":                {"SNP": 1.00, "Labour": 1.15, "Green": 1.20, "Reform": 0.90},
    "Highlands and Islands":  {"SNP": 1.10, "LibDem": 1.30, "Reform": 0.85},
    "Lothian":                {"SNP": 0.95, "Green": 1.25, "Labour": 1.05, "LibDem": 1.15},
    "Mid Scotland and Fife":  {"SNP": 1.05, "Labour": 1.05, "Reform": 1.00},
    "North East Scotland":    {"SNP": 1.05, "Conservative": 1.10, "Reform": 1.10},
    "South Scotland":         {"Conservative": 1.20, "Reform": 1.15, "SNP": 0.90},
    "West Scotland":          {"SNP": 1.00, "Labour": 1.10, "Reform": 1.05},
}

# ── feature → vote probability adjustments ───────────────────────────────────
INDEPENDENCE_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "Strong Yes": {"SNP": 2.5, "Green": 1.2, "Labour": 0.5, "Conservative": 0.3, "Reform": 0.5},
    "Lean Yes":   {"SNP": 1.6, "Green": 1.1, "Conservative": 0.6},
    "Undecided":  {},
    "Lean No":    {"SNP": 0.6, "Conservative": 1.5, "Labour": 1.2},
    "Strong No":  {"SNP": 0.4, "Conservative": 1.8, "Labour": 1.3, "Reform": 1.3},
}

PRIORITY_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "Immigration":  {"Reform": 2.0, "Conservative": 1.3, "SNP": 0.7},
    "Independence": {"SNP": 2.0, "Green": 1.4, "Conservative": 0.5},
    "Health":       {"Labour": 1.5, "SNP": 1.1},
    "Economy":      {"Reform": 1.6, "Labour": 1.3},
    "Housing":      {"Green": 1.4, "Labour": 1.2},
}

_AGE_OLD_ADJ   = {"Conservative": 1.4, "Reform": 1.3, "SNP": 0.9, "Green": 0.5}
_AGE_YOUNG_ADJ = {"Green": 1.5, "SNP": 1.2, "Conservative": 0.4, "Reform": 0.6}
_AMS_LIST_ADJ  = {"Green": 1.2, "LibDem": 1.2}   # split-ticket AMS tendency

PREV_PARTIES = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green", "Did not vote"]
PREV_VOTE_PRIORS = np.array([0.400, 0.000, 0.220, 0.227, 0.053, 0.068, 0.032])


# ── helpers ───────────────────────────────────────────────────────────────────

def _regional_base(base_priors: dict[str, float], region: str) -> np.ndarray:
    """Normalised probability vector after regional multiplier adjustments."""
    adj = REGION_ADJUSTMENTS.get(region, {})
    raw = {p: base_priors[p] * adj.get(p, 1.0) for p in PARTIES}
    total = sum(raw.values())
    return np.array([raw[p] / total for p in PARTIES])


def _compute_vote_probs(
    regions: np.ndarray,
    independence_stance: np.ndarray,
    top_priority: np.ndarray,
    previous_vote: np.ndarray,
    ages: np.ndarray,
    base_priors: dict[str, float],
    is_list_vote: bool = False,
) -> np.ndarray:
    """
    Build a (n_voters, n_parties) matrix of normalised vote probabilities.

    Algorithm
    ---------
    1. Initialise each voter from their region's adjusted MRP prior.
    2. Apply multiplicative adjustments per feature (independence stance,
       top priority, previous-vote loyalty, age cohort).
    3. For list votes additionally boost Green and LibDem to reflect the
       common AMS split-ticket pattern.
    4. Row-normalise to valid probability vectors.
    """
    n = len(regions)

    # Step 1 — regional base
    probs = np.zeros((n, len(PARTIES)))
    for region in REGIONS:
        mask = regions == region
        if mask.sum() == 0:
            continue
        probs[mask] = _regional_base(base_priors, region)

    adj = np.ones((n, len(PARTIES)))

    # Step 2a — independence stance
    for stance, mults in INDEPENDENCE_ADJUSTMENTS.items():
        mask = independence_stance == stance
        if not mask.any():
            continue
        for party, mult in mults.items():
            adj[mask, _PI[party]] *= mult

    # Step 2b — top policy priority
    for priority, mults in PRIORITY_ADJUSTMENTS.items():
        mask = top_priority == priority
        if not mask.any():
            continue
        for party, mult in mults.items():
            adj[mask, _PI[party]] *= mult

    # Step 2c — previous-vote loyalty (×2.5 own party)
    for party in PARTIES:
        mask = previous_vote == party
        if not mask.any():
            continue
        adj[mask, _PI[party]] *= 2.5

    # Step 2d — age cohort adjustments
    mask_old   = ages >= 65
    mask_young = (ages >= 18) & (ages <= 24)
    for party, mult in _AGE_OLD_ADJ.items():
        adj[mask_old, _PI[party]] *= mult
    for party, mult in _AGE_YOUNG_ADJ.items():
        adj[mask_young, _PI[party]] *= mult

    # Step 3 — AMS list-vote bonus for smaller parties
    if is_list_vote:
        for party, mult in _AMS_LIST_ADJ.items():
            adj[:, _PI[party]] *= mult

    # Step 4 — apply, guard zeros, normalise
    adjusted = probs * adj
    row_sums  = adjusted.sum(axis=1, keepdims=True)
    row_sums  = np.where(row_sums == 0, 1.0, row_sums)
    return adjusted / row_sums


# ── main generator ────────────────────────────────────────────────────────────

def generate_voters(
    n_voters: int = 12_500,
    random_seed: int = 42,
    dirichlet_concentration: float = 50.0,  # retained for API compatibility
) -> pd.DataFrame:
    """
    Generate a synthetic voter micro-panel for Scotland 2026.

    Vote intentions are sampled from per-voter probability vectors built from
    feature-based multiplicative adjustments on YouGov MRP regional priors,
    producing strong and realistic feature→label correlations.
    """
    rng = np.random.default_rng(random_seed)

    # ── region assignment ─────────────────────────────────────────────────────
    region_p = REGION_WEIGHTS / REGION_WEIGHTS.sum()
    regions  = rng.choice(REGIONS, size=n_voters, p=region_p)

    # ── demographics ──────────────────────────────────────────────────────────
    ages = rng.integers(18, 86, size=n_voters)
    age_group = pd.cut(
        ages,
        bins=[17, 24, 34, 44, 54, 64, 85],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)
    genders = rng.choice(["Male", "Female", "Other"], size=n_voters, p=[0.49, 0.49, 0.02])
    education = rng.choice(
        ["No qualifications", "Standard grades", "Highers", "HNC/HND", "Degree", "Postgraduate"],
        size=n_voters, p=[0.07, 0.18, 0.22, 0.12, 0.27, 0.14],
    )
    urban_rural = rng.choice(["Urban", "Suburban", "Rural"], size=n_voters, p=[0.45, 0.35, 0.20])

    # ── policy concern scores (0–10) ──────────────────────────────────────────
    economic_concern    = np.clip(rng.normal(6.5, 2.0, n_voters), 0, 10).round(2)
    health_concern      = np.clip(rng.normal(6.8, 1.9, n_voters), 0, 10).round(2)
    immigration_concern = np.clip(rng.normal(5.2, 2.5, n_voters), 0, 10).round(2)

    top_priority = rng.choice(
        ["Economy", "Health", "Immigration", "Independence", "Housing"],
        size=n_voters,
        p=[0.300, 0.280, 0.230, 0.070, 0.120],
    )

    # ── independence stance (5 levels) ───────────────────────────────────────
    indep_keys = list(INDEPENDENCE_PRIORS.keys())
    indep_p    = np.array(list(INDEPENDENCE_PRIORS.values()))
    indep_p    = indep_p / indep_p.sum()
    independence_stance = rng.choice(indep_keys, size=n_voters, p=indep_p)

    # ── previous vote ─────────────────────────────────────────────────────────
    previous_vote = rng.choice(PREV_PARTIES, size=n_voters, p=PREV_VOTE_PRIORS)

    # ── sentiment / identity scales ───────────────────────────────────────────
    nhs_satisfaction      = rng.choice([1, 2, 3, 4, 5], size=n_voters, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    cost_of_living_impact = rng.choice([1, 2, 3, 4, 5], size=n_voters, p=[0.08, 0.17, 0.30, 0.28, 0.17])
    party_id_strength     = rng.choice([0, 1, 2, 3],    size=n_voters, p=[0.20, 0.25, 0.30, 0.25])

    # ── vote intentions with feature-based probability adjustments ────────────
    con_probs = _compute_vote_probs(
        regions, independence_stance, top_priority, previous_vote, ages,
        CONSTITUENCY_PRIORS, is_list_vote=False,
    )
    constituency_votes = np.array([rng.choice(PARTIES, p=p) for p in con_probs])

    list_probs = _compute_vote_probs(
        regions, independence_stance, top_priority, previous_vote, ages,
        REGIONAL_LIST_PRIORS, is_list_vote=True,
    )
    list_votes = np.array([rng.choice(PARTIES, p=p) for p in list_probs])

    is_tactical = constituency_votes != list_votes

    return pd.DataFrame({
        "voter_id":             np.arange(1, n_voters + 1),
        "region":               regions,
        "age":                  ages,
        "age_group":            age_group,
        "gender":               genders,
        "education":            education,
        "urban_rural":          urban_rural,
        "economic_concern":     economic_concern,
        "health_concern":       health_concern,
        "immigration_concern":  immigration_concern,
        "top_priority":         top_priority,
        "independence_stance":  independence_stance,
        "constituency_vote":    constituency_votes,
        "list_vote":            list_votes,
        "is_tactical":          is_tactical,
        "previous_vote":        previous_vote,
        "party_id_strength":    party_id_strength,
        "nhs_satisfaction":     nhs_satisfaction,
        "cost_of_living_impact": cost_of_living_impact,
    })


def get_vote_share_summary(df: pd.DataFrame) -> dict:
    return {
        "constituency": df["constituency_vote"].value_counts(normalize=True).round(4).to_dict(),
        "list":         df["list_vote"].value_counts(normalize=True).round(4).to_dict(),
        "independence": df["independence_stance"].value_counts(normalize=True).round(4).to_dict(),
        "n_voters":     len(df),
        "tactical_rate": round(float(df["is_tactical"].mean()), 4),
    }


if __name__ == "__main__":
    cfg_path = Path(__file__).parents[2] / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    df = generate_voters(
        n_voters=cfg["data"]["n_voters"],
        random_seed=cfg["data"]["random_seed"],
        dirichlet_concentration=cfg["data"]["dirichlet_concentration"],
    )
    out_dir = Path(__file__).parents[2] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "voters.parquet", index=False)
    print(json.dumps(get_vote_share_summary(df), indent=2))
