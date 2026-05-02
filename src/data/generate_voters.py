"""
Synthetic voter data generator for Scotland 2026 Scottish Parliament election.

Draws individual vote intentions from Dirichlet distributions centred on the
YouGov MRP priors (April 2026, n=3,925).  Regional adjustments shift each
region's alpha vector before sampling, reflecting known geographic swing patterns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── YouGov MRP priors (April 2026) ─────────────────────────────────────────
PARTIES = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"]

CONSTITUENCY_PRIORS: dict[str, float] = {
    "SNP": 0.356,
    "Reform": 0.179,
    "Labour": 0.163,
    "Conservative": 0.103,
    "LibDem": 0.080,
    "Green": 0.060,
}

REGIONAL_LIST_PRIORS: dict[str, float] = {
    "SNP": 0.291,
    "Reform": 0.180,
    "Labour": 0.153,
    "Conservative": 0.111,
    "LibDem": 0.090,
    "Green": 0.085,
}

INDEPENDENCE_PRIORS: dict[str, float] = {
    "Yes": 0.466,
    "No": 0.446,
    "Undecided": 0.089,
}

REGIONS: list[str] = [
    "Central Scotland",
    "Glasgow",
    "Highlands and Islands",
    "Lothian",
    "Mid Scotland and Fife",
    "North East Scotland",
    "South Scotland",
    "West Scotland",
]

# Electorate-share weights per region
REGION_WEIGHTS: np.ndarray = np.array(
    [0.135, 0.160, 0.085, 0.155, 0.135, 0.120, 0.115, 0.095]
)

# Multiplicative regional adjustments applied to both constituency and list priors
REGION_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "Central Scotland": {"SNP": 1.05, "Labour": 1.10, "Reform": 0.95},
    "Glasgow": {"SNP": 1.00, "Labour": 1.15, "Green": 1.20, "Reform": 0.90},
    "Highlands and Islands": {"SNP": 1.10, "LibDem": 1.30, "Reform": 0.85},
    "Lothian": {"SNP": 0.95, "Green": 1.25, "Labour": 1.05, "LibDem": 1.15},
    "Mid Scotland and Fife": {"SNP": 1.05, "Labour": 1.05, "Reform": 1.00},
    "North East Scotland": {"SNP": 1.05, "Conservative": 1.10, "Reform": 1.10},
    "South Scotland": {"Conservative": 1.20, "Reform": 1.15, "SNP": 0.90},
    "West Scotland": {"SNP": 1.00, "Labour": 1.10, "Reform": 1.05},
}

# 2021 result priors for previous-vote recall
PREV_PARTIES = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green", "Did not vote"]
PREV_VOTE_PRIORS = np.array([0.400, 0.000, 0.220, 0.227, 0.053, 0.068, 0.032])


# ── helpers ─────────────────────────────────────────────────────────────────

def _adjusted_alpha(base_priors: dict[str, float], region: str) -> np.ndarray:
    """Return a normalised probability vector after regional adjustments."""
    adj = REGION_ADJUSTMENTS.get(region, {})
    adjusted = {p: base_priors[p] * adj.get(p, 1.0) for p in PARTIES}
    total = sum(adjusted.values())
    return np.array([adjusted[p] / total for p in PARTIES])


def _dirichlet_vote(
    alpha_priors: np.ndarray,
    concentration: float,
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Sample individual votes via Dirichlet-Multinomial."""
    shares = rng.dirichlet(alpha_priors * concentration, size=n)
    return [rng.choice(PARTIES, p=s) for s in shares]


# ── main generator ──────────────────────────────────────────────────────────

def generate_voters(
    n_voters: int = 12_500,
    random_seed: int = 42,
    dirichlet_concentration: float = 50.0,
) -> pd.DataFrame:
    """
    Generate a synthetic voter micro-panel for Scotland 2026.

    Parameters
    ----------
    n_voters:
        Number of synthetic voters to create.
    random_seed:
        NumPy RNG seed for full reproducibility.
    dirichlet_concentration:
        Concentration parameter κ.  Higher values pull each voter's
        effective vote probability closer to the regional mean; lower
        values introduce more individual-level noise.

    Returns
    -------
    pd.DataFrame with one row per voter and columns covering demographics,
    policy priorities, NHS/cost-of-living sentiment, and vote intentions.
    """
    rng = np.random.default_rng(random_seed)

    # ── region assignment ──────────────────────────────────────────────────
    region_p = REGION_WEIGHTS / REGION_WEIGHTS.sum()
    regions = rng.choice(REGIONS, size=n_voters, p=region_p)

    # ── demographics ──────────────────────────────────────────────────────
    ages = rng.integers(18, 86, size=n_voters)
    age_group = pd.cut(
        ages,
        bins=[17, 24, 34, 44, 54, 64, 85],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)
    genders = rng.choice(["Male", "Female", "Other"], size=n_voters, p=[0.49, 0.49, 0.02])
    education = rng.choice(
        ["No qualifications", "Standard grades", "Highers", "HNC/HND", "Degree", "Postgraduate"],
        size=n_voters,
        p=[0.07, 0.18, 0.22, 0.12, 0.27, 0.14],
    )
    urban_rural = rng.choice(["Urban", "Suburban", "Rural"], size=n_voters, p=[0.45, 0.35, 0.20])

    # ── policy concern scores (0–10) ───────────────────────────────────────
    economic_concern = np.clip(rng.normal(6.5, 2.0, n_voters), 0, 10).round(2)
    health_concern = np.clip(rng.normal(6.8, 1.9, n_voters), 0, 10).round(2)
    immigration_concern = np.clip(rng.normal(5.2, 2.5, n_voters), 0, 10).round(2)

    top_priority = rng.choice(
        ["Economy", "Health", "Immigration", "Independence"],
        size=n_voters,
        p=[0.348, 0.319, 0.261, 0.072],
    )

    # ── independence stance ────────────────────────────────────────────────
    indep_p = np.array([INDEPENDENCE_PRIORS[k] for k in ["Yes", "No", "Undecided"]])
    independence_stance = rng.choice(["Yes", "No", "Undecided"], size=n_voters, p=indep_p)

    # ── vote intentions by region ──────────────────────────────────────────
    constituency_votes = np.empty(n_voters, dtype=object)
    list_votes = np.empty(n_voters, dtype=object)

    for region in REGIONS:
        mask = regions == region
        n_r = int(mask.sum())
        if n_r == 0:
            continue
        con_alpha = _adjusted_alpha(CONSTITUENCY_PRIORS, region)
        constituency_votes[mask] = _dirichlet_vote(con_alpha, dirichlet_concentration, n_r, rng)
        lst_alpha = _adjusted_alpha(REGIONAL_LIST_PRIORS, region)
        list_votes[mask] = _dirichlet_vote(lst_alpha, dirichlet_concentration, n_r, rng)

    is_tactical = constituency_votes != list_votes

    # ── sentiment scales ───────────────────────────────────────────────────
    nhs_satisfaction = rng.choice([1, 2, 3, 4, 5], size=n_voters, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    cost_of_living_impact = rng.choice(
        [1, 2, 3, 4, 5], size=n_voters, p=[0.08, 0.17, 0.30, 0.28, 0.17]
    )

    # ── party identity ─────────────────────────────────────────────────────
    party_id_strength = rng.choice([0, 1, 2, 3], size=n_voters, p=[0.20, 0.25, 0.30, 0.25])
    previous_vote = rng.choice(PREV_PARTIES, size=n_voters, p=PREV_VOTE_PRIORS)

    return pd.DataFrame(
        {
            "voter_id": np.arange(1, n_voters + 1),
            "region": regions,
            "age": ages,
            "age_group": age_group,
            "gender": genders,
            "education": education,
            "urban_rural": urban_rural,
            "economic_concern": economic_concern,
            "health_concern": health_concern,
            "immigration_concern": immigration_concern,
            "top_priority": top_priority,
            "independence_stance": independence_stance,
            "constituency_vote": constituency_votes,
            "list_vote": list_votes,
            "is_tactical": is_tactical,
            "previous_vote": previous_vote,
            "party_id_strength": party_id_strength,
            "nhs_satisfaction": nhs_satisfaction,
            "cost_of_living_impact": cost_of_living_impact,
        }
    )


def get_vote_share_summary(df: pd.DataFrame) -> dict:
    """Return constituency and list vote-share summaries as plain dicts."""
    return {
        "constituency": df["constituency_vote"].value_counts(normalize=True).round(4).to_dict(),
        "list": df["list_vote"].value_counts(normalize=True).round(4).to_dict(),
        "independence": df["independence_stance"].value_counts(normalize=True).round(4).to_dict(),
        "n_voters": len(df),
        "tactical_rate": round(float(df["is_tactical"].mean()), 4),
    }


# ── CLI entry-point ──────────────────────────────────────────────────────────
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

    summary = get_vote_share_summary(df)
    print(json.dumps(summary, indent=2))
