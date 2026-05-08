"""
Marginal constituency analysis for Scotland 2026 election forecast.

Identifies the tightest seats by simulating constituency-level predictions
and computing the majority margin between the leading and second party.
Provides tactical voting recommendations for seats where a small vote
transfer could change the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Scotland's 73 constituencies, grouped by region ───────────────────────────
CONSTITUENCIES: dict[str, list[str]] = {
    "Central Scotland": [
        "Airdrie and Shotts",
        "Coatbridge and Chryston",
        "Cumbernauld and Kilsyth",
        "East Kilbride",
        "Hamilton, Larkhall and Stonehouse",
        "Motherwell and Wishaw",
        "Uddingston and Bellshill",
        "Falkirk East",
        "Falkirk West",
    ],
    "Glasgow": [
        "Glasgow Anniesland",
        "Glasgow Cathcart",
        "Glasgow Kelvin",
        "Glasgow Maryhill and Springburn",
        "Glasgow Pollok",
        "Glasgow Provan",
        "Glasgow Shettleston",
        "Glasgow Southside",
        "Rutherglen",
    ],
    "Highlands and Islands": [
        "Argyll and Bute",
        "Caithness, Sutherland and Ross",
        "Inverness and Nairn",
        "Moray",
        "Na h-Eileanan an Iar",
        "Orkney Islands",
        "Shetland Islands",
        "Skye, Lochaber and Badenoch",
    ],
    "Lothian": [
        "Almond Valley",
        "Edinburgh Central",
        "Edinburgh Eastern",
        "Edinburgh Northern and Leith",
        "Edinburgh Pentlands",
        "Edinburgh Southern",
        "Edinburgh Western",
        "Linlithgow",
        "Midlothian South, Tweeddale and Lauderdale",
    ],
    "Mid Scotland and Fife": [
        "Clackmannanshire and Dunblane",
        "Cowdenbeath",
        "Dunfermline",
        "Kirkcaldy",
        "Mid Fife and Glenrothes",
        "North East Fife",
        "Perthshire North",
        "Perthshire South and Kinross-shire",
        "Stirling",
    ],
    "North East Scotland": [
        "Aberdeen Central",
        "Aberdeen Donside",
        "Aberdeen South and North Kincardine",
        "Aberdeenshire East",
        "Aberdeenshire West",
        "Angus North and Mearns",
        "Angus South",
        "Dundee City East",
        "Dundee City West",
        "Banffshire and Buchan Coast",
    ],
    "South Scotland": [
        "Ayr",
        "Carrick, Cumnock and Doon Valley",
        "Clydesdale",
        "Dumfriesshire",
        "East Lothian",
        "Ettrick, Roxburgh and Berwickshire",
        "Galloway and West Dumfries",
        "Kilmarnock and Irvine Valley",
        "Cunninghame South",
    ],
    "West Scotland": [
        "Cunninghame North",
        "Dumbarton",
        "Eastwood",
        "Greenock and Inverclyde",
        "Paisley",
        "Renfrewshire North and West",
        "Renfrewshire South",
        "Strathkelvin and Bearsden",
        "West Dunbartonshire",
        "Clydebank and Milngavie",
    ],
}

ALL_CONSTITUENCIES: list[str] = [c for cs in CONSTITUENCIES.values() for c in cs]

CONSTITUENCY_TO_REGION: dict[str, str] = {
    c: r for r, cs in CONSTITUENCIES.items() for c in cs
}

PARTIES = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"]

# YouGov MRP priors used in demo mode
_PRIOR_PROBS = np.array([0.356, 0.179, 0.163, 0.103, 0.080, 0.060])


@dataclass
class MarginalSeat:
    constituency: str
    region: str
    leading_party: str
    second_party: str
    majority_margin_pp: float
    swing_needed_pp: float
    is_marginal: bool
    tactical_vote_recommendation: str


def tactical_swing_probability(
    from_party: str,
    to_party: str,
    margin_pp: float,
    *,
    base_willingness: float = 0.35,
) -> float:
    """
    Estimate the probability that tactical switching would flip a seat.

    Uses logistic decay: harder seats need broader coordination, reducing
    the probability that the swing actually materialises.

    Parameters
    ----------
    from_party:       Party the tactical voter currently intends to support.
    to_party:         Party they would switch to.
    margin_pp:        Current majority margin in percentage points.
    base_willingness: Fraction of from_party voters willing to switch (0–1).

    Returns
    -------
    float in [0, 1].
    """
    if margin_pp <= 0.0:
        return 1.0
    decay = float(np.exp(-margin_pp / 10.0))
    return round(float(np.clip(base_willingness * decay, 0.0, 1.0)), 4)


def _demo_marginals(threshold: float = 5.0) -> list[MarginalSeat]:
    """Return demo marginals from polling priors + Dirichlet noise."""
    rng = np.random.default_rng(42)
    seats: list[MarginalSeat] = []
    for region, constituencies in CONSTITUENCIES.items():
        for constituency in constituencies:
            noise = rng.dirichlet(_PRIOR_PROBS * 20)
            probs_pp = noise * 100.0
            sorted_idx = np.argsort(probs_pp)[::-1]
            lead_idx, second_idx = int(sorted_idx[0]), int(sorted_idx[1])
            margin = round(float(probs_pp[lead_idx] - probs_pp[second_idx]), 2)
            seats.append(
                MarginalSeat(
                    constituency=constituency,
                    region=region,
                    leading_party=PARTIES[lead_idx],
                    second_party=PARTIES[second_idx],
                    majority_margin_pp=margin,
                    swing_needed_pp=round(margin / 2.0, 2),
                    is_marginal=margin < threshold,
                    tactical_vote_recommendation=PARTIES[second_idx],
                )
            )
    return sorted(seats, key=lambda s: s.majority_margin_pp)


# ─────────────────────────────────────────────────────────────────────────────


def identify_marginals(
    ensemble: Any,
    pipeline: Any,
    threshold: float = 5.0,
    n_voters_per_constituency: int = 80,
) -> list[MarginalSeat]:
    """
    Use the trained ensemble to identify marginal constituencies.

    Generates synthetic voters for each constituency, runs them through the
    model, averages the predicted vote-share probabilities, and ranks seats
    by the gap between the leading and second party.

    Parameters
    ----------
    ensemble:                   Trained StackingEnsemble.
    pipeline:                   Fitted sklearn feature pipeline.
    threshold:                  Margin (pp) below which a seat is marginal.
    n_voters_per_constituency:  Synthetic voters to simulate per seat.

    Returns
    -------
    List of MarginalSeat sorted by majority_margin_pp ascending.
    """
    from src.data.generate_voters import generate_voters
    from src.features.pipeline import engineer_features, FEATURE_COLS

    seats: list[MarginalSeat] = []

    for region, constituencies in CONSTITUENCIES.items():
        for constituency in constituencies:
            seed = abs(hash(constituency)) % (2**31)
            df = generate_voters(n_voters=n_voters_per_constituency, random_seed=seed)
            df["region"] = region

            df_eng = engineer_features(df.copy())
            X_raw = df_eng[FEATURE_COLS]
            X = pipeline.transform(X_raw)

            probs_arr = ensemble.predict_proba(X)
            classes = ensemble.classes_

            mean_probs = probs_arr.mean(axis=0)
            probs_pp = {str(c): float(p * 100) for c, p in zip(classes, mean_probs)}

            sorted_parties = sorted(probs_pp, key=lambda p: probs_pp[p], reverse=True)
            lead = sorted_parties[0]
            second = sorted_parties[1]
            margin = round(probs_pp[lead] - probs_pp[second], 2)

            seats.append(
                MarginalSeat(
                    constituency=constituency,
                    region=region,
                    leading_party=lead,
                    second_party=second,
                    majority_margin_pp=margin,
                    swing_needed_pp=round(margin / 2.0, 2),
                    is_marginal=margin < threshold,
                    tactical_vote_recommendation=second,
                )
            )

    return sorted(seats, key=lambda s: s.majority_margin_pp)


def _demo_constituency_results(threshold: float = 5.0) -> list[dict]:
    """Return demo constituency results with full vote_shares from Dirichlet priors."""
    rng = np.random.default_rng(42)
    results = []
    for region, constituencies in CONSTITUENCIES.items():
        for constituency in constituencies:
            noise = rng.dirichlet(_PRIOR_PROBS * 20)
            vote_shares = {p: round(float(v), 4) for p, v in zip(PARTIES, noise)}
            sorted_parties = sorted(vote_shares, key=lambda p: vote_shares[p], reverse=True)
            winner, second = sorted_parties[0], sorted_parties[1]
            margin = round((vote_shares[winner] - vote_shares[second]) * 100, 2)
            is_marginal = margin < threshold
            results.append({
                "constituency": constituency,
                "region": region,
                "predicted_winner": winner,
                "vote_shares": vote_shares,
                "majority_margin_pp": margin,
                "is_marginal": is_marginal,
                "tactical_vote_recommendation": second if is_marginal else None,
            })
    return results


def constituency_results(
    ensemble: Any,
    pipeline: Any,
    threshold: float = 5.0,
    n_voters_per_constituency: int = 80,
) -> list[dict]:
    """Return constituency-level results including full vote_shares from the trained ensemble."""
    from src.data.generate_voters import generate_voters
    from src.features.pipeline import engineer_features, FEATURE_COLS

    results = []
    for region, constituencies in CONSTITUENCIES.items():
        for constituency in constituencies:
            seed = abs(hash(constituency)) % (2**31)
            df = generate_voters(n_voters=n_voters_per_constituency, random_seed=seed)
            df["region"] = region

            df_eng = engineer_features(df.copy())
            X_raw = df_eng[FEATURE_COLS]
            X = pipeline.transform(X_raw)

            probs_arr = ensemble.predict_proba(X)
            classes = ensemble.classes_

            mean_probs = probs_arr.mean(axis=0)
            vote_shares = {str(c): round(float(p), 4) for c, p in zip(classes, mean_probs)}
            sorted_parties = sorted(vote_shares, key=lambda p: vote_shares[p], reverse=True)
            winner, second = sorted_parties[0], sorted_parties[1]
            margin = round((vote_shares[winner] - vote_shares[second]) * 100, 2)
            is_marginal = margin < threshold

            results.append({
                "constituency": constituency,
                "region": region,
                "predicted_winner": winner,
                "vote_shares": vote_shares,
                "majority_margin_pp": margin,
                "is_marginal": is_marginal,
                "tactical_vote_recommendation": second if is_marginal else None,
            })
    return results
