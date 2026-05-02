"""
D'Hondt proportional seat allocation for Scotland 2026.

The Scottish Parliament uses the Additional Member System (AMS):
  - 73 constituency MSPs elected by First Past the Post (FPTP).
  - 56 regional list MSPs (8 regions × 7 seats) allocated by D'Hondt.

D'Hondt adjusts list-seat allocation by dividing each party's regional list
vote by (1 + seats already won in that region), preventing parties that won
many constituency seats from also sweeping the list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_CFG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

PARTIES: list[str] = _CFG["model"]["classes"]
REGIONS: list[str] = _CFG["data"]["regions"]
REGIONAL_SEATS_PER_REGION: int = _CFG["seat_projection"]["regional_seats_per_region"]
MAJORITY_THRESHOLD: int = _CFG["seat_projection"]["majority_threshold"]


@dataclass
class SeatAllocation:
    constituency: dict[str, int] = field(default_factory=dict)
    regional: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> dict[str, int]:
        all_parties = set(list(self.constituency.keys()) + list(self.regional.keys()))
        return {p: self.constituency.get(p, 0) + self.regional.get(p, 0) for p in all_parties}

    @property
    def has_majority(self) -> tuple[str | None, bool]:
        totals = self.total
        for party, seats in totals.items():
            if seats >= MAJORITY_THRESHOLD:
                return party, True
        return None, False

    def to_dataframe(self) -> pd.DataFrame:
        totals = self.total
        rows = [
            {
                "party": p,
                "constituency": self.constituency.get(p, 0),
                "regional": self.regional.get(p, 0),
                "total": totals[p],
            }
            for p in sorted(totals, key=lambda x: -totals[x])
        ]
        return pd.DataFrame(rows)


class DHondtAllocator:
    """
    Allocates regional list seats using the D'Hondt method.

    The adjustment for constituency seats already won ensures proportionality
    across the full AMS system (regional list votes are divided by
    1 + constituency seats won in that region).
    """

    def __init__(self, seats_per_region: int = REGIONAL_SEATS_PER_REGION):
        self.seats_per_region = seats_per_region

    def allocate_regional(
        self,
        regional_vote_shares: dict[str, float],
        constituency_seats_won: dict[str, int],
    ) -> dict[str, int]:
        """
        Run D'Hondt for one region.

        Parameters
        ----------
        regional_vote_shares:
            Normalised list-vote shares for this region.
        constituency_seats_won:
            Constituency seats already won in this region (adjusts divisors).
        """
        seats_allocated: dict[str, int] = {p: 0 for p in PARTIES}
        total_votes = sum(regional_vote_shares.values())
        if total_votes == 0:
            return seats_allocated

        # Start with constituency-seat adjustment
        constituency_adj: dict[str, int] = {p: constituency_seats_won.get(p, 0) for p in PARTIES}

        for _ in range(self.seats_per_region):
            quotients = {
                p: regional_vote_shares.get(p, 0.0) / (1 + seats_allocated[p] + constituency_adj[p])
                for p in PARTIES
            }
            winner = max(quotients, key=lambda p: quotients[p])
            seats_allocated[winner] += 1

        return seats_allocated

    def allocate_all_regions(
        self,
        regional_vote_shares_by_region: dict[str, dict[str, float]],
        constituency_seats_by_region: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, int]]:
        """Allocate regional list seats for all 8 regions."""
        return {
            region: self.allocate_regional(
                regional_vote_shares_by_region.get(region, {}),
                constituency_seats_by_region.get(region, {}),
            )
            for region in REGIONS
        }


def _fptp_constituency_seats(
    constituency_vote_probs: pd.DataFrame,
    constituencies_per_region: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Simulate FPTP constituency results per region.

    For each synthetic constituency draw the party with the highest
    predicted probability wins the seat.  Number of constituencies per
    region is proportional to the actual Scottish Parliament distribution.
    """
    if constituencies_per_region is None:
        constituencies_per_region = {
            "Central Scotland": 9,
            "Glasgow": 9,
            "Highlands and Islands": 8,
            "Lothian": 9,
            "Mid Scotland and Fife": 9,
            "North East Scotland": 10,
            "South Scotland": 9,
            "West Scotland": 10,
        }

    seats: dict[str, dict[str, int]] = {r: {p: 0 for p in PARTIES} for r in REGIONS}
    region_col = constituency_vote_probs.index.name or "region"

    # If the dataframe has a region column use it; otherwise distribute evenly
    if "region" in constituency_vote_probs.columns:
        for region in REGIONS:
            n_con = constituencies_per_region[region]
            region_probs = constituency_vote_probs[constituency_vote_probs["region"] == region]
            if len(region_probs) == 0:
                continue
            party_cols = [c for c in PARTIES if c in region_probs.columns]
            mean_probs = region_probs[party_cols].mean()
            # Allocate constituency seats proportionally via repeated argmax draws
            for _ in range(n_con):
                winner = mean_probs.idxmax()
                seats[region][winner] += 1
    else:
        mean_probs = constituency_vote_probs[[p for p in PARTIES if p in constituency_vote_probs.columns]].mean()
        for region in REGIONS:
            n_con = constituencies_per_region[region]
            winner = mean_probs.idxmax()
            seats[region][winner] += n_con

    return seats


def allocate_seats(
    constituency_probs: pd.DataFrame,
    list_probs: pd.DataFrame,
) -> SeatAllocation:
    """
    Full AMS seat allocation from voter-level probability arrays.

    Parameters
    ----------
    constituency_probs:
        DataFrame with one row per voter, columns = party names + 'region'.
        Values are predicted vote probabilities for constituency ballot.
    list_probs:
        Same structure for regional list ballot.

    Returns
    -------
    SeatAllocation with constituency and regional seat dicts.
    """
    allocator = DHondtAllocator()

    # ── constituency seats ─────────────────────────────────────────────────
    con_seats_by_region = _fptp_constituency_seats(constituency_probs)
    total_con_seats: dict[str, int] = {}
    for region_seats in con_seats_by_region.values():
        for party, n in region_seats.items():
            total_con_seats[party] = total_con_seats.get(party, 0) + n

    # ── regional list vote shares ──────────────────────────────────────────
    list_vote_cols = [p for p in PARTIES if p in list_probs.columns]
    regional_shares: dict[str, dict[str, float]] = {}

    for region in REGIONS:
        if "region" in list_probs.columns:
            region_df = list_probs[list_probs["region"] == region]
        else:
            region_df = list_probs

        if len(region_df) == 0:
            regional_shares[region] = {p: 0.0 for p in PARTIES}
            continue

        mean_shares = region_df[list_vote_cols].mean().to_dict()
        total = sum(mean_shares.values())
        regional_shares[region] = (
            {p: mean_shares.get(p, 0.0) / total for p in PARTIES} if total > 0
            else {p: 0.0 for p in PARTIES}
        )

    # ── regional list seats ────────────────────────────────────────────────
    regional_by_region = allocator.allocate_all_regions(regional_shares, con_seats_by_region)
    total_regional_seats: dict[str, int] = {}
    for region_seats in regional_by_region.values():
        for party, n in region_seats.items():
            total_regional_seats[party] = total_regional_seats.get(party, 0) + n

    return SeatAllocation(constituency=total_con_seats, regional=total_regional_seats)


def project_seats_from_shares(
    constituency_shares: dict[str, float],
    list_shares: dict[str, float],
) -> SeatAllocation:
    """
    Quick seat projection directly from aggregate vote shares (no voter micro-data).

    Useful for dashboard sliders where users adjust poll numbers in real-time.
    """
    allocator = DHondtAllocator()

    # Equal constituency seats across all regions using FPTP winner-takes-all
    con_seats: dict[str, int] = {p: 0 for p in PARTIES}
    con_by_region: dict[str, dict[str, int]] = {}
    constituencies_per_region = {
        "Central Scotland": 9, "Glasgow": 9, "Highlands and Islands": 8,
        "Lothian": 9, "Mid Scotland and Fife": 9, "North East Scotland": 10,
        "South Scotland": 9, "West Scotland": 10,
    }
    fptp_winner = max(constituency_shares, key=lambda p: constituency_shares.get(p, 0))
    for region in REGIONS:
        n = constituencies_per_region[region]
        con_seats[fptp_winner] = con_seats.get(fptp_winner, 0) + n
        con_by_region[region] = {fptp_winner: n, **{p: 0 for p in PARTIES if p != fptp_winner}}

    list_by_region = {region: list_shares for region in REGIONS}
    regional_by_region = allocator.allocate_all_regions(list_by_region, con_by_region)
    regional_total: dict[str, int] = {}
    for r_seats in regional_by_region.values():
        for p, n in r_seats.items():
            regional_total[p] = regional_total.get(p, 0) + n

    return SeatAllocation(constituency=con_seats, regional=regional_total)
