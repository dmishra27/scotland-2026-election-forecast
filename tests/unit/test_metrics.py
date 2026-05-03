"""
Unit tests for D'Hondt allocation and Brier score metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.dhondt import (
    DHondtAllocator,
    SeatAllocation,
    allocate_seats,
    project_seats_from_shares,
)
from src.orchestration.train_pipeline import multiclass_brier

PARTIES = ["SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"]


# ── D'Hondt unit tests ─────────────────────────────────────────────────────

class TestDHondtAllocator:
    def test_allocates_correct_number_of_seats(self):
        allocator = DHondtAllocator(seats_per_region=7)
        shares = {"SNP": 0.30, "Reform": 0.20, "Labour": 0.20,
                  "Conservative": 0.15, "LibDem": 0.10, "Green": 0.05}
        result = allocator.allocate_regional(shares, {})
        assert sum(result.values()) == 7

    def test_dominant_party_gets_most_seats(self):
        allocator = DHondtAllocator(seats_per_region=7)
        shares = {"SNP": 0.50, "Reform": 0.20, "Labour": 0.15,
                  "Conservative": 0.08, "LibDem": 0.05, "Green": 0.02}
        result = allocator.allocate_regional(shares, {})
        assert result["SNP"] >= result["Reform"]

    def test_constituency_seats_reduce_list_award(self):
        """A party that swept constituencies gets fewer regional seats."""
        allocator = DHondtAllocator(seats_per_region=7)
        shares = {"SNP": 0.40, "Reform": 0.20, "Labour": 0.20,
                  "Conservative": 0.10, "LibDem": 0.06, "Green": 0.04}
        # SNP wins 9 constituency seats in this hypothetical region
        without_adj = allocator.allocate_regional(shares, {})
        with_adj = allocator.allocate_regional(shares, {"SNP": 9})
        assert with_adj["SNP"] <= without_adj["SNP"]

    def test_zero_vote_share_gets_no_seats(self):
        allocator = DHondtAllocator(seats_per_region=7)
        shares = {"SNP": 0.50, "Reform": 0.30, "Labour": 0.20,
                  "Conservative": 0.00, "LibDem": 0.00, "Green": 0.00}
        result = allocator.allocate_regional(shares, {})
        assert result["Conservative"] == 0
        assert result["LibDem"] == 0
        assert result["Green"] == 0

    def test_all_regions_allocated(self):
        allocator = DHondtAllocator(seats_per_region=7)
        shares = {"SNP": 0.30, "Reform": 0.20, "Labour": 0.20,
                  "Conservative": 0.15, "LibDem": 0.10, "Green": 0.05}
        shares_by_region = {r: shares for r in [
            "Central Scotland", "Glasgow", "Highlands and Islands", "Lothian",
            "Mid Scotland and Fife", "North East Scotland", "South Scotland", "West Scotland",
        ]}
        result = allocator.allocate_all_regions(shares_by_region, {})
        assert len(result) == 8
        for region, r_seats in result.items():
            assert sum(r_seats.values()) == 7

    def test_empty_vote_share_returns_zeros(self):
        allocator = DHondtAllocator(seats_per_region=7)
        result = allocator.allocate_regional({}, {})
        assert sum(result.values()) == 0


class TestSeatAllocation:
    def setup_method(self):
        self.alloc = SeatAllocation(
            constituency={"SNP": 50, "Labour": 10, "Conservative": 8, "LibDem": 5,
                          "Reform": 0, "Green": 0},
            regional={"SNP": 17, "Reform": 20, "Labour": 5, "Conservative": 1,
                      "LibDem": 2, "Green": 11},
        )

    def test_total_seats_is_129(self):
        assert sum(self.alloc.total.values()) == 129

    def test_total_sum_correct(self):
        totals = self.alloc.total
        assert totals["SNP"] == 67
        assert totals["Green"] == 11

    def test_has_majority_correct(self):
        _, has_maj = self.alloc.has_majority
        assert has_maj is True

    def test_to_dataframe_shape(self):
        df = self.alloc.to_dataframe()
        assert len(df) == len(PARTIES)
        assert "total" in df.columns

    def test_to_dataframe_sorted_by_seats(self):
        df = self.alloc.to_dataframe()
        assert df["total"].iloc[0] >= df["total"].iloc[1]


class TestProjectSeatsFromShares:
    def test_total_seats_is_129(self):
        con = {"SNP": 0.356, "Reform": 0.179, "Labour": 0.163,
               "Conservative": 0.103, "LibDem": 0.080, "Green": 0.060}
        lst = {"SNP": 0.291, "Reform": 0.180, "Labour": 0.153,
               "Conservative": 0.111, "LibDem": 0.090, "Green": 0.085}
        alloc = project_seats_from_shares(con, lst)
        assert sum(alloc.total.values()) == 129

    def test_constituency_seats_total_73(self):
        con = {"SNP": 0.356, "Reform": 0.179, "Labour": 0.163,
               "Conservative": 0.103, "LibDem": 0.080, "Green": 0.060}
        lst = {"SNP": 0.291, "Reform": 0.180, "Labour": 0.153,
               "Conservative": 0.111, "LibDem": 0.090, "Green": 0.085}
        alloc = project_seats_from_shares(con, lst)
        assert sum(alloc.constituency.values()) == 73

    def test_regional_seats_total_56(self):
        con = {"SNP": 0.356, "Reform": 0.179, "Labour": 0.163,
               "Conservative": 0.103, "LibDem": 0.080, "Green": 0.060}
        lst = {"SNP": 0.291, "Reform": 0.180, "Labour": 0.153,
               "Conservative": 0.111, "LibDem": 0.090, "Green": 0.085}
        alloc = project_seats_from_shares(con, lst)
        assert sum(alloc.regional.values()) == 56

    def test_top_party_wins_most_seats(self):
        con = {"SNP": 0.50, "Reform": 0.20, "Labour": 0.15,
               "Conservative": 0.08, "LibDem": 0.04, "Green": 0.03}
        lst = {"SNP": 0.35, "Reform": 0.25, "Labour": 0.18,
               "Conservative": 0.10, "LibDem": 0.07, "Green": 0.05}
        alloc = project_seats_from_shares(con, lst)
        top = max(alloc.total, key=lambda p: alloc.total[p])
        assert top == "SNP"


# ── Brier score unit tests ─────────────────────────────────────────────────

class TestMulticlassBrier:
    def test_perfect_prediction_gives_zero(self):
        classes = PARTIES
        y_true = np.array(["SNP", "Reform", "Labour"])
        # Perfect: probability 1.0 on the correct class
        y_prob = np.zeros((3, 6))
        y_prob[0, 0] = 1.0  # SNP
        y_prob[1, 1] = 1.0  # Reform
        y_prob[2, 2] = 1.0  # Labour
        score = multiclass_brier(y_true, y_prob, classes)
        assert score < 0.01

    def test_uniform_prediction_gives_high_score(self):
        classes = PARTIES
        y_true = np.array(["SNP"] * 100)
        y_prob = np.full((100, 6), 1 / 6)
        score = multiclass_brier(y_true, y_prob, classes)
        # Mean Brier for uniform over 6 classes on one true class ≈ 0.139
        assert score > 0.10

    def test_score_between_zero_and_two(self):
        rng = np.random.default_rng(42)
        y_true = rng.choice(PARTIES, 200)
        raw = rng.dirichlet(np.ones(6), 200)
        score = multiclass_brier(y_true, raw, PARTIES)
        assert 0.0 <= score <= 2.0

    def test_better_model_has_lower_brier(self):
        """A model that concentrates probability correctly beats random."""
        classes = PARTIES
        y_true = np.array(["SNP"] * 50)

        # Good model: SNP probability 0.7
        good_prob = np.zeros((50, 6))
        good_prob[:, 0] = 0.70
        good_prob[:, 1:] = 0.06
        good_score = multiclass_brier(y_true, good_prob, classes)

        # Bad model: uniform
        bad_prob = np.full((50, 6), 1 / 6)
        bad_score = multiclass_brier(y_true, bad_prob, classes)

        assert good_score < bad_score
