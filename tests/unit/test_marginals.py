"""
Unit tests for marginal constituency analysis.

All tests run without a trained model — they exercise _demo_marginals and
tactical_swing_probability, plus the API endpoint in demo mode.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.models.marginals import (
    ALL_CONSTITUENCIES,
    CONSTITUENCIES,
    MarginalSeat,
    _demo_marginals,
    tactical_swing_probability,
)


PARTIES = {"SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"}


class TestConstituencyConstants:
    def test_total_constituency_count(self):
        assert len(ALL_CONSTITUENCIES) == 73

    def test_region_count(self):
        assert len(CONSTITUENCIES) == 8

    def test_no_duplicate_constituencies(self):
        assert len(ALL_CONSTITUENCIES) == len(set(ALL_CONSTITUENCIES))

    def test_constituency_sums_per_region(self):
        expected = {
            "Central Scotland": 9,
            "Glasgow": 9,
            "Highlands and Islands": 8,
            "Lothian": 9,
            "Mid Scotland and Fife": 9,
            "North East Scotland": 10,
            "South Scotland": 9,
            "West Scotland": 10,
        }
        for region, count in expected.items():
            assert len(CONSTITUENCIES[region]) == count, (
                f"{region}: expected {count}, got {len(CONSTITUENCIES[region])}"
            )


class TestIdentifyMarginals:
    @pytest.fixture(scope="class")
    def demo_seats(self):
        return _demo_marginals(threshold=5.0)

    def test_returns_list(self, demo_seats):
        assert isinstance(demo_seats, list)

    def test_returns_all_73_seats(self, demo_seats):
        assert len(demo_seats) == 73

    def test_all_items_are_marginal_seats(self, demo_seats):
        for seat in demo_seats:
            assert isinstance(seat, MarginalSeat)

    def test_sorted_ascending_by_margin(self, demo_seats):
        margins = [s.majority_margin_pp for s in demo_seats]
        assert margins == sorted(margins)

    def test_leading_party_is_valid(self, demo_seats):
        for seat in demo_seats:
            assert seat.leading_party in PARTIES

    def test_second_party_is_valid(self, demo_seats):
        for seat in demo_seats:
            assert seat.second_party in PARTIES

    def test_lead_differs_from_second(self, demo_seats):
        for seat in demo_seats:
            assert seat.leading_party != seat.second_party

    def test_margin_is_non_negative(self, demo_seats):
        for seat in demo_seats:
            assert seat.majority_margin_pp >= 0.0

    def test_swing_needed_is_half_margin(self, demo_seats):
        for seat in demo_seats:
            assert abs(seat.swing_needed_pp - seat.majority_margin_pp / 2.0) < 0.01


class TestMarginalThreshold:
    def test_is_marginal_flag_below_threshold(self):
        seats = _demo_marginals(threshold=5.0)
        for seat in seats:
            if seat.majority_margin_pp < 5.0:
                assert seat.is_marginal is True
            else:
                assert seat.is_marginal is False

    def test_threshold_zero_gives_no_marginals(self):
        seats = _demo_marginals(threshold=0.0)
        assert all(not s.is_marginal for s in seats)

    def test_threshold_100_gives_all_marginals(self):
        seats = _demo_marginals(threshold=100.0)
        assert all(s.is_marginal for s in seats)


class TestTacticalSwingProbability:
    def test_probability_is_in_zero_one(self):
        for margin in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]:
            p = tactical_swing_probability("SNP", "Reform", margin)
            assert 0.0 <= p <= 1.0, f"Out of range at margin={margin}: {p}"

    def test_zero_margin_returns_one(self):
        assert tactical_swing_probability("SNP", "Labour", 0.0) == 1.0

    def test_smaller_margin_gives_higher_probability(self):
        p_small = tactical_swing_probability("SNP", "Reform", 2.0)
        p_large = tactical_swing_probability("SNP", "Reform", 15.0)
        assert p_small > p_large

    def test_parties_do_not_affect_magnitude_directly(self):
        p1 = tactical_swing_probability("SNP", "Labour", 5.0)
        p2 = tactical_swing_probability("Conservative", "LibDem", 5.0)
        assert p1 == p2

    def test_returns_float(self):
        result = tactical_swing_probability("SNP", "Reform", 3.0)
        assert isinstance(result, float)


class TestApiMarginalsEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        from src.api.main import app
        return TestClient(app)

    def test_endpoint_returns_200(self, client):
        r = client.get("/seats/marginals")
        assert r.status_code == 200

    def test_returns_20_seats(self, client):
        r = client.get("/seats/marginals")
        body = r.json()
        assert len(body["seats"]) == 20

    def test_threshold_pp_is_5(self, client):
        r = client.get("/seats/marginals")
        assert r.json()["threshold_pp"] == 5.0

    def test_demo_mode_flag(self, client):
        r = client.get("/seats/marginals")
        assert r.json()["demo_mode"] is True

    def test_seats_sorted_ascending(self, client):
        r = client.get("/seats/marginals")
        margins = [s["majority_margin_pp"] for s in r.json()["seats"]]
        assert margins == sorted(margins)

    def test_required_fields_present(self, client):
        r = client.get("/seats/marginals")
        seat = r.json()["seats"][0]
        required = {
            "constituency", "region", "leading_party", "second_party",
            "majority_margin_pp", "swing_needed_pp", "is_marginal",
            "tactical_vote_recommendation",
        }
        assert required.issubset(set(seat.keys()))

    def test_leading_party_is_valid(self, client):
        r = client.get("/seats/marginals")
        for seat in r.json()["seats"]:
            assert seat["leading_party"] in PARTIES

    def test_n_marginal_is_non_negative(self, client):
        r = client.get("/seats/marginals")
        assert r.json()["n_marginal"] >= 0
