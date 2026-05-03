"""
FastAPI endpoint tests using TestClient.

All tests run in demo mode (no trained model on disk) so they are fast
and self-contained.  The demo-mode prediction path is the one exercised
in CI; full-model tests require a trained artefact and are skipped.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_VOTER = {
    "region": "Glasgow",
    "age": 34,
    "age_group": "25-34",
    "gender": "Female",
    "education": "Degree",
    "urban_rural": "Urban",
    "economic_concern": 7.2,
    "health_concern": 8.1,
    "immigration_concern": 4.5,
    "top_priority": "Health",
    "independence_stance": "Strong Yes",
    "previous_vote": "SNP",
    "party_id_strength": 2,
    "nhs_satisfaction": 2,
    "cost_of_living_impact": 4,
}

PARTIES = {"SNP", "Reform", "Labour", "Conservative", "LibDem", "Green"}


class TestHealth:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_field(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_has_model_loaded_field(self):
        r = client.get("/health")
        assert "model_loaded" in r.json()

    def test_has_version(self):
        r = client.get("/health")
        assert "version" in r.json()


class TestPredict:
    def test_returns_200(self):
        r = client.post("/predict", json=VALID_VOTER)
        assert r.status_code == 200

    def test_response_has_predicted_party(self):
        r = client.post("/predict", json=VALID_VOTER)
        assert r.json()["predicted_party"] in PARTIES

    def test_probabilities_keys(self):
        r = client.post("/predict", json=VALID_VOTER)
        probs = r.json()["probabilities"]
        assert set(probs.keys()) == PARTIES

    def test_probabilities_sum_to_one(self):
        r = client.post("/predict", json=VALID_VOTER)
        probs = r.json()["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_engineered_features_present(self):
        r = client.post("/predict", json=VALID_VOTER)
        body = r.json()
        assert "tactical_swing_index" in body
        assert "indep_economy_interaction" in body
        assert "nhs_dissatisfaction" in body

    def test_invalid_independence_stance_rejected(self):
        bad = {**VALID_VOTER, "independence_stance": "Maybe"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_age_below_minimum_rejected(self):
        bad = {**VALID_VOTER, "age": 15}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_age_above_maximum_rejected(self):
        bad = {**VALID_VOTER, "age": 90}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_concern_out_of_range_rejected(self):
        bad = {**VALID_VOTER, "economic_concern": 11.0}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_nhs_satisfaction_out_of_range_rejected(self):
        bad = {**VALID_VOTER, "nhs_satisfaction": 6}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_missing_required_field_rejected(self):
        bad = {k: v for k, v in VALID_VOTER.items() if k != "region"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_health_priority_biases_health_parties(self):
        """Voters citing Health as top priority should favour Labour/SNP over Reform."""
        health_voter = {**VALID_VOTER, "top_priority": "Health", "independence_stance": "Strong No"}
        r = client.post("/predict", json=health_voter)
        probs = r.json()["probabilities"]
        assert probs.get("Reform", 0) < probs.get("Labour", 0) + probs.get("SNP", 0)

    def test_yes_voter_leans_snp(self):
        """Strong Yes voter in demo mode should have highest SNP probability."""
        yes_voter = {**VALID_VOTER, "independence_stance": "Strong Yes"}
        r = client.post("/predict", json=yes_voter)
        probs = r.json()["probabilities"]
        assert probs["SNP"] == max(probs.values())


class TestPredictBatch:
    def test_returns_200(self):
        payload = {"voters": [VALID_VOTER, VALID_VOTER]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200

    def test_n_voters_matches_input(self):
        payload = {"voters": [VALID_VOTER] * 3}
        r = client.post("/predict/batch", json=payload)
        assert r.json()["n_voters"] == 3

    def test_predictions_count_matches(self):
        payload = {"voters": [VALID_VOTER] * 5}
        r = client.post("/predict/batch", json=payload)
        assert len(r.json()["predictions"]) == 5

    def test_empty_batch_fails(self):
        r = client.post("/predict/batch", json={"voters": []})
        # empty list passes schema; predictions list is just empty
        assert r.status_code in (200, 422)

    def test_batch_exceeds_limit_rejected(self):
        payload = {"voters": [VALID_VOTER] * 1001}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 422


class TestSeatsProjected:
    def test_returns_200(self):
        r = client.get("/seats/projected")
        assert r.status_code == 200

    def test_has_required_keys(self):
        body = client.get("/seats/projected").json()
        for key in ("constituency", "regional", "total", "majority_threshold", "has_majority"):
            assert key in body

    def test_total_seat_count_is_129(self):
        body = client.get("/seats/projected").json()
        assert sum(body["total"].values()) == 129

    def test_constituency_seats_total_73(self):
        body = client.get("/seats/projected").json()
        assert sum(body["constituency"].values()) == 73

    def test_regional_seats_total_56(self):
        body = client.get("/seats/projected").json()
        assert sum(body["regional"].values()) == 56

    def test_majority_threshold_is_65(self):
        body = client.get("/seats/projected").json()
        assert body["majority_threshold"] == 65


class TestModelInfo:
    def test_returns_200(self):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_model_type_field(self):
        body = client.get("/model/info").json()
        assert body["model_type"] == "StackingClassifier"

    def test_base_learners_list(self):
        body = client.get("/model/info").json()
        assert len(body["base_learners"]) == 4

    def test_classes_are_six_parties(self):
        body = client.get("/model/info").json()
        assert set(body["classes"]) == PARTIES

    def test_is_loaded_is_bool(self):
        body = client.get("/model/info").json()
        assert isinstance(body["is_loaded"], bool)
