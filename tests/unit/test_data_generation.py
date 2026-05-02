"""Unit tests for the synthetic voter data generator."""

import numpy as np
import pandas as pd
import pytest

from src.data.generate_voters import (
    PARTIES,
    REGIONS,
    generate_voters,
    get_vote_share_summary,
)


class TestGenerateVoters:
    def test_row_count(self):
        df = generate_voters(n_voters=500, random_seed=1)
        assert len(df) == 500

    def test_required_columns(self):
        df = generate_voters(n_voters=100, random_seed=2)
        required = [
            "voter_id", "region", "age", "age_group", "gender", "education",
            "urban_rural", "economic_concern", "health_concern",
            "immigration_concern", "top_priority", "independence_stance",
            "constituency_vote", "list_vote", "is_tactical",
            "previous_vote", "party_id_strength", "nhs_satisfaction",
            "cost_of_living_impact",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_vote_columns_valid_parties(self):
        df = generate_voters(n_voters=300, random_seed=3)
        assert set(df["constituency_vote"].unique()).issubset(set(PARTIES))
        assert set(df["list_vote"].unique()).issubset(set(PARTIES))

    def test_regions_valid(self):
        df = generate_voters(n_voters=300, random_seed=4)
        assert set(df["region"].unique()).issubset(set(REGIONS))

    def test_age_range(self):
        df = generate_voters(n_voters=300, random_seed=5)
        assert df["age"].min() >= 18
        assert df["age"].max() <= 85

    def test_concern_scores_bounded(self):
        df = generate_voters(n_voters=300, random_seed=6)
        for col in ["economic_concern", "health_concern", "immigration_concern"]:
            assert df[col].min() >= 0.0
            assert df[col].max() <= 10.0

    def test_nhs_satisfaction_range(self):
        df = generate_voters(n_voters=300, random_seed=7)
        assert set(df["nhs_satisfaction"].unique()).issubset({1, 2, 3, 4, 5})

    def test_party_id_strength_range(self):
        df = generate_voters(n_voters=300, random_seed=8)
        assert set(df["party_id_strength"].unique()).issubset({0, 1, 2, 3})

    def test_voter_ids_unique(self):
        df = generate_voters(n_voters=500, random_seed=9)
        assert df["voter_id"].nunique() == 500

    def test_reproducibility(self):
        df1 = generate_voters(n_voters=200, random_seed=42)
        df2 = generate_voters(n_voters=200, random_seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_voters(n_voters=200, random_seed=1)
        df2 = generate_voters(n_voters=200, random_seed=2)
        assert not df1["constituency_vote"].equals(df2["constituency_vote"])

    def test_independence_stance_values(self):
        df = generate_voters(n_voters=1000, random_seed=10)
        assert set(df["independence_stance"].unique()).issubset({"Yes", "No", "Undecided"})

    def test_tactical_voter_detection(self):
        df = generate_voters(n_voters=500, random_seed=11)
        expected_tactical = df["constituency_vote"] != df["list_vote"]
        pd.testing.assert_series_equal(df["is_tactical"], expected_tactical, check_names=False)

    def test_constituency_snp_share_approx_prior(self):
        """SNP constituency share should be within ±5pp of the 35.6% prior."""
        df = generate_voters(n_voters=12_500, random_seed=42)
        snp_share = (df["constituency_vote"] == "SNP").mean()
        assert abs(snp_share - 0.356) < 0.05, f"SNP share {snp_share:.3f} too far from prior"

    def test_all_regions_represented(self):
        df = generate_voters(n_voters=5000, random_seed=13)
        assert set(df["region"].unique()) == set(REGIONS)


class TestGetVoteShareSummary:
    def test_summary_keys(self, sample_voters):
        summary = get_vote_share_summary(sample_voters)
        assert set(summary.keys()) == {"constituency", "list", "independence", "n_voters", "tactical_rate"}

    def test_summary_shares_sum_to_one(self, sample_voters):
        summary = get_vote_share_summary(sample_voters)
        assert abs(sum(summary["constituency"].values()) - 1.0) < 1e-6
        assert abs(sum(summary["list"].values()) - 1.0) < 1e-6

    def test_tactical_rate_between_zero_and_one(self, sample_voters):
        summary = get_vote_share_summary(sample_voters)
        assert 0.0 <= summary["tactical_rate"] <= 1.0

    def test_n_voters_matches(self, sample_voters):
        summary = get_vote_share_summary(sample_voters)
        assert summary["n_voters"] == len(sample_voters)
