"""Pydantic request/response schemas for the election forecast API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


def _age_to_group(age: int) -> str:
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


class VoterFeatures(BaseModel):
    region: str = Field(..., description="Scottish Parliament region")
    age: int = Field(..., ge=18, le=85)
    age_group: Optional[str] = Field(None, description="Age band (auto-derived from age if omitted)")
    gender: str
    education: str
    urban_rural: str
    economic_concern: float = Field(..., ge=0.0, le=10.0)
    health_concern: float = Field(..., ge=0.0, le=10.0)
    immigration_concern: float = Field(..., ge=0.0, le=10.0)
    top_priority: str
    independence_stance: str = Field(..., pattern="^(Strong Yes|Lean Yes|Undecided|Lean No|Strong No)$")
    previous_vote: str
    party_id_strength: int = Field(..., ge=0, le=3)
    nhs_satisfaction: int = Field(..., ge=1, le=5)
    cost_of_living_impact: int = Field(..., ge=1, le=5)

    @model_validator(mode="after")
    def derive_age_group(self) -> "VoterFeatures":
        if self.age_group is None:
            self.age_group = _age_to_group(self.age)
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "region": "Glasgow",
            "age": 34,
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
    }}


class PredictionResponse(BaseModel):
    predicted_party: str
    probabilities: dict[str, float]
    tactical_swing_index: float
    indep_economy_interaction: float
    nhs_dissatisfaction: float


class BatchPredictRequest(BaseModel):
    voters: list[VoterFeatures] = Field(..., max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]
    n_voters: int


class SeatProjectionResponse(BaseModel):
    constituency: dict[str, int]
    regional: dict[str, int]
    total: dict[str, int]
    majority_threshold: int
    governing_party: Optional[str]
    has_majority: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    base_learners: list[str]
    meta_learner: str
    classes: list[str]
    n_features: int
    is_loaded: bool
    mlflow_run_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MarginalSeatResponse(BaseModel):
    constituency: str
    region: str
    leading_party: str
    second_party: str
    majority_margin_pp: float
    swing_needed_pp: float
    is_marginal: bool
    tactical_vote_recommendation: str


class MarginalsResponse(BaseModel):
    seats: list[MarginalSeatResponse]
    n_marginal: int
    threshold_pp: float
    demo_mode: bool


class RetrainStatusResponse(BaseModel):
    status: str  # idle / queued / running / complete / failed
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None
