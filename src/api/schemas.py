"""Pydantic request/response schemas for the election forecast API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class VoterFeatures(BaseModel):
    region: str = Field(..., description="Scottish Parliament region")
    age: int = Field(..., ge=18, le=85)
    age_group: str = Field(..., description="Age band, e.g. '25-34'")
    gender: str
    education: str
    urban_rural: str
    economic_concern: float = Field(..., ge=0.0, le=10.0)
    health_concern: float = Field(..., ge=0.0, le=10.0)
    immigration_concern: float = Field(..., ge=0.0, le=10.0)
    top_priority: str
    independence_stance: str = Field(..., pattern="^(Yes|No|Undecided)$")
    previous_vote: str
    party_id_strength: int = Field(..., ge=0, le=3)
    nhs_satisfaction: int = Field(..., ge=1, le=5)
    cost_of_living_impact: int = Field(..., ge=1, le=5)

    model_config = {"json_schema_extra": {
        "example": {
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
            "independence_stance": "Yes",
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
