"""
FastAPI serving layer for Scotland 2026 election forecast.

Demo-mode fallback: when no trained model is present on disk the API
returns predictions derived from the YouGov MRP polling priors so every
endpoint is always functional for testing and the Streamlit dashboard.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MarginalSeatResponse,
    MarginalsResponse,
    ModelInfoResponse,
    PredictionResponse,
    RetrainStatusResponse,
    SeatProjectionResponse,
    VoterFeatures,
)

# ── config ───────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

PARTIES: list[str] = _CFG["model"]["classes"]
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/ensemble.pkl"))
PIPELINE_PATH = Path(os.getenv("PIPELINE_PATH", "models/pipeline.pkl"))
VERSION = "0.1.0"

_RETRAIN_LOG = Path("/tmp/retrain.log")
_LATEST_MODEL_PATH = Path("models/latest/ensemble.pkl")
_LATEST_PIPELINE_PATH = Path("models/latest/pipeline.pkl")
_METRICS_PATH = Path("metrics/train_metrics.json")

# ── MRP prior probabilities (YouGov April 2026) ──────────────────────────────
_PRIOR_CON = {
    "SNP": 0.356, "Reform": 0.179, "Labour": 0.163,
    "Conservative": 0.103, "LibDem": 0.080, "Green": 0.060,
}
_PRIOR_LIST = {
    "SNP": 0.291, "Reform": 0.180, "Labour": 0.153,
    "Conservative": 0.111, "LibDem": 0.090, "Green": 0.085,
}

# ── app state ────────────────────────────────────────────────────────────────
_state: dict[str, Any] = {
    "ensemble": None,
    "pipeline": None,
    "demo_mode": True,
    "marginals_cache": None,  # computed once after model load
}

_retrain_job: dict[str, Any] = {"status": "idle"}
_retrain_lock = threading.Lock()


def _try_load_models() -> None:
    """Load ensemble + pipeline from disk; silently stay in demo mode if absent."""
    try:
        import joblib

        if MODEL_PATH.exists() and PIPELINE_PATH.exists():
            _state["ensemble"] = joblib.load(MODEL_PATH)
            _state["pipeline"] = joblib.load(PIPELINE_PATH)
            _state["demo_mode"] = False
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load_models()
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Scotland 2026 Election Forecast API",
    description="Voter-level vote-intention prediction and D'Hondt seat projection.",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _voter_to_row(v: VoterFeatures) -> pd.DataFrame:
    return pd.DataFrame([v.model_dump()])


def _engineer_row(row: pd.DataFrame) -> pd.DataFrame:
    from src.features.pipeline import engineer_features
    return engineer_features(row.copy())


def _engineered_features(row: pd.DataFrame) -> tuple[float, float, float]:
    eng = _engineer_row(row)
    return (
        float(eng["tactical_swing_index"].iloc[0]),
        float(eng["indep_economy_interaction"].iloc[0]),
        float(eng["nhs_dissatisfaction"].iloc[0]),
    )


def _demo_predict(voter: VoterFeatures) -> PredictionResponse:
    """Rule-based prediction using polling priors + simple adjustments."""
    probs = dict(_PRIOR_CON)

    # Independence stance → SNP / unionist bias
    if voter.independence_stance == "Strong Yes":
        probs["SNP"] = min(probs["SNP"] * 1.5, 0.85)
        probs["Green"] = min(probs["Green"] * 1.2, 0.85)
    elif voter.independence_stance == "Lean Yes":
        probs["SNP"] = min(probs["SNP"] * 1.25, 0.85)
    elif voter.independence_stance == "Lean No":
        probs["SNP"] *= 0.75
        probs["Conservative"] = min(probs["Conservative"] * 1.3, 0.85)
    elif voter.independence_stance == "Strong No":
        probs["SNP"] *= 0.55
        probs["Conservative"] = min(probs["Conservative"] * 1.6, 0.85)

    # Top priority → party boost
    boosts = {
        "Economy": "Reform", "Health": "Labour",
        "Immigration": "Reform", "Independence": "SNP", "Housing": "Green",
    }
    boosted = boosts.get(voter.top_priority)
    if boosted:
        probs[boosted] = probs[boosted] * 1.20

    # Normalise
    total = sum(probs.values())
    probs = {p: round(v / total, 4) for p, v in probs.items()}
    predicted = max(probs, key=lambda p: probs[p])

    row = _voter_to_row(voter)
    tsi, iei, nd = _engineered_features(row)

    return PredictionResponse(
        predicted_party=predicted,
        probabilities=probs,
        tactical_swing_index=round(tsi, 4),
        indep_economy_interaction=round(iei, 4),
        nhs_dissatisfaction=round(nd, 4),
    )


def _model_predict(voter: VoterFeatures) -> PredictionResponse:
    """Model-based prediction using the trained ensemble."""
    row = _voter_to_row(voter)
    eng = _engineer_row(row)

    from src.features.pipeline import FEATURE_COLS
    X_raw = eng[FEATURE_COLS]
    X = _state["pipeline"].transform(X_raw)
    probs_arr = _state["ensemble"].predict_proba(X)[0]
    classes = _state["ensemble"].classes_
    probs = {str(c): round(float(p), 4) for c, p in zip(classes, probs_arr)}
    predicted = max(probs, key=lambda p: probs[p])

    tsi, iei, nd = _engineered_features(row)
    return PredictionResponse(
        predicted_party=predicted,
        probabilities=probs,
        tactical_swing_index=round(tsi, 4),
        indep_economy_interaction=round(iei, 4),
        nhs_dissatisfaction=round(nd, 4),
    )


# ── retrain worker ───────────────────────────────────────────────────────────

def _run_retrain() -> None:
    _retrain_job["status"] = "running"
    _retrain_job["started_at"] = datetime.datetime.utcnow().isoformat()
    try:
        _RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_RETRAIN_LOG, "w") as log_fh:
            proc = subprocess.run(
                [
                    sys.executable, "scripts/train_models.py",
                    "--n-trials", "20",
                    "--model-dir", "models/latest",
                ],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )

        if proc.returncode != 0:
            _retrain_job["status"] = "failed"
            _retrain_job["error"] = f"Training process exited with code {proc.returncode}"
            _retrain_job["finished_at"] = datetime.datetime.utcnow().isoformat()
            return

        metrics: dict[str, Any] = {}
        if _METRICS_PATH.exists():
            with open(_METRICS_PATH) as f:
                metrics = json.load(f)

        import joblib
        if _LATEST_MODEL_PATH.exists() and _LATEST_PIPELINE_PATH.exists():
            _state["ensemble"] = joblib.load(_LATEST_MODEL_PATH)
            _state["pipeline"] = joblib.load(_LATEST_PIPELINE_PATH)
            _state["demo_mode"] = False
            _state["marginals_cache"] = None

        _retrain_job["status"] = "complete"
        _retrain_job["metrics"] = metrics
        _retrain_job["finished_at"] = datetime.datetime.utcnow().isoformat()

    except Exception as exc:
        _retrain_job["status"] = "failed"
        _retrain_job["error"] = str(exc)
        _retrain_job["finished_at"] = datetime.datetime.utcnow().isoformat()


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=not _state["demo_mode"],
        version=VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(voter: VoterFeatures) -> PredictionResponse:
    result = _demo_predict(voter) if _state["demo_mode"] else _model_predict(voter)
    try:
        from src.monitoring.drift import log_prediction
        top_prob = max(result.probabilities.values())
        log_prediction(voter.model_dump(), result.predicted_party, top_prob)
    except Exception:
        pass
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    if len(req.voters) > _CFG["api"]["max_batch_size"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size exceeds max {_CFG['api']['max_batch_size']}",
        )
    fn = _demo_predict if _state["demo_mode"] else _model_predict
    predictions = [fn(v) for v in req.voters]
    return BatchPredictResponse(predictions=predictions, n_voters=len(predictions))


@app.get("/seats/projected", response_model=SeatProjectionResponse, tags=["seats"])
def seats_projected() -> SeatProjectionResponse:
    """
    Return D'Hondt seat projection.

    In demo mode uses the YouGov MRP priors directly.
    In model mode uses aggregate predicted vote shares from the held-out
    voter sample (written to disk at training time).
    """
    from src.models.dhondt import project_seats_from_shares

    results_path = Path("data/processed/seat_projection.yaml")
    cached = None
    if results_path.exists():
        with open(results_path) as f:
            cached = yaml.safe_load(f)
    if cached and "constituency" in cached and "regional" in cached:
        allocation = type("_A", (), cached)()
        con = cached["constituency"]
        reg = cached["regional"]
    else:
        allocation = project_seats_from_shares(_PRIOR_CON, _PRIOR_LIST)
        con = allocation.constituency
        reg = allocation.regional

    total = {p: con.get(p, 0) + reg.get(p, 0) for p in PARTIES}
    gov_party, has_maj = None, False
    for p, s in total.items():
        if s >= _CFG["seat_projection"]["majority_threshold"]:
            gov_party, has_maj = p, True
            break

    return SeatProjectionResponse(
        constituency=con,
        regional=reg,
        total=total,
        majority_threshold=_CFG["seat_projection"]["majority_threshold"],
        governing_party=gov_party,
        has_majority=has_maj,
    )


@app.get("/seats/marginals", response_model=MarginalsResponse, tags=["seats"])
def seats_marginals() -> MarginalsResponse:
    """
    Return the top 20 tightest marginal constituencies.

    In demo mode uses YouGov MRP priors with Dirichlet noise.
    In model mode uses the trained ensemble to simulate constituency-level
    vote shares (result cached after first call to avoid repeated inference).
    """
    from src.models.marginals import identify_marginals, _demo_marginals

    threshold = 5.0

    if _state["demo_mode"]:
        all_seats = _demo_marginals(threshold=threshold)
    else:
        if _state["marginals_cache"] is None:
            _state["marginals_cache"] = identify_marginals(
                _state["ensemble"], _state["pipeline"], threshold=threshold
            )
        all_seats = _state["marginals_cache"]

    top20 = all_seats[:20]
    return MarginalsResponse(
        seats=[
            MarginalSeatResponse(
                constituency=s.constituency,
                region=s.region,
                leading_party=s.leading_party,
                second_party=s.second_party,
                majority_margin_pp=s.majority_margin_pp,
                swing_needed_pp=s.swing_needed_pp,
                is_marginal=s.is_marginal,
                tactical_vote_recommendation=s.tactical_vote_recommendation,
            )
            for s in top20
        ],
        n_marginal=sum(1 for s in all_seats if s.is_marginal),
        threshold_pp=threshold,
        demo_mode=_state["demo_mode"],
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    n_features = 0
    run_id = None
    if not _state["demo_mode"]:
        try:
            n_features = _state["pipeline"].n_features_in_
        except AttributeError:
            pass

    return ModelInfoResponse(
        model_type="StackingClassifier",
        base_learners=_CFG["model"]["base_learners"],
        meta_learner=_CFG["model"]["meta_learner"],
        classes=PARTIES,
        n_features=n_features,
        is_loaded=not _state["demo_mode"],
        mlflow_run_id=run_id,
    )


@app.post(
    "/model/retrain",
    response_model=RetrainStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["meta"],
)
def retrain() -> RetrainStatusResponse:
    with _retrain_lock:
        if _retrain_job.get("status") in ("queued", "running"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A retrain job is already in progress.",
            )
        _retrain_job.clear()
        _retrain_job.update({
            "status": "queued",
            "started_at": None,
            "finished_at": None,
            "metrics": None,
            "error": None,
        })

    threading.Thread(target=_run_retrain, daemon=True).start()
    return RetrainStatusResponse(**_retrain_job)


@app.get("/model/retrain/status", response_model=RetrainStatusResponse, tags=["meta"])
def retrain_status() -> RetrainStatusResponse:
    return RetrainStatusResponse(**_retrain_job)
