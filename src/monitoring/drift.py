"""
Evidently-based drift and model quality monitoring.

Two report types:
  DataDriftPreset     — compares training distribution vs recent predictions.
  ClassificationPreset — model quality on labelled evaluation data.

Prediction logging appends each /predict call to a JSONL file so that
future drift reports have a growing pool of real-world inputs to compare
against the training reference.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MONITORING_DIR = Path("data/monitoring")
PREDICTIONS_LOG = MONITORING_DIR / "predictions.jsonl"
REPORTS_DIR = Path("reports")


def log_prediction(
    voter_features: dict[str, Any],
    prediction: str,
    probability: float,
) -> None:
    """
    Append one prediction to the JSONL log.

    Safe to call from the API — never raises, so a logging failure cannot
    break the prediction response.
    """
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "probability": round(float(probability), 4),
        **{k: v for k, v in voter_features.items()},
    }
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def generate_drift_report(
    reference_path: Path | str,
    current_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """
    Generate an Evidently data-drift HTML report.

    Parameters
    ----------
    reference_path : Path to reference parquet (e.g. data/raw/voters.parquet).
    current_path   : Path to current data — parquet file or JSONL log.
    output_path    : Destination for the HTML report.

    Returns
    -------
    dict with keys: n_features_drifted, total_features, drift_detected, report_path.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    reference_path = Path(reference_path)
    current_path = Path(current_path)
    output_path = Path(output_path)

    reference = pd.read_parquet(reference_path)

    if str(current_path).endswith(".jsonl"):
        rows = []
        with open(current_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        current = pd.DataFrame(rows)
    else:
        current = pd.read_parquet(current_path)

    # Align to shared columns only
    common_cols = [c for c in reference.columns if c in current.columns]
    reference = reference[common_cols].copy()
    current = current[common_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))

    result_dict = report.as_dict()
    drift_result = result_dict.get("metrics", [{}])[0].get("result", {})
    n_drifted = int(drift_result.get("number_of_drifted_columns", 0))
    n_total = int(drift_result.get("number_of_columns", len(common_cols)))
    drift_detected = bool(drift_result.get("dataset_drift", False))

    return {
        "n_features_drifted": n_drifted,
        "total_features": n_total,
        "drift_detected": drift_detected,
        "report_path": str(output_path),
    }


def generate_model_quality_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.DataFrame | np.ndarray,
    output_path: Path | str,
) -> None:
    """
    Generate an Evidently classification quality HTML report.

    Parameters
    ----------
    y_true      : Ground-truth party labels.
    y_pred      : Predicted party labels.
    y_proba     : Predicted probabilities — shape (n_samples, n_classes).
    output_path : Destination for the HTML report.
    """
    from evidently.report import Report
    from evidently.metric_preset import ClassificationPreset

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(y_proba, np.ndarray):
        y_proba_df = pd.DataFrame(y_proba)
    else:
        y_proba_df = y_proba.reset_index(drop=True)

    eval_df = pd.DataFrame({
        "target": np.asarray(y_true),
        "prediction": np.asarray(y_pred),
    })
    eval_df = pd.concat([eval_df, y_proba_df], axis=1)

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=None, current_data=eval_df)
    report.save_html(str(output_path))
