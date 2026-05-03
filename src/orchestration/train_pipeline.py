"""
End-to-end training pipeline for Scotland 2026 election forecast.

Steps
-----
1. Load or generate voter data
2. Validate schema and class balance
3. Feature engineering (ColumnTransformer pipeline)
4. Stratified train / val / test split
5. Train all five base models independently with Optuna HPO
6. Build stacking ensemble with LR meta-learner
7. Evaluate on held-out test set (accuracy, F1-macro, Brier score)
8. Compute SHAP feature importances
9. Log everything to MLflow
10. Persist ensemble + feature pipeline to disk
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder

from src.data.generate_voters import generate_voters, get_vote_share_summary
from src.features.pipeline import build_feature_pipeline, engineer_features, FEATURE_COLS, TARGET_COL
from src.models.base_models import (
    train_catboost,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.models.ensemble import StackingEnsemble
from src.models.explainability import compute_shap_values, get_feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

PARTIES: list[str] = _CFG["model"]["classes"]


# ── validation ────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> None:
    required = [
        "voter_id", "region", "age", "constituency_vote", "list_vote",
        "independence_stance", "economic_concern", "health_concern",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    unknown_parties = set(df["constituency_vote"].unique()) - set(PARTIES)
    if unknown_parties:
        raise ValueError(f"Unknown parties in data: {unknown_parties}")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Null values found in columns: {null_cols}")

    log.info("DataFrame validation passed — %d rows, %d columns", len(df), len(df.columns))


# ── Brier score helper ────────────────────────────────────────────────────────

def multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray, classes: list) -> float:
    """Mean Brier score over all classes (one-vs-rest).

    Accepts either string or integer-encoded y_true.  When classes is a list
    of integers the encoding step is skipped.
    """
    scores = []
    if len(classes) > 0 and isinstance(classes[0], str):
        # Use caller-supplied order so y_prob[:, i] aligns with classes[i].
        # LabelEncoder sorts alphabetically and breaks that alignment.
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_enc = np.array([class_to_idx[c] for c in y_true])
    else:
        y_enc = np.asarray(y_true)
    for i in range(len(classes)):
        y_bin = (y_enc == i).astype(int)
        scores.append(brier_score_loss(y_bin, y_prob[:, i]))
    return float(np.mean(scores))


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    data_path: Optional[Path] = None,
    n_voters: int = _CFG["data"]["n_voters"],
    random_seed: int = _CFG["data"]["random_seed"],
    n_trials: int = _CFG["model"]["optuna_trials"],
    model_dir: Path = Path("models"),
    tracking_uri: str = _CFG["mlflow"]["tracking_uri"],
    experiment_name: str = _CFG["mlflow"]["experiment_name"],
) -> dict:
    """
    Run the full training pipeline.

    Parameters
    ----------
    data_path: If provided, load parquet; otherwise generate synthetic data.
    n_voters:  Number of synthetic voters to generate.
    random_seed: RNG seed for reproducibility.
    n_trials:  Optuna trials per base learner.
    model_dir: Directory to write model artefacts.
    tracking_uri: MLflow tracking URI.
    experiment_name: MLflow experiment name.

    Returns
    -------
    dict of final test metrics.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. data ────────────────────────────────────────────────────────────
    if data_path and Path(data_path).exists():
        log.info("Loading voter data from %s", data_path)
        df = pd.read_parquet(data_path)
    else:
        log.info("Generating %d synthetic voters (seed=%d)", n_voters, random_seed)
        df = generate_voters(n_voters=n_voters, random_seed=random_seed)
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_dir / "voters.parquet", index=False)

    # ── 2. validate ────────────────────────────────────────────────────────
    validate_dataframe(df)
    vote_summary = get_vote_share_summary(df)
    log.info("Vote share summary: %s", json.dumps(vote_summary["constituency"], indent=2))

    # ── 3. feature engineering ─────────────────────────────────────────────
    df_eng = engineer_features(df.copy())
    X_raw = df_eng[FEATURE_COLS]
    y = df_eng[TARGET_COL].values

    # ── 4. split ───────────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y, test_size=0.15, stratify=y, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=random_seed
    )
    log.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # ── 5. feature pipeline fit ────────────────────────────────────────────
    feat_pipeline = build_feature_pipeline()
    X_tr = feat_pipeline.fit_transform(X_train)
    X_vl = feat_pipeline.transform(X_val)
    X_te = feat_pipeline.transform(X_test)

    # ── 6. train base models ───────────────────────────────────────────────
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    base_results: dict[str, dict] = {}
    trainers = {
        "xgboost": train_xgboost,
        "lightgbm": train_lightgbm,
        "catboost": train_catboost,
        "random_forest": train_random_forest,
        "logistic_regression": train_logistic_regression,
    }

    with mlflow.start_run(run_name=f"{_CFG['mlflow']['run_name_prefix']}-base-models"):
        mlflow.log_params({"n_voters": n_voters, "n_trials": n_trials, "seed": random_seed})
        for name, trainer_fn in trainers.items():
            log.info("Training %s ...", name)
            clf, best_params, val_f1 = trainer_fn(X_tr, y_train, X_vl, y_val, n_trials=n_trials)
            base_results[name] = {"clf": clf, "params": best_params, "val_f1": val_f1}
            mlflow.log_metric(f"{name}_val_f1", val_f1)
            mlflow.log_params({f"{name}__{k}": v for k, v in best_params.items()})
            log.info("  %s val F1-macro: %.4f", name, val_f1)

    # ── 7. stacking ensemble ───────────────────────────────────────────────
    log.info("Training stacking ensemble ...")
    ensemble = StackingEnsemble(n_trials=n_trials)
    ensemble.fit(X_tr, y_train)

    # ── 8. evaluate ────────────────────────────────────────────────────────
    # ensemble.predict() returns decoded string labels; internal probs are
    # indexed by the integer-encoded class order from ensemble.le_
    test_preds = ensemble.predict(X_te)          # string party names
    test_preds_enc = ensemble.le_.transform(test_preds)
    test_probs = ensemble.predict_proba(X_te)
    y_test_enc = ensemble.le_.transform(y_test)

    test_metrics = {
        "test_accuracy": round(accuracy_score(y_test, test_preds), 4),
        "test_f1_macro": round(f1_score(y_test, test_preds, average="macro", zero_division=0), 4),
        "test_log_loss": round(log_loss(y_test_enc, test_probs), 4),
        "test_brier_score": round(multiclass_brier(y_test_enc, test_probs, list(range(len(ensemble.classes_)))), 4),
    }
    per_party_f1 = {
        f"test_f1_{ensemble.classes_[i]}": round(v, 4)
        for i, v in enumerate(f1_score(y_test_enc, test_preds_enc, average=None, zero_division=0))
    }
    test_metrics.update(per_party_f1)
    log.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    # ── 9. SHAP ────────────────────────────────────────────────────────────
    log.info("Computing SHAP values ...")
    try:
        shap_vals, _ = compute_shap_values(ensemble, X_te, max_samples=300)
        # Get feature names from the fitted pipeline
        try:
            feat_names = feat_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        except AttributeError:
            feat_names = [f"f{i}" for i in range(shap_vals.shape[1])]
        shap_df = get_feature_importance(shap_vals, feat_names, top_n=20)
        shap_path = model_dir / "shap_importance.csv"
        shap_df.to_csv(shap_path, index=False)
        log.info("SHAP importance saved to %s", shap_path)
    except Exception as exc:
        log.warning("SHAP computation failed: %s", exc)
        shap_path = None

    # ── 10. MLflow final run ───────────────────────────────────────────────
    with mlflow.start_run(run_name=f"{_CFG['mlflow']['run_name_prefix']}-ensemble"):
        mlflow.log_metrics(test_metrics)
        ens_path = model_dir / "ensemble.pkl"
        pipe_path = model_dir / "pipeline.pkl"
        joblib.dump(ensemble, ens_path)
        joblib.dump(feat_pipeline, pipe_path)
        mlflow.log_artifact(str(ens_path))
        mlflow.log_artifact(str(pipe_path))
        if shap_path and Path(shap_path).exists():
            mlflow.log_artifact(str(shap_path))

    # ── 11. persist seat projection ────────────────────────────────────────
    region_col = df_eng["region"] if "region" in df_eng.columns else None
    _save_seat_projection(ensemble, feat_pipeline, X_te, df_eng.iloc[-len(X_te):])

    log.info("Pipeline complete.  Models saved to %s", model_dir)
    return test_metrics


def _save_seat_projection(ensemble, feat_pipeline, X_te: np.ndarray, df_slice: pd.DataFrame) -> None:
    """Persist D'Hondt seat projection from test-set predictions."""
    from src.models.dhondt import project_seats_from_shares

    probs = ensemble.predict_proba(X_te)
    classes = list(ensemble.model_.classes_)
    mean_probs = probs.mean(axis=0)
    con_shares = {c: float(p) for c, p in zip(classes, mean_probs)}
    list_shares = con_shares  # approximate list shares from same model

    allocation = project_seats_from_shares(con_shares, list_shares)

    def _native(d: dict) -> dict:
        return {str(k): int(v) for k, v in d.items()}

    out = {
        "constituency": _native(allocation.constituency),
        "regional": _native(allocation.regional),
    }
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "seat_projection.yaml", "w") as f:
        yaml.safe_dump(out, f)
    log.info("Seat projection saved.")
