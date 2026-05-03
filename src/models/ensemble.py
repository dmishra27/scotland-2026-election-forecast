"""
Stacking ensemble for Scotland 2026 vote-intention classification.

Architecture
------------
Base learners (level-0):
    XGBoostClassifier, LGBMClassifier, CatBoostClassifier, RandomForestClassifier

Meta-learner (level-1):
    LogisticRegression with L2 regularisation

Each base learner is tuned independently with Optuna (n_trials from config).
The final stacking model is trained on out-of-fold predictions from CV,
with all artefacts logged to MLflow.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

_CFG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

N_TRIALS: int = _CFG["model"]["optuna_trials"]
CV_FOLDS: int = _CFG["model"]["cv_folds"]
CLASSES: list[str] = _CFG["model"]["classes"]


# ── Optuna search spaces ─────────────────────────────────────────────────────

def _xgb_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return log_loss(y, oof)


def _lgbm_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "verbose": -1,
        "n_jobs": -1,
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return log_loss(y, oof)


def _catboost_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "verbose": False,
        "allow_writing_files": False,
    }
    model = CatBoostClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return log_loss(y, oof)


def _rf_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return log_loss(y, oof)


_OBJECTIVES = {
    "xgboost": _xgb_objective,
    "lightgbm": _lgbm_objective,
    "catboost": _catboost_objective,
    "random_forest": _rf_objective,
}


def _tune_estimator(name: str, X: np.ndarray, y: np.ndarray, n_trials: int) -> dict[str, Any]:
    """Run Optuna study and return best params for *name*."""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _OBJECTIVES[name](trial, X, y),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params


def _build_estimator(name: str, params: dict[str, Any]):
    """Instantiate a base learner from tuned params."""
    if name == "xgboost":
        return XGBClassifier(**params, use_label_encoder=False, eval_metric="mlogloss", verbosity=0, n_jobs=-1)
    if name == "lightgbm":
        return LGBMClassifier(**params, verbose=-1, n_jobs=-1)
    if name == "catboost":
        return CatBoostClassifier(**params, verbose=False, allow_writing_files=False)
    if name == "random_forest":
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown estimator: {name}")


# ── StackingEnsemble class ───────────────────────────────────────────────────

class StackingEnsemble:
    """Stacking ensemble with Optuna-tuned base learners and LR meta-learner.

    Internally encodes string class labels to integers so XGBoost 3.x and
    other strict classifiers are satisfied.  The public API always accepts
    and returns original string party labels.
    """

    def __init__(self, n_trials: int = N_TRIALS, cv_folds: int = CV_FOLDS):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params_: dict[str, dict] = {}
        self.model_: StackingClassifier | None = None
        self.classes_: list[str] = CLASSES
        self.le_: LabelEncoder | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        # Encode string labels → integers for XGBoost/other strict learners
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = list(self.le_.classes_)

        estimators = []
        for name in _CFG["model"]["base_learners"]:
            params = _tune_estimator(name, X, y_enc, self.n_trials)
            self.best_params_[name] = params
            estimators.append((name, _build_estimator(name, params)))

        meta = LogisticRegression(max_iter=1000, C=1.0)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        self.model_ = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=cv,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        self.model_.fit(X, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return string party labels."""
        return self.le_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y_enc = self.le_.transform(y)
        preds_enc = self.model_.predict(X)
        probs = self.predict_proba(X)
        return {
            "accuracy": round(accuracy_score(y_enc, preds_enc), 4),
            "f1_macro": round(f1_score(y_enc, preds_enc, average="macro"), 4),
            "log_loss": round(log_loss(y_enc, probs), 4),
        }

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "StackingEnsemble":
        return joblib.load(path)


# ── top-level training function ───────────────────────────────────────────────

def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    model_save_path: Path | None = None,
) -> StackingEnsemble:
    """Train, evaluate, and MLflow-log a StackingEnsemble."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp_name = experiment_name or _CFG["mlflow"]["experiment_name"]
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=_CFG["mlflow"]["run_name_prefix"]):
        mlflow.log_params({"n_trials": N_TRIALS, "cv_folds": CV_FOLDS, "n_train": len(X_train)})

        ensemble = StackingEnsemble(n_trials=N_TRIALS, cv_folds=CV_FOLDS)
        ensemble.fit(X_train, y_train)

        for name, params in ensemble.best_params_.items():
            mlflow.log_params({f"{name}__{k}": v for k, v in params.items()})

        metrics = ensemble.evaluate(X_val, y_val)
        mlflow.log_metrics(metrics)

        if model_save_path:
            model_save_path = Path(model_save_path)
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            ensemble.save(model_save_path)
            mlflow.log_artifact(str(model_save_path))

    return ensemble
