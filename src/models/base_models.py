"""
Individual Optuna HPM trainers for each base learner in the stacking ensemble.

Each trainer function returns (fitted_clf, best_params, val_f1_macro).
CatBoost and RandomForest also wrap with isotonic probability calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CFG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

N_TRIALS: int = _CFG["model"]["optuna_trials"]
CV_FOLDS: int = _CFG["model"]["cv_folds"]
_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def _encode(y_train: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode string labels to integers. Returns (y_train_enc, y_val_enc, le)."""
    le = LabelEncoder()
    return le.fit_transform(y_train), le.transform(y_val), le


def _oof_f1(clf, X: np.ndarray, y: np.ndarray) -> float:
    oof = cross_val_predict(clf, X, y, cv=_CV)
    return float(f1_score(y, oof, average="macro", zero_division=0))


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_trials: int = N_TRIALS,
) -> tuple[Any, dict, float]:
    y_tr, y_vl, le = _encode(y_train, y_val)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "n_jobs": -1,
        }
        clf = XGBClassifier(**params)
        return -_oof_f1(clf, X_train, y_tr)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    clf = XGBClassifier(**best, eval_metric="mlogloss", verbosity=0, n_jobs=-1)
    clf.fit(X_train, y_tr)
    val_f1 = f1_score(y_vl, clf.predict(X_val), average="macro", zero_division=0)
    return clf, best, float(val_f1)


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_trials: int = N_TRIALS,
) -> tuple[Any, dict, float]:
    y_tr, y_vl, le = _encode(y_train, y_val)

    def objective(trial: optuna.Trial) -> float:
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
        clf = LGBMClassifier(**params)
        return -_oof_f1(clf, X_train, y_tr)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    clf = LGBMClassifier(**best, verbose=-1, n_jobs=-1)
    clf.fit(X_train, y_tr)
    val_f1 = f1_score(y_vl, clf.predict(X_val), average="macro", zero_division=0)
    return clf, best, float(val_f1)


# ── CatBoost ──────────────────────────────────────────────────────────────────

def train_catboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_trials: int = N_TRIALS,
    calibrate: bool = True,
) -> tuple[Any, dict, float]:
    y_tr, y_vl, le = _encode(y_train, y_val)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "verbose": False,
            "allow_writing_files": False,
        }
        clf = CatBoostClassifier(**params)
        return -_oof_f1(clf, X_train, y_tr)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    base_clf = CatBoostClassifier(**best, verbose=False, allow_writing_files=False)
    if calibrate:
        clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    else:
        clf = base_clf
    clf.fit(X_train, y_tr)
    val_f1 = f1_score(y_vl, clf.predict(X_val), average="macro", zero_division=0)
    return clf, best, float(val_f1)


# ── RandomForest + isotonic calibration ──────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_trials: int = N_TRIALS,
    calibrate: bool = True,
) -> tuple[Any, dict, float]:
    y_tr, y_vl, le = _encode(y_train, y_val)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params, random_state=42)
        return -_oof_f1(clf, X_train, y_tr)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    base_clf = RandomForestClassifier(**best, random_state=42, n_jobs=-1)
    if calibrate:
        clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    else:
        clf = base_clf
    clf.fit(X_train, y_tr)
    val_f1 = f1_score(y_vl, clf.predict(X_val), average="macro", zero_division=0)
    return clf, best, float(val_f1)


# ── Logistic Regression (meta-learner / standalone) ──────────────────────────

def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_trials: int = N_TRIALS,
) -> tuple[Any, dict, float]:
    y_tr, y_vl, le = _encode(y_train, y_val)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 100.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        clf = LogisticRegression(C=C, solver=solver, max_iter=1000)
        return -_oof_f1(clf, X_train, y_tr)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=min(n_trials, 30), show_progress_bar=False)
    best = study.best_params
    clf = LogisticRegression(**best, max_iter=1000)
    clf.fit(X_train, y_tr)
    val_f1 = f1_score(y_vl, clf.predict(X_val), average="macro", zero_division=0)
    return clf, best, float(val_f1)
