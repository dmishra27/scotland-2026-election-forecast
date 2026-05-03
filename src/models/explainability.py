"""
SHAP-based explainability for the Scotland 2026 stacking ensemble.

Uses TreeExplainer for base-learner SHAP values and a weighted average
across base learners to produce an ensemble-level explanation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", message=".*SHAP.*")


def compute_shap_values(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    max_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for the stacking ensemble.

    Falls back to KernelExplainer if TreeExplainer is unavailable for a
    given base learner.  Returns mean absolute SHAP values aggregated over
    classes.

    Parameters
    ----------
    model:
        Fitted StackingEnsemble.
    X:
        Feature matrix (post-transform).
    max_samples:
        Cap on number of background/foreground samples to bound compute time.

    Returns
    -------
    (shap_values, base_values)
        shap_values: shape (n_samples, n_features), mean |SHAP| over classes
        base_values: shape (n_classes,) expected model output
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    n = min(max_samples, len(X_arr))
    idx = np.random.default_rng(42).choice(len(X_arr), size=n, replace=False)
    X_sample = X_arr[idx]

    all_shap: list[np.ndarray] = []

    for name, estimator in model.model_.named_estimators_.items():
        try:
            explainer = shap.TreeExplainer(estimator)
            sv = explainer.shap_values(X_sample)
            # sv is list[array] for multi-class tree models
            if isinstance(sv, list):
                sv = np.stack(sv, axis=-1)  # (n, features, classes)
            else:
                sv = sv[:, :, np.newaxis]
            # mean |SHAP| over classes
            all_shap.append(np.abs(sv).mean(axis=-1))
        except Exception:
            # KernelExplainer fallback (slow, limited samples)
            background = shap.kmeans(X_sample, min(10, n))
            explainer = shap.KernelExplainer(estimator.predict_proba, background)
            sv = explainer.shap_values(X_sample[:50])
            if isinstance(sv, list):
                sv = np.stack(sv, axis=-1)
            all_shap.append(np.abs(sv).mean(axis=-1))

    shap_values = np.mean(all_shap, axis=0)  # average over base learners
    base_values = model.predict_proba(X_sample).mean(axis=0)

    return shap_values, base_values


def get_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return a DataFrame of mean absolute SHAP feature importances.

    Parameters
    ----------
    shap_values:
        Output from compute_shap_values — shape (n_samples, n_features).
    feature_names:
        Feature names aligned with shap_values columns.
    top_n:
        Number of top features to return.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.nlargest(top_n, "mean_abs_shap").reset_index(drop=True)
