"""Macro-averaged ROC AUC over the 12 findings (host score)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def per_finding_roc_auc(y_true, y_score) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_score must be 2D arrays of the same shape")
    aucs = np.empty(y_true.shape[1], dtype=np.float64)
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            aucs[i] = np.nan
        else:
            aucs[i] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
    return aucs


def macro_roc_auc(y_true, y_score) -> float:
    """Mean of per-finding ROC AUC — the competition metric.

    Findings that are single-class in a split are skipped (nanmean), matching
    the public 2.5D ResNet34 baseline's compute_macro_auc guard.
    """
    aucs = per_finding_roc_auc(y_true, y_score)
    if np.isnan(aucs).all():
        return 0.5
    return float(np.nanmean(aucs))
