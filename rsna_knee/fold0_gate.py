"""Score a fold-0 OOF file. Mill can kill the recipe only on the same studies.

``data/mill_v10_holdout870.csv`` covers 174 of the 882 fold-0 studies. That
intersection is not the fold. A kill requires both models on every scored
fold-0 study. Soft labels at exactly 0.5 are silence and stay out of the AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.schema import FINDINGS, ID_COLUMN


def _aligned(pred: pd.DataFrame, labels: pd.DataFrame, ids: set[str] | None) -> tuple[np.ndarray, np.ndarray]:
    left = pred.copy()
    left[ID_COLUMN] = left[ID_COLUMN].astype(str)
    right = labels.copy()
    right[ID_COLUMN] = right[ID_COLUMN].astype(str)
    if ids is not None:
        left = left[left[ID_COLUMN].isin(ids)]
    merged = left[[ID_COLUMN, *FINDINGS]].merge(
        right[[ID_COLUMN, *FINDINGS]], on=ID_COLUMN, suffixes=("_p", "_y")
    )
    if merged.empty:
        raise ValueError("no shared studies")
    y = np.empty((len(merged), len(FINDINGS)), np.float64)
    s = np.empty_like(y)
    for i, name in enumerate(FINDINGS):
        raw = pd.to_numeric(merged[f"{name}_y"], errors="coerce").to_numpy(np.float64)
        score = pd.to_numeric(merged[f"{name}_p"], errors="coerce").to_numpy(np.float64)
        known = np.isfinite(raw) & np.isfinite(score) & (raw != 0.5)
        y[:, i] = np.where(known, (raw > 0.5).astype(np.float64), np.nan)
        s[:, i] = np.where(known, score, np.nan)
    return y, s


def macro_on_known(y: np.ndarray, score: np.ndarray) -> tuple[float, int]:
    """AUC on finite pairs only. A 0.5 label was already set to nan by the aligner."""
    from sklearn.metrics import roc_auc_score

    if not np.isfinite(y).any():
        raise ValueError("no known labels")
    aucs = []
    for i in range(y.shape[1]):
        mask = np.isfinite(y[:, i]) & np.isfinite(score[:, i])
        if mask.sum() < 2 or len(np.unique(y[mask, i])) < 2:
            aucs.append(np.nan)
            continue
        aucs.append(float(roc_auc_score(y[mask, i], score[mask, i])))
    vals = np.asarray(aucs, dtype=np.float64)
    if np.isnan(vals).all():
        macro = 0.5
    else:
        macro = float(np.nanmean(vals))
    return macro, int(np.isfinite(y).any(axis=1).sum())


def fold0_decision(
    oof: pd.DataFrame,
    labels: pd.DataFrame,
    mill: pd.DataFrame | None,
    fold_ids: set[str],
) -> dict:
    """Compare OOF to labels. Mill kills only when it covers the same rows."""
    y, s = _aligned(oof, labels, fold_ids)
    ours, n = macro_on_known(y, s)
    out = {"oof_macro": ours, "n": n, "fold_n": len(fold_ids), "decision": "incomplete"}
    if mill is None:
        out["reason"] = "no mill file"
        return out
    mill_ids = set(mill[ID_COLUMN].astype(str))
    covered = fold_ids & mill_ids & set(oof[ID_COLUMN].astype(str))
    if covered != fold_ids:
        out["mill_n"] = len(covered)
        out["reason"] = "mill does not cover this fold"
        return out
    y_m, s_m = _aligned(mill, labels, fold_ids)
    mill_macro, n_m = macro_on_known(y_m, s_m)
    out["mill_macro"] = mill_macro
    out["mill_n"] = n_m
    if n_m != n:
        out["reason"] = "row counts differ"
        return out
    out["decision"] = "kill" if ours < mill_macro else "keep"
    out["reason"] = "same studies"
    return out
