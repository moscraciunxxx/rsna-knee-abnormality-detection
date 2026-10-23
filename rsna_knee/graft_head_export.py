"""Fit/holdout split and grafted loss matrices for the frozen head.

Importing this module does not train, does not read Report into the outputs,
and does not write submission.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.schema import FINDINGS, ID_COLUMN
from rsna_knee.soft_labels import graft_fit_matrices
from rsna_knee.v4_split import llm_soft_holdout_indices, make_steven_v4_split


def graft_head_splits(
    train: pd.DataFrame,
    v4: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Non-gold fit ids, mill-style holdout ids, and expert gold ids.

    The three sets are disjoint. Gold never enters the fit.
    """
    split = make_steven_v4_split(train, v4)
    fit_idx, hold_idx = llm_soft_holdout_indices(split.n_train, frac=0.2, seed=2026)
    fit_ids = tuple(str(split.train_ids[int(i)]) for i in fit_idx.tolist())
    hold_ids = tuple(str(split.train_ids[int(i)]) for i in hold_idx.tolist())
    gold_ids = tuple(str(uid) for uid in split.val_ids)
    if set(fit_ids) & set(gold_ids):
        raise ValueError("gold UID leaked into graft head fit ids")
    if set(hold_ids) & set(gold_ids):
        raise ValueError("gold UID leaked into graft head holdout ids")
    if set(fit_ids) & set(hold_ids):
        raise ValueError("holdout leaked into graft head fit ids")
    if len(fit_ids) + len(hold_ids) != split.n_train:
        raise ValueError("graft head split does not cover the non-gold rows")
    return fit_ids, hold_ids, gold_ids


def graft_head_frames(
    v4: pd.DataFrame,
    pilkwang: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Targets and weights for the fit ids only. Holdout ids are returned, not fit.

    Columns are the official finding header. Report is not a column.
    """
    fit_ids, hold_ids, _gold_ids = graft_head_splits(train, v4)
    y, w = graft_fit_matrices(v4, pilkwang, train, fit_ids)
    if y.shape != (len(fit_ids), len(FINDINGS)) or w.shape != y.shape:
        raise ValueError("graft head frame shape drifted")
    targets = pd.DataFrame(y, columns=list(FINDINGS))
    targets.insert(0, ID_COLUMN, list(fit_ids))
    weights = pd.DataFrame(w, columns=list(FINDINGS))
    weights.insert(0, ID_COLUMN, list(fit_ids))
    if "Report" in targets.columns or "Report" in weights.columns:
        raise ValueError("graft head frames must not carry Report")
    return targets, weights, hold_ids
