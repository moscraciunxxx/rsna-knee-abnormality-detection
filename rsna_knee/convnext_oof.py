"""Author ConvNeXt v0 fold ids. Gold stays at fold -1 and is never a train row.

The published reader ships only ``cnxt_v0_fold0.pt`` .. ``fold2.pt``. A study is
out of fold only when it is scored by the checkpoint that held that fold out.
Folds 3 and 4 were inside every published checkpoint, so they are not OOF.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
ID_COL = "StudyInstanceUID"
OOF_FOLDS = (0, 1, 2)
N_FOLDS = 5
FOLD_SEED = 0


def report_group(text: object) -> str:
    """Same report hash the author's make_labels.py uses for GroupKFold."""
    return re.sub(r"\W+", "", str(text).lower())


def author_v0_folds(train: pd.DataFrame) -> pd.Series:
    """Fold id per study, row-aligned to ``make_labels.py`` v0.

    Gold studies (all 12 official labels present) get -1. Weak studies get
    0..4 from GroupKFold(5) on the report hash, RandomState(0).
    """
    missing = [c for c in (ID_COL, "Report", *LABELS) if c not in train.columns]
    if missing:
        raise ValueError(f"train.csv missing {missing}")
    work = train.reset_index(drop=True)
    gold_mask = work[list(LABELS)].notna().all(axis=1).to_numpy()
    folds = np.full(len(work), -1, dtype=int)
    weak_idx = np.flatnonzero(~gold_mask)
    groups = work.loc[~gold_mask, "Report"].map(report_group).to_numpy()
    order = np.random.RandomState(FOLD_SEED).permutation(int(weak_idx.size))
    assigned = np.zeros(int(weak_idx.size), dtype=int)
    for k, (_, va) in enumerate(GroupKFold(N_FOLDS).split(order, groups=groups[order])):
        assigned[order[va]] = k
    folds[weak_idx] = assigned
    return pd.Series(folds, index=work[ID_COL].astype(str), name="fold")


def oof_ids_by_fold(
    folds: pd.Series, holdout_ids: Sequence[str]
) -> dict[int, tuple[str, ...]]:
    """Holdout studies that folds 0, 1, or 2 can score without having trained on them."""
    by_fold: dict[int, tuple[str, ...]] = {}
    for fold_id in OOF_FOLDS:
        chosen: list[str] = []
        for uid in holdout_ids:
            key = str(uid)
            if key not in folds.index:
                raise KeyError(f"holdout id missing from author folds: {key}")
            got = int(folds.loc[key])
            if got == -1:
                raise ValueError(f"gold study leaked into the OOF holdout: {key}")
            if got == fold_id:
                chosen.append(key)
        by_fold[fold_id] = tuple(chosen)
    return by_fold
