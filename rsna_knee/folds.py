"""Study-grouped folds for OOF. StudyInstanceUID is not a model feature."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.schema import ID_COLUMN
from rsna_knee.v4_split import expert_gold_ids

FOLD_SEED = 2026
N_FOLDS = 5


def assign_study_folds(
    train: pd.DataFrame,
    *,
    n_folds: int = N_FOLDS,
    seed: int = FOLD_SEED,
) -> pd.DataFrame:
    """One fold id per study. Gold rows are flagged and stay out of the loss.

    Sizes differ by at most one. The same seed gives the same assignment.
    """
    if ID_COLUMN not in train.columns:
        raise ValueError("train table missing StudyInstanceUID")
    if n_folds < 2:
        raise ValueError("need at least 2 folds")
    ids = train[ID_COLUMN].astype(str)
    if ids.duplicated().any():
        raise ValueError("duplicate StudyInstanceUID")
    gold = set(expert_gold_ids(train))
    order = ids.tolist()
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(order))
    fold = np.empty(len(order), dtype=np.int64)
    for rank, idx in enumerate(perm.tolist()):
        fold[int(idx)] = int(rank % n_folds)
    out = pd.DataFrame(
        {
            ID_COLUMN: order,
            "fold": fold,
            "is_gold": [1 if uid in gold else 0 for uid in order],
        }
    )
    if set(out.loc[out["is_gold"] == 1, ID_COLUMN]) != gold:
        raise ValueError("gold flag does not match expert gold ids")
    return out
