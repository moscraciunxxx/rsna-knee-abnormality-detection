"""Build a kernels-only submission table with clipped probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS


def clip_probabilities(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    return np.clip(arr, 0.0, 1.0)


def build_submission(study_ids, probabilities) -> pd.DataFrame:
    ids = [str(x) for x in list(study_ids)]
    probs = clip_probabilities(np.asarray(probabilities, dtype=np.float64))
    if probs.ndim != 2 or probs.shape[1] != len(FINDINGS):
        raise ValueError(
            f"probabilities must be (n_studies, {len(FINDINGS)}), got {probs.shape}"
        )
    if len(ids) != probs.shape[0]:
        raise ValueError("study_ids length must match probability rows")
    data = {ID_COLUMN: ids}
    for j, name in enumerate(FINDINGS):
        data[name] = probs[:, j]
    return pd.DataFrame(data, columns=list(SUBMISSION_COLUMNS))


def write_submission(table: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.loc[:, list(SUBMISSION_COLUMNS)]
    ordered.to_csv(out, index=False)
    return out
