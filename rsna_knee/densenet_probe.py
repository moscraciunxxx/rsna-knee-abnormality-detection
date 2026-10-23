"""8-study timing math for a frozen RadImageNet DenseNet121 encode.

The full 4349-study train is not started here. Gold expert rows are excluded
from the probe sample so they cannot enter a loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.schema import FINDINGS, ID_COLUMN
from rsna_knee.v4_split import expert_gold_ids

PROBE_STUDIES = 8
IMAGES_PER_STUDY = 18
N_ENCODE_STUDIES = 4407
ENCODE_ABORT_S = 2.8 * 3600.0


def probe_study_ids(
    train: pd.DataFrame,
    *,
    n: int = PROBE_STUDIES,
    seed: int = 2026,
) -> tuple[str, ...]:
    """Non-gold study IDs. The returned set is disjoint from the expert 58."""
    gold = set(expert_gold_ids(train))
    frame = train.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    pool = [uid for uid in frame[ID_COLUMN].tolist() if uid not in gold]
    if len(pool) < int(n):
        raise ValueError(f"need {n} non-gold studies, have {len(pool)}")
    rng = np.random.default_rng(int(seed))
    chosen_at = np.sort(rng.choice(len(pool), size=int(n), replace=False))
    chosen = tuple(pool[int(i)] for i in chosen_at.tolist())
    if set(chosen) & gold:
        raise ValueError("gold UID leaked into the DenseNet probe sample")
    return chosen


def projected_encode_seconds(
    elapsed_s: float,
    n_images: int,
    *,
    n_studies: int = N_ENCODE_STUDIES,
    images_per_study: int = IMAGES_PER_STUDY,
) -> float:
    """Scale a measured image batch onto the full encode set."""
    elapsed = float(elapsed_s)
    images = int(n_images)
    if elapsed <= 0.0 or images <= 0:
        raise ValueError("elapsed time and image count must be positive")
    return elapsed * (int(n_studies) * int(images_per_study)) / images


def encode_probe_aborts(
    elapsed_s: float,
    n_images: int,
    *,
    limit_s: float = ENCODE_ABORT_S,
) -> bool:
    """True when the projected full encode is strictly above 2.8 h."""
    return projected_encode_seconds(elapsed_s, n_images) > float(limit_s)
