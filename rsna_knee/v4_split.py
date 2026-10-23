"""Steven v4 train split: expert gold 58 is val-only and never in the loss.

Train targets come from stevenleehans `llm_labels_v4_blend.csv`. Val labels come
from the 58 expert rows in `train.csv`. The one gold UID missing from some LLM
tables still stays val-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rsna_knee.schema import FINDINGS, ID_COLUMN

GOLD_AUC_ABORT_THRESHOLD = 0.94


class GoldOverfitAbort(RuntimeError):
    """Gold-58 macro AUC is ≫ 0.94 — leak/overfit alarm. Do not overlay."""


@dataclass(frozen=True)
class StevenV4Split:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    y_train: np.ndarray
    y_val: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_ids)

    @property
    def n_val(self) -> int:
        return len(self.val_ids)


def expert_gold_ids(train: pd.DataFrame) -> tuple[str, ...]:
    missing = [c for c in (ID_COLUMN, *FINDINGS) if c not in train.columns]
    if missing:
        raise ValueError(f"train table missing columns {missing}")
    frame = train.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    mask = frame[list(FINDINGS)].notna().all(axis=1)
    ids = tuple(frame.loc[mask, ID_COLUMN].tolist())
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate gold StudyInstanceUID")
    return ids


def should_abort_gold_overfit(
    gold_macro: float, threshold: float = GOLD_AUC_ABORT_THRESHOLD
) -> bool:
    """True when gold macro AUC is strictly greater than 0.94 (≫ 0.94 alarm)."""
    return float(gold_macro) > float(threshold)


def abort_if_gold_overfit(
    gold_macro: float, threshold: float = GOLD_AUC_ABORT_THRESHOLD
) -> None:
    if should_abort_gold_overfit(gold_macro, threshold=threshold):
        raise GoldOverfitAbort(
            f"gold macro AUC {float(gold_macro):.4f} > {float(threshold):.2f} "
            "(leak/overfit alarm; gold 58 must stay val-only; do not overlay)"
        )


# A new frozen head must beat this mill-870 macro before any overlay.
# probe-b's measured holdout was 0.7453 and that run was already a stop.
HOLDOUT_MACRO_STOP = 0.745


def frozen_head_gate(gold_macro: float, holdout_macro: float) -> str:
    """abort, stop, or continue for a frozen head. Never an overlay by itself.

    Gold macro strictly above 0.94 aborts. A mill-870 holdout at or below
    0.745 stops. Non-finite inputs stop. Continue means the private numbers
    cleared both gates and a later overlay law still applies.
    """
    gold = float(gold_macro)
    hold = float(holdout_macro)
    if not np.isfinite(gold) or not np.isfinite(hold):
        return "stop"
    if should_abort_gold_overfit(gold):
        return "abort"
    if hold <= HOLDOUT_MACRO_STOP:
        return "stop"
    return "continue"


def make_steven_v4_split(
    train: pd.DataFrame,
    v4: pd.DataFrame,
    *,
    gold_ids: tuple[str, ...] | None = None,
) -> StevenV4Split:
    """Train = v4 rows whose UID is not gold. Val = all expert gold 58.

    Val y is the expert 0/1 matrix from train.csv, not the LLM scores.
    Train y is the steven v4 soft scores. Gold UIDs never appear in train_ids.
    """
    gold = expert_gold_ids(train) if gold_ids is None else tuple(str(x) for x in gold_ids)
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("no expert gold rows")

    tr = train.copy()
    tr[ID_COLUMN] = tr[ID_COLUMN].astype(str)
    gold_frame = tr[tr[ID_COLUMN].isin(gold_set)].drop_duplicates(ID_COLUMN)
    if len(gold_frame) != len(gold_set):
        raise ValueError("gold UIDs missing from train.csv")
    gold_frame = gold_frame.set_index(ID_COLUMN).loc[list(gold)]
    y_val = gold_frame[list(FINDINGS)].to_numpy(dtype=np.float64)

    labels = v4.copy()
    if ID_COLUMN not in labels.columns:
        raise ValueError("steven v4 table missing StudyInstanceUID")
    missing_f = [c for c in FINDINGS if c not in labels.columns]
    if missing_f:
        raise ValueError(f"steven v4 missing findings {missing_f}")
    labels[ID_COLUMN] = labels[ID_COLUMN].astype(str)
    labels = labels.drop_duplicates(ID_COLUMN)
    train_frame = labels[~labels[ID_COLUMN].isin(gold_set)].copy()
    if train_frame[ID_COLUMN].isin(gold_set).any():
        raise ValueError("gold UID leaked into steven v4 train split")
    train_ids = tuple(train_frame[ID_COLUMN].tolist())
    y_train = train_frame[list(FINDINGS)].to_numpy(dtype=np.float64)
    if set(train_ids) & gold_set:
        raise ValueError("gold 58 must never be in the train loss")
    return StevenV4Split(
        train_ids=train_ids,
        val_ids=gold,
        y_train=y_train,
        y_val=y_val,
    )


def llm_soft_holdout_indices(
    n: int, *, frac: float = 0.2, seed: int = 2026
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic fit/holdout indices over non-gold steven v4 rows.

    Gold 58 is already excluded from `n`. Holdout y is LLM-soft (not expert gold).
    """
    if n < 2:
        raise ValueError("need at least 2 non-gold rows for an LLM-soft holdout")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_ho = max(1, min(int(n) - 1, int(round(int(n) * float(frac)))))
    holdout = np.sort(perm[:n_ho])
    fit = np.sort(perm[n_ho:])
    if set(fit.tolist()) & set(holdout.tolist()):
        raise ValueError("LLM-soft holdout leaked into fit")
    return fit, holdout
