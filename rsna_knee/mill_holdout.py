"""Mill V10 encodings of the seed-2026 LLM-soft holdout (~870). Not gold-58 folklore.

Join is fail-closed when mill preds for those UIDs are missing. Overlay license
is computed on that table only; gold-58 Δ is not a substitute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rsna_knee.fusion import (
    FusionError,
    is_ruled_out_dino3s_replace_family,
    is_ruled_out_jointline_mill_radimagenet_family,
    is_ruled_out_public_native384_family,
    is_ruled_out_r336ov_family,
    is_uniform_all_finding_overlay,
    overlay_finding_set,
    overlay_weights_from_findings,
)
from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS
from rsna_knee.v4_split import llm_soft_holdout_indices, make_steven_v4_split

HOLD_SEED = 2026
HOLD_FRAC = 0.2
OVERLAY_MAX_WEIGHT = 0.15
OVERLAY_MAX_FINDINGS = 2
OVERLAY_MIN_DELTA = 0.01
MISSING_MILL_BANNER = (
    "MILL V10 PREDS: MISSING — cannot compute mill-vs-holdout AUC"
)


class MillHoldoutMissing(RuntimeError):
    """Mill encodings of the 870 holdout IDs are absent. Do not overlay."""


@dataclass(frozen=True)
class MillHoldoutJoin:
    ids: tuple[str, ...]
    mill: np.ndarray
    y_bin: np.ndarray
    per_auc: np.ndarray
    macro: float


def holdout_ids_and_y(
    train: pd.DataFrame,
    v4: pd.DataFrame,
    *,
    frac: float = HOLD_FRAC,
    seed: int = HOLD_SEED,
) -> tuple[tuple[str, ...], np.ndarray]:
    """LLM-soft holdout UIDs and binarized v4 labels. Gold 58 never included."""
    split = make_steven_v4_split(train, v4)
    fit, holdout = llm_soft_holdout_indices(split.n_train, frac=frac, seed=seed)
    ids = tuple(split.train_ids[int(i)] for i in holdout.tolist())
    if set(ids) & set(split.val_ids):
        raise ValueError("LLM-soft holdout overlapped gold 58")
    y = (split.y_train[holdout] >= 0.5).astype(np.float64)
    return ids, y


def join_mill_holdout(
    holdout_ids: tuple[str, ...] | list[str],
    mill_frame: pd.DataFrame | None,
    y_holdout: np.ndarray,
) -> MillHoldoutJoin:
    """Align mill preds to holdout UIDs. Fail-closed if any UID is missing."""
    ids = tuple(str(x) for x in holdout_ids)
    if mill_frame is None or len(mill_frame) == 0:
        raise MillHoldoutMissing(MISSING_MILL_BANNER)
    missing_cols = [c for c in (ID_COLUMN, *FINDINGS) if c not in mill_frame.columns]
    if missing_cols:
        raise MillHoldoutMissing(f"{MISSING_MILL_BANNER}; mill table missing {missing_cols}")
    mill = mill_frame.copy()
    mill[ID_COLUMN] = mill[ID_COLUMN].astype(str)
    mill = mill.drop_duplicates(ID_COLUMN)
    have = set(mill[ID_COLUMN])
    missing = [i for i in ids if i not in have]
    if missing:
        raise MillHoldoutMissing(
            f"{MISSING_MILL_BANNER}; missing {len(missing)}/{len(ids)} holdout UIDs"
        )
    mill = mill.set_index(ID_COLUMN).loc[list(ids)]
    mill_s = mill[list(FINDINGS)].to_numpy(dtype=np.float64)
    y = np.asarray(y_holdout, dtype=np.float64)
    if y.shape != mill_s.shape:
        raise ValueError(f"y_holdout shape {y.shape} != mill shape {mill_s.shape}")
    per = per_finding_roc_auc(y, mill_s)
    macro = macro_roc_auc(y, mill_s)
    return MillHoldoutJoin(ids=ids, mill=mill_s, y_bin=y, per_auc=per, macro=macro)


def licensed_overlay_weights(
    mill_per: np.ndarray | dict[str, float],
    new_per: np.ndarray | dict[str, float],
    *,
    min_delta: float = OVERLAY_MIN_DELTA,
    max_findings: int = OVERLAY_MAX_FINDINGS,
    weight: float = OVERLAY_MAX_WEIGHT,
) -> dict[str, float]:
    """Subset overlay ≪ 0.50 only where new beats mill on the 870. Else all zeros.

    Refuses CNN-tri all-12, r336ov PF OA+Synovitis set, V41/logfix, dino3s, jointline.
    """
    if isinstance(mill_per, dict):
        mill_v = np.array([float(mill_per[f]) for f in FINDINGS], dtype=np.float64)
    else:
        mill_v = np.asarray(mill_per, dtype=np.float64)
    if isinstance(new_per, dict):
        new_v = np.array([float(new_per[f]) for f in FINDINGS], dtype=np.float64)
    else:
        new_v = np.asarray(new_per, dtype=np.float64)
    if mill_v.shape != (len(FINDINGS),) or new_v.shape != (len(FINDINGS),):
        raise ValueError("per-finding vectors must be length 12")
    deltas = new_v - mill_v
    order = np.argsort(-deltas)
    wins: list[str] = []
    for i in order:
        if len(wins) >= int(max_findings):
            break
        if not np.isfinite(deltas[i]):
            continue
        if float(deltas[i]) <= float(min_delta):
            continue
        wins.append(FINDINGS[int(i)])
    zeros = {name: 0.0 for name in FINDINGS}
    if not wins:
        return zeros
    w = overlay_weights_from_findings(wins, float(weight))
    selected = overlay_finding_set(w)
    if is_uniform_all_finding_overlay(w, min_weight=0.5):
        raise FusionError("uniform overlay weight >= 0.50 on all 12 findings is the CNN-tri 0.917 pattern")
    if is_ruled_out_r336ov_family(selected):
        raise FusionError("r336ov PF OA+Synovitis family is ruled out")
    if is_ruled_out_public_native384_family(selected):
        raise FusionError("native384 V41/logfix family is ruled out")
    if is_ruled_out_dino3s_replace_family(selected):
        raise FusionError("dino3s MCL+Baker's replace family is ruled out")
    if is_ruled_out_jointline_mill_radimagenet_family(selected):
        raise FusionError("jointline mill-RadImageNet family is ruled out")
    if float(weight) >= 0.50:
        raise FusionError("overlay weight must be ≪ 0.50")
    if len(selected) > int(max_findings):
        raise FusionError("overlay at most 1–2 findings")
    return w


def decide_private_overlay(
    mill_per: np.ndarray,
    new_per: np.ndarray,
) -> tuple[str, dict[str, float] | None]:
    """Private-first overlay policy on mill-870 AUCs, not gold-58 and not public 0.936.

    Returns (decision, weights_or_none):
    - ``stop``: new < mill on every finding — keep mill, no Submit
    - ``overlay``: licensed subset ≪ 0.50, not a ruled-out family
    - ``hold``: no legal overlay; keep mill, no Submit
    """
    if vitb_loses_every_finding(mill_per, new_per):
        return "stop", None
    try:
        weights = licensed_overlay_weights(mill_per, new_per)
    except FusionError:
        return "hold", None
    if not any(float(v) > 0.0 for v in weights.values()):
        return "hold", None
    return "overlay", weights


def vitb_loses_every_finding(
    mill_per: np.ndarray, new_per: np.ndarray
) -> bool:
    """True when the new head is strictly below mill on every finite finding."""
    mill_v = np.asarray(mill_per, dtype=np.float64).reshape(-1)
    new_v = np.asarray(new_per, dtype=np.float64).reshape(-1)
    if mill_v.shape != new_v.shape or mill_v.size != len(FINDINGS):
        raise ValueError("per-finding vectors must be length 12")
    ok = np.isfinite(mill_v) & np.isfinite(new_v)
    if not bool(ok.any()):
        return False
    return bool(np.all(new_v[ok] < mill_v[ok]))


def mill_holdout_table(ids: tuple[str, ...], mill_s: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(mill_s, columns=list(FINDINGS))
    frame.insert(0, ID_COLUMN, list(ids))
    return frame[list(SUBMISSION_COLUMNS)]
