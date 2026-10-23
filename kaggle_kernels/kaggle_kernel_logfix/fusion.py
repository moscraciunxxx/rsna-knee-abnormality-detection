"""Per-finding rank fusion for scored kernels.

CNN-tri (55792478 / 0.917) equal-weighted a weaker 12-head overlay at 0.50 on
every finding. V41 (55782774 / 0.936) overlayed only Medial OA / Synovitis /
Contusion and tied the V10 mill. This helper is the shipped correction: keep
the V10 ranking unless a finding is explicitly weighted, and refuse the
uniform all-12 0.50 pattern.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

try:
    from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS
except ImportError:  # Kaggle kernel copy of this file, no package
    ID_COLUMN = "StudyInstanceUID"
    FINDINGS = (
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
    )
    SUBMISSION_COLUMNS = (ID_COLUMN, *FINDINGS)

# V41 log: these three were overlayed at 0.50 and the publicScore stayed 0.936.
V41_SATURATED_FINDINGS: tuple[str, ...] = ("Medial OA", "Synovitis", "Contusion")

# H&S frontier pool minus V41's saturated three. Log-correct overlay targets.
LOGFIX_OVERLAY_FINDINGS: tuple[str, ...] = (
    "ACL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Baker's",
    "Fracture",
)
LOGFIX_OVERLAY_WEIGHT = 0.40


class FusionError(ValueError):
    """Illegal fusion — including the CNN-tri 0.917 uniform 0.50 all-12 blend."""


def overlay_weights_from_findings(
    findings: list[str] | tuple[str, ...],
    weight: float,
) -> dict[str, float]:
    out = {name: 0.0 for name in FINDINGS}
    for name in findings:
        if name not in out:
            raise FusionError(f"unknown finding {name!r}")
        out[name] = float(weight)
    return out


def is_uniform_all_finding_overlay(
    weights: Mapping[str, float],
    *,
    min_weight: float = 0.5,
) -> bool:
    """True when every finding gets the same overlay weight at or above min_weight."""
    vals = [float(weights.get(name, 0.0)) for name in FINDINGS]
    if not vals:
        return False
    if any(abs(v - vals[0]) > 1e-9 for v in vals):
        return False
    return vals[0] >= min_weight


def per_finding_rank_blend(
    anchor: pd.DataFrame,
    overlay: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Rank-blend overlay into anchor only on findings with weight > 0.

    Raises FusionError on the CNN-tri pattern (uniform weight >= 0.50 on all 12).
    Findings with weight 0 keep the anchor column unchanged.
    """
    if list(anchor.columns) != list(SUBMISSION_COLUMNS):
        raise FusionError("anchor columns must match the official header")
    if list(overlay.columns) != list(SUBMISSION_COLUMNS):
        raise FusionError("overlay columns must match the official header")
    if anchor[ID_COLUMN].tolist() != overlay[ID_COLUMN].tolist():
        raise FusionError("anchor and overlay study order must match")
    if is_uniform_all_finding_overlay(weights, min_weight=0.5):
        raise FusionError(
            "uniform overlay weight >= 0.50 on all 12 findings is the CNN-tri 0.917 pattern"
        )

    out = anchor.copy()
    for name in FINDINGS:
        w = float(weights.get(name, 0.0))
        if w <= 0.0:
            continue
        if w > 1.0:
            raise FusionError(f"overlay weight for {name!r} must be in (0, 1]")
        mixed = (1.0 - w) * anchor[name].rank(method="average", pct=True) + w * overlay[
            name
        ].rank(method="average", pct=True)
        out[name] = mixed.rank(method="average", pct=True)

    values = out.drop(columns=[ID_COLUMN]).to_numpy(np.float64)
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise FusionError("fusion produced values outside [0, 1]")
    return out[list(SUBMISSION_COLUMNS)]
