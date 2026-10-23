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

# Spatially tighter RadImageNet FOV than the mill's 130 mm crop. Not native384.
# 55822934 COMPLETE 0.935 — ruled out as this goal's "new family."
JOINTLINE_OVERLAY_FINDINGS: tuple[str, ...] = (
    "Synovitis",
    "Medial Meniscus",
    "Lateral Meniscus",
)
JOINTLINE_OVERLAY_WEIGHT = 0.22
JOINTLINE_CROP_MM = 90.0

# DINOv3 ViT-S/16+ full-finetune (surasan092 v13.1). Not mill RadImageNet, not native384.
# Fold-0 gold: Baker's 0.947, MCL 0.914. Replace those two; leave mill elsewhere.
DINO3S_REPLACE_FINDINGS: tuple[str, ...] = ("MCL", "Baker's")
DINO3S_REPLACE_WEIGHT = 1.0
DINO3S_ENCODER = "dinov3-vits16plus"
DINO3S_CROP_MM = 130.0
DINO3S_IMAGE_SIZE = 336
DINO3S_SLICES_PER_SLOT = 12
DINO3S_WINDOW = 3

# Mill TTA_TARGET_POOL omitted these two (mean windows). Peak-slice max + CoAtNet
# hflip at 0.15. Not 1.0 replace, not mill RadImageNet, not native384, not all-12.
TTAMAX_OVERLAY_FINDINGS: tuple[str, ...] = ("PF OA", "Effusion")
TTAMAX_OVERLAY_WEIGHT = 0.15
TTAMAX_DINO_POOL = "max"

# 58-row private fold vs mill. DINOv2-B r336 is loadable; do not overlay until Δ>0.
GOLDFOLD_ENCODER = "dinov2-vitb14-r336"
GOLDFOLD_N_GOLD = 58
GOLDFOLD_OVERLAY_WEIGHT = 0.0

# Goldfold 58-row mill vs r336 (verified). Overlay only well-powered mill-weak wins.
# Skip Lateral OA (11 pos) and Baker's (12 pos, dino3s 1.0 already burned).
# 55856648 COMPLETE 0.935 — 58-row Δ did not transfer. Ruled out as this family.
R336OV_OVERLAY_FINDINGS: tuple[str, ...] = ("PF OA", "Synovitis")
R336OV_OVERLAY_WEIGHT = 0.20
R336OV_ENCODER = "dinov2-vitb14-r336"

# Pinned sklearn-1.8.0 ConvNeXt OOF, n=517. 1000-draw blended-macro CI lower
# bound cleared +0.003 at weight 0.15 on these seven. Not a TEST submit until
# the mill control reproduces 55779066. Not the 2-finding ACL+Effusion mix.
CONVNEXT_OOF_K7_FINDINGS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Medial OA",
    "Effusion",
    "Synovitis",
    "Baker's",
)
CONVNEXT_OOF_K7_WEIGHT = 0.15
# Same seven findings. Private macro CI at 0.45 is +0.0088 to +0.0123
# versus +0.0035 to +0.0048 at 0.15. 0.50 fail-closes the whole blend.
CONVNEXT_OOF_K7_HI_WEIGHT = 0.45


class FusionError(ValueError):
    """Illegal fusion — including the CNN-tri 0.917 uniform 0.50 all-12 blend."""


def convnext_oof_k7_weights(weight: float = CONVNEXT_OOF_K7_WEIGHT) -> dict[str, float]:
    """Seven-finding ConvNeXt OOF mix. Refuses an all-12 or ruled-out set."""
    if len(CONVNEXT_OOF_K7_FINDINGS) >= len(FINDINGS):
        raise FusionError("k7 overlay must stay a subset of the 12 findings")
    if float(weight) >= 0.50:
        raise FusionError("k7 weight must stay under 0.50")
    weights = overlay_weights_from_findings(CONVNEXT_OOF_K7_FINDINGS, float(weight))
    selected = overlay_finding_set(weights)
    if is_uniform_all_finding_overlay(weights, min_weight=0.5):
        raise FusionError("k7 mix must not be the uniform all-12 pattern")
    if is_ruled_out_r336ov_family(selected, weight=float(weight)):
        raise FusionError("k7 mix must not be the r336ov pair")
    if is_ruled_out_public_native384_family(selected):
        raise FusionError("k7 mix must not be a native384 subset")
    if is_ruled_out_dino3s_replace_family(selected, weight=float(weight)):
        raise FusionError("k7 mix must not be the dino3s replace")
    if is_ruled_out_jointline_mill_radimagenet_family(selected):
        raise FusionError("k7 mix must not be the jointline subset")
    return weights


def convnext_oof_k7_hi_weights() -> dict[str, float]:
    """Higher legal weight on the same seven findings. Not a public submit by itself."""
    return convnext_oof_k7_weights(CONVNEXT_OOF_K7_HI_WEIGHT)


# Clearly positive findings only. Lateral Meniscus (CI lo +0.00019) and PF OA
# (CI lo +0.00061) are left on mill with Lateral OA, Contusion, and Fracture.
# Kept weights are 0.49. Joint macro CI on the 517 is +0.00923 to +0.01304.
CONVNEXT_OOF_UNEQUAL_WEIGHTS: dict[str, float] = {
    "ACL": 0.49,
    "MCL": 0.49,
    "Medial Meniscus": 0.49,
    "Lateral Meniscus": 0.0,
    "Medial OA": 0.49,
    "Lateral OA": 0.0,
    "PF OA": 0.0,
    "Effusion": 0.49,
    "Synovitis": 0.49,
    "Baker's": 0.49,
    "Contusion": 0.0,
    "Fracture": 0.0,
}


def convnext_oof_unequal_weights() -> dict[str, float]:
    """Licensed unequal map. One weight >= 0.50 fail-closes the whole blend."""
    weights = {name: float(CONVNEXT_OOF_UNEQUAL_WEIGHTS[name]) for name in FINDINGS}
    if any(value >= 0.50 for value in weights.values()):
        raise FusionError("unequal overlay weight must stay under 0.50")
    if sum(value > 0.0 for value in weights.values()) >= len(FINDINGS):
        raise FusionError("unequal overlay must stay a subset of the 12 findings")
    selected = overlay_finding_set(weights)
    if is_uniform_all_finding_overlay(weights, min_weight=0.5):
        raise FusionError("unequal overlay must not be the uniform all-12 pattern")
    if is_ruled_out_r336ov_family(selected):
        raise FusionError("unequal overlay must not be the r336ov pair")
    if is_ruled_out_public_native384_family(selected):
        raise FusionError("unequal overlay must not be a native384 subset")
    if is_ruled_out_dino3s_replace_family(selected):
        raise FusionError("unequal overlay must not be the dino3s replace")
    if is_ruled_out_jointline_mill_radimagenet_family(selected):
        raise FusionError("unequal overlay must not be the jointline subset")
    return weights


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


def overlay_finding_set(weights: Mapping[str, float]) -> frozenset[str]:
    return frozenset(name for name in FINDINGS if float(weights.get(name, 0.0)) > 0.0)


def is_ruled_out_public_native384_family(
    findings: list[str] | tuple[str, ...] | frozenset[str],
) -> bool:
    """True for the two public-native384 subsets that already tied 0.936."""
    selected = frozenset(findings)
    return selected == frozenset(V41_SATURATED_FINDINGS) or selected == frozenset(
        LOGFIX_OVERLAY_FINDINGS
    )


def is_ruled_out_dino3s_replace_family(
    findings: list[str] | tuple[str, ...] | frozenset[str],
    *,
    weight: float | None = None,
) -> bool:
    """True for the weight-1.0 MCL + Baker's DINOv3-S+ replace that scored 0.908."""
    selected = frozenset(findings)
    if selected != frozenset(DINO3S_REPLACE_FINDINGS):
        return False
    if weight is None:
        return True
    return abs(float(weight) - float(DINO3S_REPLACE_WEIGHT)) < 1e-9


def is_ruled_out_r336ov_family(
    findings: list[str] | tuple[str, ...] | frozenset[str],
    *,
    weight: float | None = None,
) -> bool:
    """True for the PF OA + Synovitis r336 @0.20 overlay that scored 0.935."""
    selected = frozenset(findings)
    if selected != frozenset(R336OV_OVERLAY_FINDINGS):
        return False
    if weight is None:
        return True
    return abs(float(weight) - float(R336OV_OVERLAY_WEIGHT)) < 1e-9


def is_ruled_out_jointline_mill_radimagenet_family(
    findings: list[str] | tuple[str, ...] | frozenset[str],
    *,
    crop_mm: float | None = None,
) -> bool:
    """True for the 90 mm mill-RadImageNet e13 overlay that scored 0.935."""
    selected = frozenset(findings)
    if selected != frozenset(JOINTLINE_OVERLAY_FINDINGS):
        return False
    if crop_mm is None:
        return True
    return abs(float(crop_mm) - float(JOINTLINE_CROP_MM)) < 1e-6


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


def _dummy_rank_trap(values: np.ndarray, atol: float = 1e-8) -> bool:
    """True when every entry is 0, 0.5, or 1, or the array is empty / non-finite."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return True
    rounded = np.round(arr / 0.5) * 0.5
    if not np.allclose(arr, rounded, atol=atol):
        return False
    uniq = {float(x) for x in np.unique(np.round(rounded, decimals=8))}
    return uniq.issubset({0.0, 0.5, 1.0})


def fail_closed_rank_blend(
    anchor: pd.DataFrame,
    overlay: pd.DataFrame | None,
    weights: Mapping[str, float],
    *,
    min_std: float = 1e-8,
) -> pd.DataFrame:
    """Rank-blend a legal subset. Any bad overlay returns the anchor unchanged.

    Missing frame, column mismatch, NaN, dummy ranks {0, 0.5, 1}, a flat
    column, weight >= 0.50, and the ruled-out finding sets all return anchor.
    """
    if list(anchor.columns) != list(SUBMISSION_COLUMNS):
        raise FusionError("anchor columns must match the official header")
    if overlay is None or list(overlay.columns) != list(SUBMISSION_COLUMNS):
        return anchor.copy()
    w = {name: float(weights.get(name, 0.0)) for name in FINDINGS}
    selected = overlay_finding_set(w)
    if not selected:
        return anchor.copy()
    if any(value >= 0.50 for value in w.values()):
        return anchor.copy()
    if is_uniform_all_finding_overlay(w, min_weight=0.5):
        return anchor.copy()
    if is_ruled_out_r336ov_family(selected):
        return anchor.copy()
    if is_ruled_out_public_native384_family(selected):
        return anchor.copy()
    if is_ruled_out_dino3s_replace_family(selected):
        return anchor.copy()
    if is_ruled_out_jointline_mill_radimagenet_family(selected):
        return anchor.copy()

    left = anchor.drop_duplicates(ID_COLUMN, keep="first")
    right = overlay.drop_duplicates(ID_COLUMN, keep="first")
    right = right.set_index(ID_COLUMN).reindex(left[ID_COLUMN].tolist()).reset_index()
    block = right[list(selected)].to_numpy(dtype=np.float64)
    if _dummy_rank_trap(block):
        return anchor.copy()
    for name in selected:
        column = right[name].to_numpy(dtype=np.float64)
        if not np.isfinite(column).all() or float(np.std(column)) < float(min_std):
            return anchor.copy()
    try:
        return per_finding_rank_blend(left, right[list(SUBMISSION_COLUMNS)], w)
    except FusionError:
        return anchor.copy()
