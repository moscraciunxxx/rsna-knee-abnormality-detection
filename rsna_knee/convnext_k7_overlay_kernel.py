"""TEST overlay: mill V10 plus the seven-finding ConvNeXt mix at weight 0.15.

Findings: ACL, MCL, Medial Meniscus, Medial OA, Effusion, Synovitis, Baker's.
Not the two-finding ACL+Effusion mix. Importing this module does not start
inference and does not submit.
"""

from __future__ import annotations

try:
    from rsna_knee.convnext_overlay_kernel import HEADER, run_overlay
    from rsna_knee.fusion import (
        CONVNEXT_OOF_K7_FINDINGS,
        CONVNEXT_OOF_K7_WEIGHT,
        convnext_oof_k7_weights,
    )
except ImportError:  # unpacked next to this file on Kaggle
    from convnext_overlay_kernel import HEADER, run_overlay
    from fusion import (
        CONVNEXT_OOF_K7_FINDINGS,
        CONVNEXT_OOF_K7_WEIGHT,
        convnext_oof_k7_weights,
    )

TAG = "[cnxt-k7]"


def control_score_allows_push(public_score: float | None) -> bool:
    """True only when the mill control's COMPLETE publicScore is at least 0.936."""
    if public_score is None:
        return False
    return float(public_score) >= 0.936


def main() -> None:
    weights = convnext_oof_k7_weights()
    if abs(float(weights["ACL"]) - CONVNEXT_OOF_K7_WEIGHT) > 1e-9:
        raise RuntimeError("k7 weight drifted")
    if "PF OA" in CONVNEXT_OOF_K7_FINDINGS:
        raise RuntimeError("k7 must not include the ruled-out PF OA column")
    print(f"{TAG} header {HEADER}", flush=True)
    run_overlay(CONVNEXT_OOF_K7_FINDINGS, CONVNEXT_OOF_K7_WEIGHT, tag=TAG)
