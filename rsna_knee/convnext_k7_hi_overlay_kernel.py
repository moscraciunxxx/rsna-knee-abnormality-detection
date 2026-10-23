"""TEST overlay: the same seven ConvNeXt findings at weight 0.45.

Do not submit this while 56926403 (weight 0.15) is the scored test of transfer.
Importing this module does not start inference.
"""

from __future__ import annotations

try:
    from rsna_knee.convnext_overlay_kernel import HEADER, run_overlay
    from rsna_knee.fusion import (
        CONVNEXT_OOF_K7_FINDINGS,
        CONVNEXT_OOF_K7_HI_WEIGHT,
        convnext_oof_k7_hi_weights,
    )
except ImportError:
    from convnext_overlay_kernel import HEADER, run_overlay
    from fusion import (
        CONVNEXT_OOF_K7_FINDINGS,
        CONVNEXT_OOF_K7_HI_WEIGHT,
        convnext_oof_k7_hi_weights,
    )

TAG = "[cnxt-k7-hi]"


def main() -> None:
    weights = convnext_oof_k7_hi_weights()
    if abs(float(weights["Effusion"]) - CONVNEXT_OOF_K7_HI_WEIGHT) > 1e-9:
        raise RuntimeError("k7 hi weight drifted")
    if len(CONVNEXT_OOF_K7_FINDINGS) != 7:
        raise RuntimeError("k7 hi set must stay seven findings")
    print(f"{TAG} header {HEADER}", flush=True)
    run_overlay(CONVNEXT_OOF_K7_FINDINGS, CONVNEXT_OOF_K7_HI_WEIGHT, tag=TAG)
