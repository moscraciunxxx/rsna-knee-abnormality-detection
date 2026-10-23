"""TEST overlay: unequal ConvNeXt weights on the pinned 517-study license.

Dropped findings (blended-delta CI lower bound <= 0) stay on mill.
Importing this module does not start inference and does not submit.
"""

from __future__ import annotations

try:
    from rsna_knee.convnext_overlay_kernel import HEADER, run_overlay_map
    from rsna_knee.fusion import convnext_oof_unequal_weights
except ImportError:
    from convnext_overlay_kernel import HEADER, run_overlay_map
    from fusion import convnext_oof_unequal_weights

TAG = "[cnxt-k7-uneq]"


def main() -> None:
    weights = convnext_oof_unequal_weights()
    if any(float(value) >= 0.50 for value in weights.values()):
        raise RuntimeError("unequal weight must stay under 0.50")
    dropped = ("Lateral Meniscus", "Lateral OA", "PF OA", "Contusion", "Fracture")
    if any(weights[name] != 0.0 for name in dropped):
        raise RuntimeError("dropped findings must stay on mill")
    print(f"{TAG} header {HEADER}", flush=True)
    run_overlay_map(weights, tag=TAG)
