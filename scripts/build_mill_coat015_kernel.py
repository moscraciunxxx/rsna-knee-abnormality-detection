"""NO-GO gate for a mill V10 + 4-member CoAt-arm notebook.

The private license is MCL+Effusion at 0.15 on that arm. The measured clocks
do not clear 8.75h on the 1322 plan_cache count. This script does not write a
kernel and does not Submit.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.scoring_time import (  # noqa: E402
    COAT_ARM_END_S,
    COAT_ARM_START_S,
    N_HOLDOUT,
    N_PLAN_CACHE,
    REPAIRED_STAGE_START_S,
    SCORING_CAP_S,
    SAFE_MARGIN_S,
    coat_arm_wall_seconds,
    measured_mill_coat_plan,
    scale_wall_seconds,
    scoring_run_allowed,
)


def main() -> int:
    plan = measured_mill_coat_plan()
    repair_s = coat_arm_wall_seconds(REPAIRED_STAGE_START_S, COAT_ARM_END_S)
    repair_scaled = scale_wall_seconds(repair_s, N_HOLDOUT, N_PLAN_CACHE)
    cut_sum = float(plan["sum_s"]) - repair_scaled
    cut_ok = scoring_run_allowed(cut_sum, 0.0)
    print(
        f"mill_scaled_h={float(plan['mill_scaled_s']) / 3600:.3f} "
        f"coat_scaled_h={float(plan['coat_scaled_s']) / 3600:.3f} "
        f"sum_h={float(plan['sum_s']) / 3600:.3f} "
        f"cap_h={SCORING_CAP_S / 3600:.3f} margin_min={SAFE_MARGIN_S / 60:.0f} "
        f"allowed={plan['allowed']}"
    )
    print(
        f"drop_repairv1_sum_h={cut_sum / 3600:.3f} clock_ok={cut_ok} "
        "but that is a 3-member arm, not the gated 4-member vector. Still no Submit."
    )
    if plan["allowed"]:
        print("GO would require a kernel. This build does not push.")
        return 0
    print("NO-GO: do not push mill+CoAt arm. Do not Submit. Keep 55779066.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
