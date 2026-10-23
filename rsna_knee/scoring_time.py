"""Whether a mill-plus-overlay kernel can finish inside the scoring cap.

Stage times are inputs from a kernel log. This module does not read Kaggle.
"""
from __future__ import annotations

# Kernel-log clocks. 1322 is int(0.3 * 4407) from plan_cache, not a counted hidden cohort.
MILL_HOLDOUT_WALL_S = 10325.287214072  # rsna-knee-mill-holdout-870, max stream time, n=870
COAT_ARM_START_S = 13671.495493888855  # residual_coat START
COAT_ARM_END_S = 24868.04927420616  # repairv1_coat COMPLETE
REPAIRED_STAGE_START_S = 22009.784915208817  # repairv1_coat START
N_HOLDOUT = 870
N_PLAN_CACHE = 1322

SCORING_CAP_S = 9.0 * 3600.0
# Startup, a second DICOM walk, and the kernel's own 8h TIME_BUDGET sit inside
# the 9h platform cap. Four minutes of slack is not a finish.
SAFE_MARGIN_S = 15.0 * 60.0


def coat_arm_wall_seconds(arm_start_s: float, arm_end_s: float) -> float:
    """Wall clock from the first CoAt-arm stage start to the last stage end."""
    start = float(arm_start_s)
    end = float(arm_end_s)
    if end < start:
        raise ValueError(f"CoAt arm end {end} is before start {start}")
    return end - start


def scale_wall_seconds(seconds: float, n_from: int, n_to: int) -> float:
    """Linear study-count scale. Both counts must be positive."""
    src = int(n_from)
    dst = int(n_to)
    if src <= 0 or dst <= 0:
        raise ValueError(f"study counts must be positive, got {src} -> {dst}")
    return float(seconds) * (float(dst) / float(src))


def scoring_run_allowed(
    mill_s: float,
    coat_s: float,
    *,
    cap_s: float = SCORING_CAP_S,
    margin_s: float = SAFE_MARGIN_S,
) -> bool:
    """True only when mill plus the CoAt arm ends at least margin_s before cap_s."""
    return (float(mill_s) + float(coat_s)) <= (float(cap_s) - float(margin_s))


def mill_plus_coat_plan(
    mill_wall_s: float,
    coat_start_s: float,
    coat_end_s: float,
    n_from: int,
    n_to: int,
) -> dict[str, float | bool]:
    """Scale a measured mill run and a measured CoAt-arm window onto n_to studies.

    ``mill_wall_s`` is the kernel log clock for ``n_from`` studies, not a guess.
    """
    coat = coat_arm_wall_seconds(coat_start_s, coat_end_s)
    mill_scaled = scale_wall_seconds(mill_wall_s, n_from, n_to)
    coat_scaled = scale_wall_seconds(coat, n_from, n_to)
    total = mill_scaled + coat_scaled
    return {
        "mill_scaled_s": mill_scaled,
        "coat_scaled_s": coat_scaled,
        "sum_s": total,
        "allowed": scoring_run_allowed(mill_scaled, coat_scaled),
    }


def mill_kernel_output_covers_hidden(n_rows: int, n_hidden: int) -> bool:
    """True when a mounted mill submission has one row per hidden test study.

    A 3-row public placeholder does not cover the hidden test. Kaggle Save
    Version output is that placeholder, not submission 55779066's hidden preds.
    """
    hidden = int(n_hidden)
    rows = int(n_rows)
    if hidden <= 3 or rows <= 3:
        return False
    return rows >= hidden


def coat_only_overlay_allowed(
    n_mount_rows: int,
    n_hidden: int,
    coat_scaled_s: float,
) -> bool:
    """CoAt-only overlay is allowed only with hidden mill rows and time slack.

    Mill is not re-inferred, so its seconds are 0. The mount must still cover
    every hidden study or the blend is fail-closed to nothing.
    """
    if not mill_kernel_output_covers_hidden(n_mount_rows, n_hidden):
        return False
    return scoring_run_allowed(0.0, coat_scaled_s)


def measured_mill_coat_plan() -> dict[str, float | bool]:
    """Plan for the logged mill holdout plus the logged 4-member CoAt arm."""
    return mill_plus_coat_plan(
        MILL_HOLDOUT_WALL_S,
        COAT_ARM_START_S,
        COAT_ARM_END_S,
        N_HOLDOUT,
        N_PLAN_CACHE,
    )


# rsna-knee-convnext-mil870-gate log: "[cnxt870] done elapsed=813.1s" on n=870.
CONVNEXT_HOLDOUT_WALL_S = 813.1


def convnext_overlay_estimate_s(n_studies: int) -> float:
    """Scaled mill V10 plus ConvNeXt reader wall clock for n_studies."""
    return scale_wall_seconds(
        MILL_HOLDOUT_WALL_S + CONVNEXT_HOLDOUT_WALL_S,
        N_HOLDOUT,
        n_studies,
    )


def convnext_overlay_clock_allowed(n_studies: int) -> bool:
    """True when mill plus the ConvNeXt reader finishes before the 8.75 h line.

    ``n_studies`` is the discovered hidden-test count, not the 1322 plan_cache
    stand-in. A non-positive count is refused.
    """
    n = int(n_studies)
    if n <= 0:
        return False
    return convnext_overlay_estimate_s(n) < (SCORING_CAP_S - SAFE_MARGIN_S)
