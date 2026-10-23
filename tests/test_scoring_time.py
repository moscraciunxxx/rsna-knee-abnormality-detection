"""Mill + CoAt-arm timing gate. Inputs are kernel-log clocks, not the 4.2h guess."""

import subprocess
import sys
from pathlib import Path

from rsna_knee.scoring_time import (
    N_PLAN_CACHE,
    SCORING_CAP_S,
    SAFE_MARGIN_S,
    measured_mill_coat_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def test_measured_mill_plus_coat_arm_misses_8h45_on_the_1322_plan():
    plan = measured_mill_coat_plan()
    assert plan["allowed"] is False
    assert plan["sum_s"] > SCORING_CAP_S - SAFE_MARGIN_S
    assert N_PLAN_CACHE == 1322
    assert plan["mill_scaled_s"] > 4.0 * 3600.0
    assert plan["coat_scaled_s"] > 4.5 * 3600.0


def test_public_mill_kernel_output_does_not_cover_a_hidden_overlay():
    import pandas as pd

    from rsna_knee.scoring_time import (
        COAT_ARM_END_S,
        COAT_ARM_START_S,
        N_HOLDOUT,
        N_PLAN_CACHE,
        coat_arm_wall_seconds,
        coat_only_overlay_allowed,
        mill_kernel_output_covers_hidden,
        scale_wall_seconds,
    )

    path = ROOT / "artifacts" / "mill_v10_kernel_output" / "submission.csv"
    frame = pd.read_csv(path)
    n_rows = len(frame)
    assert n_rows == 3
    assert "Baker's" in frame.columns
    assert mill_kernel_output_covers_hidden(n_rows, N_PLAN_CACHE) is False
    coat_870 = coat_arm_wall_seconds(COAT_ARM_START_S, COAT_ARM_END_S)
    coat_plan = scale_wall_seconds(coat_870, N_HOLDOUT, N_PLAN_CACHE)
    assert coat_plan < SCORING_CAP_S - SAFE_MARGIN_S
    assert coat_only_overlay_allowed(n_rows, N_PLAN_CACHE, coat_plan) is False


def test_effnet870_kernel_is_private_and_matches_mill_holdout_ids():
    import importlib.util

    import pandas as pd

    from rsna_knee.mill_holdout import holdout_ids_and_y
    from rsna_knee.schema import ID_COLUMN

    path = ROOT / "kaggle_kernel_effnet870" / "main.py"
    spec = importlib.util.spec_from_file_location("effnet870_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = path.read_text(encoding="utf-8")
    assert "DO NOT SUBMIT" in text
    assert "submission.csv" not in text
    assert "Baker's" in text
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(
        ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv",
        dtype={ID_COLUMN: str},
    )
    expect, _ = holdout_ids_and_y(train, v4)
    assert module.holdout_ids(train, v4) == list(expect)


def test_effnet870_artifact_stops_and_is_not_submitted():
    import pandas as pd

    from rsna_knee.mill_holdout import decide_private_overlay, holdout_ids_and_y, join_mill_holdout
    from rsna_knee.schema import ID_COLUMN

    path = ROOT / "artifacts" / "effnet870" / "effnet_holdout870.csv"
    note = (ROOT / "artifacts" / "effnet870" / "DO_NOT_SUBMIT_HOLDOUT.txt").read_text(encoding="utf-8")
    assert "DO NOT SUBMIT" in note
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(
        ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv",
        dtype={ID_COLUMN: str},
    )
    ids, y = holdout_ids_and_y(train, v4)
    mill = pd.read_csv(ROOT / "data" / "mill_v10_holdout870.csv", dtype={ID_COLUMN: str})
    new = pd.read_csv(path, dtype={ID_COLUMN: str})
    decision, weights = decide_private_overlay(
        join_mill_holdout(ids, mill, y).per_auc,
        join_mill_holdout(ids, new, y).per_auc,
    )
    assert decision == "stop"
    assert weights is None
    assert "Loss is inconclusive" in (ROOT / "kaggle_kernel_effnet870" / "main.py").read_text(encoding="utf-8")


def test_build_script_refuses_to_emit_a_kernel():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_mill_coat015_kernel.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "NO-GO" in proc.stdout
    assert "Do not Submit" in proc.stdout
    assert not (ROOT / "kaggle_kernel_mill_coat015").exists()
