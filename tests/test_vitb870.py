"""Drive mill-870 CSV as the join artifact and ViT-B 870 encode guards."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.fusion import FusionError
from rsna_knee.guards import (
    kernel_discovers_competitions_mount,
    kernel_has_t4_gpu_probe,
    kernel_keeps_bakers_apostrophe,
    kernel_reads_test_report_column,
    kernel_source_text,
)
from rsna_knee.mill_holdout import (
    MillHoldoutMissing,
    decide_private_overlay,
    holdout_ids_and_y,
    join_mill_holdout,
    licensed_overlay_weights,
    mill_holdout_table,
    vitb_loses_every_finding,
)
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "train.csv"
GOLD_CSV = ROOT / "data" / "gold58_ids.csv"
V4_CSV = ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv"
MILL870 = ROOT / "data" / "mill_v10_holdout870.csv"
KERNEL_VITB = ROOT / "kaggle_kernel_vitb870" / "main.py"


@pytest.fixture(scope="module")
def holdout():
    if not V4_CSV.is_file():
        pytest.skip(f"steven v4 CSV missing: {V4_CSV}")
    train = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    v4 = pd.read_csv(V4_CSV, dtype={ID_COLUMN: str})
    return holdout_ids_and_y(train, v4)


def test_mill870_csv_is_the_join_artifact_not_a_log_line():
    assert MILL870.is_file(), "mill-870 must be a real CSV artifact"
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    assert list(mill.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in mill.columns
    assert mill.shape[1] == 13
    assert mill.shape[0] >= 800
    if not V4_CSV.is_file():
        pytest.skip(f"steven v4 CSV missing: {V4_CSV}")
    train = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    v4 = pd.read_csv(V4_CSV, dtype={ID_COLUMN: str})
    ids, y = holdout_ids_and_y(train, v4)
    gold = set(pd.read_csv(GOLD_CSV, dtype={ID_COLUMN: str})[ID_COLUMN].astype(str))
    joined = join_mill_holdout(ids, mill, y)
    assert set(joined.ids) == set(ids)
    assert set(joined.ids).isdisjoint(gold)
    assert joined.mill.shape == (len(ids), 12)
    assert len(joined.ids) == mill.shape[0]


def test_vitb_table_fail_closes_if_uids_are_not_the_870():
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    ids = tuple(mill[ID_COLUMN].astype(str))
    y = np.zeros((len(ids), 12))
    y[:, 0] = 1
    y[0, 0] = 0
    wrong = mill_holdout_table(("not-a-holdout",), np.zeros((1, 12)))
    with pytest.raises(MillHoldoutMissing, match="MISSING"):
        join_mill_holdout(ids, wrong, y)


def test_holdout_y_is_llm_soft_not_gold58(holdout):
    """Overlay y is v4≥0.5 on the 870, never the 58 expert rows."""
    ids, y = holdout
    gold = set(pd.read_csv(GOLD_CSV, dtype={ID_COLUMN: str})[ID_COLUMN].astype(str))
    assert set(ids).isdisjoint(gold)
    gold_frame = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    gold_ids = gold_frame.loc[
        gold_frame[list(FINDINGS)].notna().all(axis=1), ID_COLUMN
    ].astype(str)
    assert len(gold_ids) == 58
    assert y.shape[0] != 58
    assert y.shape[0] >= 800


def test_decide_private_overlay_stops_or_holds_without_public_0936():
    mill = np.linspace(0.80, 0.91, 12)
    lose = mill - 0.04
    decision, weights = decide_private_overlay(mill, lose)
    assert decision == "stop"
    assert weights is None
    tie = mill.copy()
    decision2, weights2 = decide_private_overlay(mill, tie)
    assert decision2 == "hold"
    one = mill.copy()
    one[FINDINGS.index("Effusion")] = mill[FINDINGS.index("Effusion")] + 0.05
    decision3, weights3 = decide_private_overlay(mill, one)
    assert decision3 == "overlay"
    assert weights3 is not None
    assert weights3["Effusion"] > 0.0
    assert weights3["Effusion"] < 0.50


def test_vitb_loses_every_finding_is_the_stop_gate():
    mill = np.linspace(0.80, 0.91, 12)
    lose = mill - 0.05
    assert vitb_loses_every_finding(mill, lose) is True
    win_one = lose.copy()
    win_one[3] = mill[3] + 0.02
    assert vitb_loses_every_finding(mill, win_one) is False
    zeros = licensed_overlay_weights(mill, lose)
    assert all(v == 0.0 for v in zeros.values())


def test_r336ov_pair_still_ruled_out_even_if_vitb_wins_those_two():
    mill = np.full(12, 0.80)
    new = mill.copy()
    new[FINDINGS.index("PF OA")] = 0.95
    new[FINDINGS.index("Synovitis")] = 0.95
    with pytest.raises(FusionError, match="r336ov"):
        licensed_overlay_weights(mill, new, max_findings=2, weight=0.15)


def test_saturday_after_vitb_writes_stop_spec(tmp_path, capsys):
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    worse = mill.copy()
    for col in FINDINGS:
        worse[col] = 1.0 - mill[col].astype(float)
    vitb_path = tmp_path / "vitb_holdout870.csv"
    spec_path = tmp_path / "overlay_spec.json"
    worse.to_csv(vitb_path, index=False)
    import runpy
    import sys

    argv = sys.argv
    sys.argv = [
        "saturday_after_vitb.py",
        "--mill",
        str(MILL870),
        "--vitb",
        str(vitb_path),
        "--spec",
        str(spec_path),
    ]
    try:
        runpy.run_path(str(ROOT / "scripts" / "saturday_after_vitb.py"), run_name="__main__")
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "STOP_PUBLIC_TRAINS" in out
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["decision"] == "stop"
    assert spec["keep"] == "55779066"


def test_licensed_overlay_builder_refuses_r336ov_pair(tmp_path, monkeypatch):
    import importlib.util

    spec_path = ROOT / "scripts" / "build_licensed_overlay_kernel.py"
    spec = importlib.util.spec_from_file_location("bld_ov", spec_path)
    bld = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bld)
    ov_dir = tmp_path / "ov"
    ov_dir.mkdir()
    spec_file = ov_dir / "overlay_spec.json"
    spec_file.write_text(
        json.dumps({"decision": "overlay", "findings": ["PF OA", "Synovitis"], "weight": 0.15}),
        encoding="utf-8",
    )
    monkeypatch.setattr(bld, "DST", ov_dir)
    monkeypatch.setattr(bld, "SPEC", spec_file)
    with pytest.raises(SystemExit, match="r336ov"):
        bld.main()


def test_join_script_stop_when_synthetic_vitb_loses_every_finding(tmp_path, capsys):
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    worse = mill.copy()
    for col in FINDINGS:
        worse[col] = 1.0 - mill[col].astype(float)
    vitb_path = tmp_path / "vitb_holdout870.csv"
    worse.to_csv(vitb_path, index=False)
    import runpy
    import sys

    argv = sys.argv
    sys.argv = [
        "join_mill_holdout.py",
        "--mill",
        str(MILL870),
        "--new",
        str(vitb_path),
    ]
    try:
        runpy.run_path(str(ROOT / "scripts" / "join_mill_holdout.py"), run_name="__main__")
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "STOP_PUBLIC_TRAINS" in out
    assert "keep 55779066" in out


def test_vitb870_kernel_is_holdout_encode_not_dino_v5_not_report():
    src = kernel_source_text(KERNEL_VITB)
    assert kernel_has_t4_gpu_probe(src)
    assert kernel_discovers_competitions_mount(src)
    assert kernel_keeps_bakers_apostrophe(src)
    assert not kernel_reads_test_report_column(src)
    assert "VITB_HOLD_ONLY" in src
    assert "r336_best.pt" in src
    assert "vitb_holdout870.csv" in src
    assert "HOLDOUT_ONLY" in src
    assert "HOLD_SEED = 2026" in src
    assert "VITB_HOLD_ONLY = True" in src
    assert "dinosaur-v5" not in src.lower()
    assert "do not submit" in src.lower() or "DO NOT SUBMIT" in src
    assert "\nrun_dinov2()\n" not in src
    assert "SystemExit" not in src
    assert "mill_v10_holdout870.csv" in src
    assert "STOP_PUBLIC_TRAINS" in src
    mill_shipped = ROOT / "kaggle_kernel_vitb870" / "mill_v10_holdout870.csv"
    assert mill_shipped.is_file()
    assert mill_shipped.stat().st_size > 1000


def test_completed_vitb_artifact_holds_and_does_not_license_overlay(holdout):
    """Real vitb870 output through the shipped join and overlay policy."""
    vitb_path = ROOT / "artifacts" / "saturday_vitb870" / "vitb_holdout870.csv"
    spec_path = ROOT / "artifacts" / "saturday_vitb870" / "overlay_spec.json"
    assert vitb_path.is_file(), "COMPLETE vitb_holdout870.csv missing"
    ids, y = holdout
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    vitb = pd.read_csv(vitb_path, dtype={ID_COLUMN: str})
    assert list(vitb.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in vitb.columns
    mill_j = join_mill_holdout(ids, mill, y)
    vitb_j = join_mill_holdout(ids, vitb, y)
    assert len(mill_j.ids) == 870
    decision, weights = decide_private_overlay(mill_j.per_auc, vitb_j.per_auc)
    assert decision == "hold"
    assert weights is None
    delta = vitb_j.per_auc - mill_j.per_auc
    syn = list(FINDINGS).index("Synovitis")
    assert delta[syn] > 0
    assert float(np.nanmax(delta)) <= 0.01
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["decision"] == "hold"
    assert spec["keep"] == "55779066"
    assert spec["do_not_submit_holdout_ids"] is True
    assert spec["findings"] == []
    assert spec["weight"] == 0.0


def test_completed_probe_b_artifact_stops_on_every_finding(holdout):
    """Real frozen DINOv2-B probe through the shipped join. Stop means no Submit."""
    probe_path = ROOT / "artifacts" / "probe_b870" / "probe_holdout870.csv"
    assert probe_path.is_file(), "COMPLETE probe_holdout870.csv missing"
    ids, y = holdout
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    probe = pd.read_csv(probe_path, dtype={ID_COLUMN: str})
    assert list(probe.columns) == list(SUBMISSION_COLUMNS)
    mill_j = join_mill_holdout(ids, mill, y)
    probe_j = join_mill_holdout(ids, probe, y)
    decision, weights = decide_private_overlay(mill_j.per_auc, probe_j.per_auc)
    assert decision == "stop"
    assert weights is None
    assert bool(np.all(probe_j.per_auc < mill_j.per_auc))
    protocol = (ROOT / "artifacts" / "probe_b870" / "protocol_log.txt").read_text(encoding="utf-8")
    assert "gold_in_train=0" in protocol
    assert "holdout_in_train=0" in protocol
    assert "abort_threshold=0.94" in protocol
    assert "abort=False" in protocol
    assert "unfreeze_last=0" in protocol


def test_completed_coatnet_gate_licenses_mcl_and_effusion_only(holdout):
    """Real coatnet870 CSV through the shipped overlay policy. Not r336, not a Submit."""
    from rsna_knee.fusion import (
        is_ruled_out_dino3s_replace_family,
        is_ruled_out_r336ov_family,
    )

    coat_path = ROOT / "artifacts" / "coatnet870" / "coatnet_holdout870.csv"
    spec_path = ROOT / "artifacts" / "coatnet870" / "overlay_spec.json"
    assert coat_path.is_file(), "COMPLETE coatnet_holdout870.csv missing"
    ids, y = holdout
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    coat = pd.read_csv(coat_path, dtype={ID_COLUMN: str})
    assert list(coat.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in coat.columns
    mill_j = join_mill_holdout(ids, mill, y)
    coat_j = join_mill_holdout(ids, coat, y)
    decision, weights = decide_private_overlay(mill_j.per_auc, coat_j.per_auc)
    assert decision == "overlay"
    assert weights is not None
    licensed = {name for name, weight in weights.items() if weight > 0}
    assert licensed == {"MCL", "Effusion"}
    assert max(weights.values()) <= 0.20
    assert not is_ruled_out_r336ov_family(licensed, weight=max(weights.values()))
    assert not is_ruled_out_dino3s_replace_family(licensed, weight=max(weights.values()))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["decision"] == "overlay"
    assert spec["findings"] == ["MCL", "Effusion"]
    assert spec["weight"] == 0.15
    assert spec["do_not_submit_holdout_ids"] is True
    assert spec["keep"] == "55779066"
