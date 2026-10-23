"""CoAtNet mill-870 private gate: same 870 IDs, holdout comp shim, no Submit."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.coatnet_gate import SHIM_SOURCE, holdout_study_ids, shim_namespace
from rsna_knee.guards import (
    kernel_discovers_competitions_mount,
    kernel_has_t4_gpu_probe,
    kernel_keeps_bakers_apostrophe,
    kernel_reads_test_report_column,
    kernel_source_text,
)
from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS

TRAIN_CSV = ROOT / "data" / "train.csv"
V4_CSV = ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv"
MILL870 = ROOT / "data" / "mill_v10_holdout870.csv"
KERNEL = ROOT / "kaggle_kernel_coatnet870_gate" / "main.py"
META = ROOT / "kaggle_kernel_coatnet870_gate" / "kernel-metadata.json"
VITB_SPEC = ROOT / "kaggle_kernel_mill870_ov" / "overlay_spec.json"
CLAIM_META = ROOT / "refs" / "ctogaurav__rsna-knee-pure-coatnet-0946" / "kernel-metadata.json"


@pytest.fixture(scope="module")
def built_kernel():
    import runpy

    runpy.run_path(str(ROOT / "scripts" / "build_coatnet870_gate_kernel.py"), run_name="__main__")
    assert KERNEL.is_file()
    return KERNEL


def test_shim_ids_match_mill_csv_and_library():
    if not V4_CSV.is_file():
        pytest.skip(f"steven v4 CSV missing: {V4_CSV}")
    train = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    v4 = pd.read_csv(V4_CSV, dtype={ID_COLUMN: str})
    library = holdout_study_ids(train, v4)
    ids_y, _y = holdout_ids_and_y(train, v4)
    assert library == ids_y
    assert len(library) == 870
    ns = shim_namespace()
    pasted = ns["_gate_holdout_ids"](TRAIN_CSV, V4_CSV)
    assert tuple(pasted) == library
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    assert list(mill.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in mill.columns
    assert tuple(mill[ID_COLUMN].astype(str)) == library
    gold = set(ns["_gate_gold_ids"](train))
    assert gold.isdisjoint(library)
    assert len(gold) == 58


def test_stage_comp_links_only_the_holdout(tmp_path, monkeypatch):
    real = tmp_path / "real"
    series = real / "train_series"
    series.mkdir(parents=True)
    ids = ["uid-a", "uid-b", "uid-c"]
    rows = []
    for uid in ids:
        (series / uid / "ser1").mkdir(parents=True)
        rows.append(
            {
                "StudyInstanceUID": uid,
                "SeriesInstanceUID": uid + "-s",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": "Sagittal",
            }
        )
    pd.DataFrame(rows).to_csv(real / "train_series.csv", index=False)
    linked: list[tuple[str, str]] = []

    def fake_symlink(src, dst, target_is_directory=False):
        linked.append((str(src), str(dst)))
        Path(dst).mkdir()

    monkeypatch.setattr(os, "symlink", fake_symlink)
    ns = shim_namespace()
    dest = tmp_path / "hold"
    ns["_gate_stage_comp"](real, ["uid-c", "uid-a"], dest)
    test_ids = pd.read_csv(dest / "test.csv", dtype={ID_COLUMN: str})[ID_COLUMN].tolist()
    assert test_ids == ["uid-c", "uid-a"]
    series_out = pd.read_csv(dest / "test_series.csv", dtype={ID_COLUMN: str})
    assert set(series_out[ID_COLUMN]) == {"uid-c", "uid-a"}
    assert "uid-b" not in set(series_out[ID_COLUMN])
    sample = pd.read_csv(dest / "sample_submission.csv")
    assert list(sample.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in sample.columns
    assert len(sample) == 2
    train_stub = pd.read_csv(dest / "train.csv")
    assert len(train_stub) == 2
    assert [Path(item[1]).name for item in linked] == ["uid-c", "uid-a"]
    with pytest.raises(RuntimeError, match="missing"):
        ns["_gate_stage_comp"](real, ["uid-a", "missing-uid"], tmp_path / "other")


def test_kernel_contract(built_kernel):
    src = kernel_source_text(built_kernel)
    ast.parse(src)
    assert src.count("from __future__ import annotations") == 1
    assert kernel_has_t4_gpu_probe(src)
    assert kernel_discovers_competitions_mount(src)
    assert kernel_keeps_bakers_apostrophe(src)
    assert not kernel_reads_test_report_column(src)
    assert "DO NOT SUBMIT" in src
    assert "coatnet_holdout870.csv" in src
    assert "HOLD_SEED = 2026" in src
    assert "EXPECTED_HOLDOUT_N = 870" in src
    assert "gold 58 leaked into coatnet-gate train ids" in src
    assert "RSNA_COMP_ROOT" in src
    assert "find_root forced RSNA_COMP_ROOT" in src
    assert "competitions submit" not in src
    assert "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191" not in src
    assert src.find("nvidia-smi") < src.find("import torch")
    assert src.find("RSNA_COMP_ROOT") < src.find("# ===== notebook cell =====")
    assert src.find("7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d") < src.find(
        "# <<<COATNET870_TAIL"
    )
    assert src.find("coatnet_holdout870.csv") > src.find("# <<<COATNET870_TAIL")
    assert "_gate_pred_path.unlink()" in src
    assert "decide_private_overlay" in src
    assert "saturday_after_coatnet.py" in src
    shim = src.split("# <<<COATNET870_SHIM", 1)[1].split("# >>>COATNET870_SHIM", 1)[0]
    assert shim.strip() == SHIM_SOURCE.strip()
    meta = json.loads(META.read_text(encoding="utf-8"))
    claim = json.loads(CLAIM_META.read_text(encoding="utf-8"))
    assert meta["id"] == "vitaliecervinschi/rsna-knee-coatnet870-gate"
    assert meta["is_private"] is True
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    assert meta["kernel_type"] == "script"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["docker_image"] == claim["docker_image"]
    assert meta["kernel_sources"] == claim["kernel_sources"]
    assert meta["model_sources"] == claim["model_sources"]
    for slug in claim["dataset_sources"]:
        assert slug in meta["dataset_sources"]
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert (ROOT / "kaggle_kernel_coatnet870_gate" / "mill_v10_holdout870.csv").is_file()
    assert "Baker's" in FINDINGS


def test_saturday_after_coatnet_does_not_touch_vitb_spec(tmp_path, capsys):
    before = VITB_SPEC.read_text(encoding="utf-8")
    mill = pd.read_csv(MILL870, dtype={ID_COLUMN: str})
    worse = mill.copy()
    for col in FINDINGS:
        worse[col] = 1.0 - mill[col].astype(float)
    coat_path = tmp_path / "coatnet_holdout870.csv"
    spec_path = tmp_path / "overlay_spec.json"
    worse.to_csv(coat_path, index=False)
    import runpy

    argv = sys.argv
    sys.argv = [
        "saturday_after_coatnet.py",
        "--mill",
        str(MILL870),
        "--coatnet",
        str(coat_path),
        "--spec",
        str(spec_path),
    ]
    try:
        runpy.run_path(str(ROOT / "scripts" / "saturday_after_coatnet.py"), run_name="__main__")
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "STOP_PUBLIC_TRAINS" in out
    assert "keep 55779066" in out
    assert "DO NOT SUBMIT" in out
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["decision"] == "stop"
    assert spec["keep"] == "55779066"
    assert spec["findings"] == []
    assert spec["weight"] == 0.0
    assert VITB_SPEC.read_text(encoding="utf-8") == before
    vitb = json.loads(before)
    assert vitb["decision"] == "hold"
    assert vitb["weight"] == 0.0
