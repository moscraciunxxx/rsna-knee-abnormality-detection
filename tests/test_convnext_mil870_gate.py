"""ConvNeXt MIL mill-870 private gate sketch: metadata, guards, no Submit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KERNEL = ROOT / "kaggle_kernel_convnext_mil870_gate" / "main.py"
META = ROOT / "kaggle_kernel_convnext_mil870_gate" / "kernel-metadata.json"
PREP = ROOT / "docs" / "CONVNEXT_MIL870_GATE_PREP.md"
BUILDER = ROOT / "scripts" / "build_convnext_mil870_gate_kernel.py"


def test_prep_doc_exists():
    assert PREP.is_file()
    text = PREP.read_text(encoding="utf-8")
    assert "goodpjw2008/rsna-knee-2-5d-convnext-reader" in text
    assert "DO NOT SUBMIT" in text or "No Submit" in text or "Never Submit" in text


def test_metadata_mounts_reader_only_plus_v4():
    data = json.loads(META.read_text(encoding="utf-8"))
    assert data["id"] == "vitaliecervinschi/rsna-knee-convnext-mil870-gate"
    assert data["is_private"] is True
    assert data["enable_gpu"] is True
    assert "goodpjw2008/rsna-knee-2-5d-convnext-reader" in data["dataset_sources"]
    assert "stevenleehans/rsna-knee-llm-report-labels" in data["dataset_sources"]
    # Must NOT mount the full coat stack (that would compete / correlate)
    coatish = [s for s in data["dataset_sources"] if "coatnet" in s or "coat-resgated" in s]
    assert coatish == [], coatish


def test_main_is_holdout_gate_not_submit():
    text = KERNEL.read_text(encoding="utf-8")
    assert "DO NOT SUBMIT" in text
    assert "convnext_holdout870.csv" in text
    assert "TIME_BUDGET_S" in text
    assert "Baker's" in text
    assert "competitions submit" not in text.lower()


def test_builder_validates():
    import runpy
    runpy.run_path(str(BUILDER), run_name="__main__")


def test_decision_script_protects_other_specs():
    sat = (ROOT / "scripts" / "saturday_after_convnext.py").read_text(encoding="utf-8")
    assert "PROTECTED" in sat
    assert "coatnet870" in sat
