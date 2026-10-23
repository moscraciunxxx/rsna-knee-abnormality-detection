"""Build / refresh kaggle_kernel_convnext_mil870_gate. Does not push. Does not Submit."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "kaggle_kernel_convnext_mil870_gate"
MILL = ROOT / "data" / "mill_v10_holdout870.csv"


def main() -> None:
    meta = DST / "kernel-metadata.json"
    main_py = DST / "main.py"
    if not meta.is_file() or not main_py.is_file():
        raise SystemExit(f"missing kernel sketch under {DST}; restore from docs/CONVNEXT_MIL870_GATE_PREP.md")
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["id"] == "vitaliecervinschi/rsna-knee-convnext-mil870-gate"
    assert "goodpjw2008/rsna-knee-2-5d-convnext-reader" in data["dataset_sources"]
    assert "stevenleehans/rsna-knee-llm-report-labels" in data["dataset_sources"]
    text = main_py.read_text(encoding="utf-8")
    assert "DO NOT SUBMIT" in text
    assert "convnext_holdout870.csv" in text
    assert "TIME_BUDGET_S" in text
    if MILL.is_file():
        print(f"mill bar ready: {MILL}")
    print(f"kernel sketch OK at {DST}")
    print("Push only when GPU lane free: kaggle kernels push -p kaggle_kernel_convnext_mil870_gate -t 5400")
    print("Do NOT push while coatnet MCL+Effusion overlay is RUNNING. Do NOT Submit.")


if __name__ == "__main__":
    main()
