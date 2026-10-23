"""Patch r336ov into a mill-870-licensed TEST overlay. Do not use PF OA+Synovitis @0.20."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_kernel_r336ov"
DST = ROOT / "kaggle_kernel_mill870_ov"
SPEC = DST / "overlay_spec.json"


def main() -> None:
    if not SPEC.is_file():
        raise SystemExit(f"missing {SPEC}; run saturday_after_vitb.py first")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec.get("decision") != "overlay":
        raise SystemExit(f"spec decision is {spec.get('decision')!r}; do not build overlay")
    if spec.get("source") == "coatnet870-gate":
        raise SystemExit(
            "coatnet870 licensed MCL+Effusion on the CoAt stack, not r336; "
            "do not retarget kaggle_kernel_r336ov"
        )
    findings = tuple(spec["findings"])
    weight = float(spec["weight"])
    if set(findings) == {"PF OA", "Synovitis"}:
        raise SystemExit("r336ov PF OA+Synovitis family is ruled out")
    if weight >= 0.50:
        raise SystemExit("overlay weight must be ≪ 0.50")
    if not findings or len(findings) > 2:
        raise SystemExit("overlay at most 1–2 findings")
    DST.mkdir(exist_ok=True)
    text = (SRC / "main.py").read_text(encoding="utf-8")
    text = text.replace(
        'R336OV_OVERLAY_FINDINGS: tuple[str, ...] = ("PF OA", "Synovitis")',
        f"R336OV_OVERLAY_FINDINGS: tuple[str, ...] = {findings!r}",
    )
    text = text.replace(
        "R336OV_OVERLAY_WEIGHT = 0.20",
        f"R336OV_OVERLAY_WEIGHT = {weight}",
    )
    if 'R336OV_OVERLAY_FINDINGS: tuple[str, ...] = ("PF OA", "Synovitis")' in text:
        raise SystemExit("failed to retarget overlay findings")
    (DST / "main.py").write_text(text, encoding="utf-8")
    meta = (SRC / "kernel-metadata.json").read_text(encoding="utf-8")
    meta = meta.replace("vitaliecervinschi/rsna-knee-r336ov-v1", "vitaliecervinschi/rsna-knee-mill870-ov")
    meta = meta.replace("RSNA Knee r336ov v1", "RSNA Knee mill870 ov")
    (DST / "kernel-metadata.json").write_text(meta, encoding="utf-8")
    print(f"wrote overlay kernel findings={findings} weight={weight}")


if __name__ == "__main__":
    main()
