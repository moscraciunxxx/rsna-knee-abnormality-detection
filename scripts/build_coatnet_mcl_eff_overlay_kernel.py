from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "kaggle_kernel_coatnet_mcl_eff_overlay"
MILL_SRC = ROOT / "kaggle_kernel_blend_script" / "main.py"
COAT_SRC = ROOT / "kaggle_kernel_coatnet_test" / "main.py"
MILL_META = ROOT / "kaggle_kernel_blend_script" / "kernel-metadata.json"
COAT_META = ROOT / "kaggle_kernel_coatnet_test" / "kernel-metadata.json"
FUSION = ROOT / "rsna_knee" / "fusion.py"
SPEC = ROOT / "artifacts" / "coatnet870" / "overlay_spec.json"

MAIN_PY = r"""#!/usr/bin/env python3
"""RSNA Knee TEST overlay: mill V10 base + CoAtNet MCL+Effusion @0.15.

Licensed by artifacts/coatnet870/overlay_spec.json (coatnet870-gate).
Push / Save Version only. Do NOT Submit without Vitaly explicit OK.
Keep scored anchor 55779066 until a later publicScore is strictly > 0.936.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# Real hard wall (not a log-string). Daemon thread kills the process.
HARD_LIMIT_S = int(8.75 * 3600)
BLEND_RESERVE_S = 180
T0 = time.time()


def _watchdog() -> None:
    time.sleep(HARD_LIMIT_S)
    print(f"[mcl-eff-ov] WATCHDOG hard kill after {HARD_LIMIT_S}s", flush=True)
    os._exit(99)


threading.Thread(target=_watchdog, daemon=True, name="hard-watchdog").start()
print(
    "[mcl-eff-ov] mill V10 + CoAtNet MCL+Effusion @0.15 TEST overlay. "
    f"Hard watchdog {HARD_LIMIT_S}s. Do not Submit without Vitaly OK. keep 55779066.",
    flush=True,
)

HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
MILL_PY = HERE / "_mill_v10.py"
COAT_PY = HERE / "_coatnet_stack.py"
MILL_CSV = WORK / "mill_v10_base.csv"
COAT_CSV = WORK / "coatnet_test_raw.csv"
SUB_CSV = WORK / "submission.csv"


def _remaining() -> float:
    return HARD_LIMIT_S - (time.time() - T0)


def _run_phase(label: str, script: Path, timeout_s: float) -> None:
    if timeout_s < 60:
        raise TimeoutError(f"{label}: only {timeout_s:.0f}s left; refusing to start")
    print(
        f"[mcl-eff-ov] START {label} timeout={timeout_s:.0f}s remaining={_remaining():.0f}s",
        flush=True,
    )
    if SUB_CSV.is_file():
        SUB_CSV.unlink()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(WORK),
        timeout=timeout_s,
        check=False,
    )
    print(
        f"[mcl-eff-ov] END {label} rc={proc.returncode} elapsed={time.time()-T0:.0f}s",
        flush=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} exited rc={proc.returncode}")
    if not SUB_CSV.is_file():
        raise RuntimeError(f"{label} did not write submission.csv")


# ---- Phase 1: mill V10 (anchor family) ----
mill_timeout = max(60.0, min(_remaining() - BLEND_RESERVE_S, HARD_LIMIT_S * 0.45))
try:
    _run_phase("mill_v10", MILL_PY, mill_timeout)
    shutil.copy2(SUB_CSV, MILL_CSV)
    print(f"[mcl-eff-ov] saved {MILL_CSV}", flush=True)
except Exception as exc:
    print(f"[mcl-eff-ov] mill phase FAILED: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    raise SystemExit(f"mill V10 base required for overlay: {exc}") from exc

# ---- Phase 2: CoAtNet stack (overlay source) ----
coat_timeout = max(60.0, _remaining() - BLEND_RESERVE_S)
coat_ok = False
try:
    _run_phase("coatnet_stack", COAT_PY, coat_timeout)
    shutil.copy2(SUB_CSV, COAT_CSV)
    coat_ok = True
    print(f"[mcl-eff-ov] saved {COAT_CSV}", flush=True)
except subprocess.TimeoutExpired as exc:
    print(f"[mcl-eff-ov] coatnet TIMEOUT; fail-closed to mill: {exc}", flush=True)
except Exception as exc:
    print(
        f"[mcl-eff-ov] coatnet FAILED; fail-closed to mill: {type(exc).__name__}: {exc}",
        flush=True,
    )
    traceback.print_exc()

# ---- Phase 3: licensed overlay blend ----
sys.path.insert(0, str(HERE))
import pandas as pd
from fusion import overlay_weights_from_findings, per_finding_rank_blend

FINDINGS = ("MCL", "Effusion")
WEIGHT = 0.15

mill = pd.read_csv(MILL_CSV, dtype={"StudyInstanceUID": str})
if not coat_ok:
    mill.to_csv(SUB_CSV, index=False)
    print(
        "[mcl-eff-ov] wrote mill-only submission.csv (coatnet unavailable). "
        "keep 55779066 mindset.",
        flush=True,
    )
    raise SystemExit(0)

coat = pd.read_csv(COAT_CSV, dtype={"StudyInstanceUID": str})
mill["StudyInstanceUID"] = mill["StudyInstanceUID"].astype(str)
coat["StudyInstanceUID"] = coat["StudyInstanceUID"].astype(str)
mill = mill.drop_duplicates("StudyInstanceUID")
coat = coat.drop_duplicates("StudyInstanceUID")
ids = mill["StudyInstanceUID"].tolist()
coat = coat.set_index("StudyInstanceUID").reindex(ids).reset_index()
if coat.drop(columns=["StudyInstanceUID"]).isna().any().any():
    raise RuntimeError("coatnet rows missing after reindex to mill")

weights = overlay_weights_from_findings(FINDINGS, WEIGHT)
out = per_finding_rank_blend(mill, coat, weights)
out.to_csv(SUB_CSV, index=False)
print(
    f"[mcl-eff-ov] OVERLAY MCL+Effusion w={WEIGHT:.2f} on mill V10; "
    f"shape={out.shape}; elapsed={time.time()-T0:.0f}s; "
    f"not 1.0 replace; not r336ov; not Submit without Vitaly OK; keep 55779066.",
    flush=True,
)
"""


def _merge(a: list, b: list) -> list:
    out: list = []
    seen: set = set()
    for x in list(a) + list(b):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec.get("decision") != "overlay":
        raise SystemExit(f"spec decision is {spec.get('decision')!r}")
    if set(spec.get("findings") or []) != {"MCL", "Effusion"}:
        raise SystemExit(f"unexpected findings {spec.get('findings')!r}")
    if abs(float(spec.get("weight", 0)) - 0.15) > 1e-9:
        raise SystemExit(f"unexpected weight {spec.get('weight')!r}")
    if not MILL_SRC.is_file() or not COAT_SRC.is_file():
        raise SystemExit("missing mill or coatnet source kernel")

    DST.mkdir(exist_ok=True)
    (DST / "_mill_v10.py").write_text(MILL_SRC.read_text(encoding="utf-8"), encoding="utf-8")
    (DST / "_coatnet_stack.py").write_text(COAT_SRC.read_text(encoding="utf-8"), encoding="utf-8")
    (DST / "fusion.py").write_text(FUSION.read_text(encoding="utf-8"), encoding="utf-8")
    (DST / "overlay_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    (DST / "main.py").write_text(MAIN_PY, encoding="utf-8")
    compile(MAIN_PY, str(DST / "main.py"), "exec")

    mill_meta = json.loads(MILL_META.read_text(encoding="utf-8"))
    coat_meta = json.loads(COAT_META.read_text(encoding="utf-8"))
    meta = {
        "id": "vitaliecervinschi/rsna-knee-coatnet-mcl-eff-ov",
        "title": "RSNA Knee coatnet MCL+Effusion ov",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": _merge(
            mill_meta.get("dataset_sources", []),
            coat_meta.get("dataset_sources", []),
        ),
        "kernel_sources": _merge(
            mill_meta.get("kernel_sources", []),
            coat_meta.get("kernel_sources", []),
        ),
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": _merge(
            mill_meta.get("model_sources", []),
            coat_meta.get("model_sources", []),
        ),
        "docker_image": coat_meta["docker_image"],
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DST}")
    print(f"slug={meta['id']}")
    print(f"files={sorted(p.name for p in DST.iterdir())}")
    print(f"main_bytes={(DST / 'main.py').stat().st_size}")
    print(f"mill_bytes={(DST / '_mill_v10.py').stat().st_size}")
    print(f"coat_bytes={(DST / '_coatnet_stack.py').stat().st_size}")
    print(f"n_datasets={len(meta['dataset_sources'])}")


if __name__ == "__main__":
    main()