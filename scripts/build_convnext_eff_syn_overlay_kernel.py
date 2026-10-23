"""Write the ConvNeXt Effusion+Synovitis TEST kernel. Does not push or Submit.

56915024 was still PENDING when this builder was added. A later session may
Submit only after that publicScore is known and the kernel's own discovered-n
clock is under 8.75 h.
"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "kaggle_kernel_convnext_eff_syn_overlay"
SPEC = ROOT / "artifacts" / "convnext_mil870" / "overlay_spec.json"
MILL_SRC = ROOT / "kaggle_kernel_blend_script" / "main.py"
MILL_META = ROOT / "kaggle_kernel_blend_script" / "kernel-metadata.json"

EMBED = {
    "fusion.py": ROOT / "rsna_knee" / "fusion.py",
    "scoring_time.py": ROOT / "rsna_knee" / "scoring_time.py",
    "study_count.py": ROOT / "rsna_knee" / "study_count.py",
    "convnext_overlay_kernel.py": ROOT / "rsna_knee" / "convnext_overlay_kernel.py",
    "_mill_v10.py": MILL_SRC,
}

WRAPPER = '''#!/usr/bin/env python3
"""RSNA Knee TEST overlay. Unpack shipped modules, then run the clock check.

Do not competitions-submit a train-holdout file. Keep 55779066 unless a later
COMPLETE publicScore is strictly greater than 0.936.
"""
from __future__ import annotations

import base64
import sys
import zlib
from pathlib import Path

BLOBS = __BLOBS__


def _unpack() -> None:
    work = Path("/kaggle/working")
    work.mkdir(parents=True, exist_ok=True)
    for name, blob in BLOBS.items():
        (work / name).write_bytes(zlib.decompress(base64.b64decode(blob)))
    sys.path.insert(0, str(work))


if __name__ == "__main__":
    _unpack()
    from convnext_overlay_kernel import main
    main()
'''


def _blob(path: Path) -> str:
    return base64.b64encode(zlib.compress(path.read_bytes(), level=9)).decode("ascii")


def build_kernel(spec_path: Path = SPEC, dst: Path = DST) -> Path:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("decision") != "overlay":
        raise SystemExit(f"spec decision is {spec.get('decision')!r}")
    if set(spec.get("findings") or []) != {"Effusion", "Synovitis"}:
        raise SystemExit(f"unexpected findings {spec.get('findings')!r}")
    if abs(float(spec.get("weight", 0.0)) - 0.15) > 1e-9:
        raise SystemExit(f"unexpected weight {spec.get('weight')!r}")
    if spec.get("source") != "convnext-mil870-gate":
        raise SystemExit(f"unexpected source {spec.get('source')!r}")
    if spec.get("do_not_submit_holdout_ids") is not True:
        raise SystemExit("holdout ids must stay unsubmitted")
    if spec.get("keep") != "55779066":
        raise SystemExit(f"spec keep is {spec.get('keep')!r}")
    for path in EMBED.values():
        if not path.is_file():
            raise SystemExit(f"missing embed {path}")

    blobs = {name: _blob(path) for name, path in EMBED.items()}
    dst.mkdir(parents=True, exist_ok=True)
    wrapper = WRAPPER.replace("__BLOBS__", repr(blobs))
    (dst / "main.py").write_text(wrapper, encoding="utf-8")
    compile(wrapper, str(dst / "main.py"), "exec")
    (dst / "overlay_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    mill_meta = json.loads(MILL_META.read_text(encoding="utf-8"))
    meta = {
        "id": "vitaliecervinschi/rsna-knee-convnext-eff-syn-ov",
        "title": "RSNA Knee convnext Effusion Synovitis ov",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": list(mill_meta.get("dataset_sources", []))
        + ["goodpjw2008/rsna-knee-2-5d-convnext-reader"],
        "kernel_sources": list(mill_meta.get("kernel_sources", [])),
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": list(mill_meta.get("model_sources", [])),
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
        ),
    }
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dst}")
    print("DO NOT SUBMIT while 56915024 is PENDING. Do not push from this script.")
    print(f"slug={meta['id']}")
    print(f"main_bytes={(dst / 'main.py').stat().st_size}")
    return dst


def main() -> None:
    build_kernel()


if __name__ == "__main__":
    main()
