"""Write the seven-finding ConvNeXt TEST kernel. Does not push or Submit.

Push only after submission 56921398 is COMPLETE with publicScore >= 0.936.
Docker is the anchor image 37c64f. Kernel source is dinosaur-v5-train.
dinosaur-v3-train was rejected by Kaggle and must not be attached.
"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
DST = ROOT / "kaggle_kernel_convnext_k7_overlay"
BOOT = ROOT / "artifacts" / "convnext_oof870" / "bootstrap_k7.json"
MILL_SRC = ROOT / "kaggle_kernel_blend_script" / "main.py"
MILL_META = ROOT / "kaggle_kernel_blend_script" / "kernel-metadata.json"
ANCHOR_DOCKER = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
DINOSAUR_V5 = "romantamrazov/rsna-knee-dinosaur-v5-train"

EMBED = {
    "fusion.py": ROOT / "rsna_knee" / "fusion.py",
    "scoring_time.py": ROOT / "rsna_knee" / "scoring_time.py",
    "study_count.py": ROOT / "rsna_knee" / "study_count.py",
    "convnext_overlay_kernel.py": ROOT / "rsna_knee" / "convnext_overlay_kernel.py",
    "convnext_k7_overlay_kernel.py": ROOT / "rsna_knee" / "convnext_k7_overlay_kernel.py",
    "_mill_v10.py": MILL_SRC,
}

WRAPPER = '''#!/usr/bin/env python3
"""RSNA Knee seven-finding ConvNeXt TEST overlay. Does not read a report.

Keep 55779066 unless a later COMPLETE publicScore is strictly greater than 0.936.
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
    from convnext_k7_overlay_kernel import main
    main()
'''


def _blob(path: Path) -> str:
    return base64.b64encode(zlib.compress(path.read_bytes(), level=9)).decode("ascii")


def build_kernel(dst: Path = DST) -> Path:
    from rsna_knee.fusion import CONVNEXT_OOF_K7_FINDINGS, CONVNEXT_OOF_K7_WEIGHT

    boot = json.loads(BOOT.read_text(encoding="utf-8"))
    if boot.get("do_not_submit") is not True:
        raise SystemExit("bootstrap record must keep do_not_submit true until a scored run")
    seven = boot.get("seven") or {}
    if seven.get("clears_0.003") is not True:
        raise SystemExit("k7 macro CI does not clear +0.003")
    if set(seven.get("findings") or []) != set(CONVNEXT_OOF_K7_FINDINGS):
        raise SystemExit(f"bootstrap findings {seven.get('findings')!r} != k7 set")
    if abs(float(seven.get("weight", 0.0)) - CONVNEXT_OOF_K7_WEIGHT) > 1e-9:
        raise SystemExit("bootstrap weight is not 0.15")
    for path in EMBED.values():
        if not path.is_file():
            raise SystemExit(f"missing embed {path}")

    blobs = {name: _blob(path) for name, path in EMBED.items()}
    dst.mkdir(parents=True, exist_ok=True)
    wrapper = WRAPPER.replace("__BLOBS__", repr(blobs))
    (dst / "main.py").write_text(wrapper, encoding="utf-8")
    compile(wrapper, str(dst / "main.py"), "exec")

    mill_meta = json.loads(MILL_META.read_text(encoding="utf-8"))
    sources = [DINOSAUR_V5]
    if any("dinosaur-v3-train" in item for item in sources):
        raise SystemExit("dinosaur-v3-train was rejected")
    meta = {
        "id": "vitaliecervinschi/rsna-knee-convnext-k7-ov",
        "title": "RSNA Knee convnext k7 ov",
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
        "kernel_sources": sources,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": list(mill_meta.get("model_sources", [])),
        "docker_image": ANCHOR_DOCKER,
    }
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    spec = {
        "decision": "overlay",
        "keep": "55779066",
        "do_not_submit_holdout_ids": True,
        "test_submit": False,
        "source": "convnext-oof870-k7",
        "findings": list(CONVNEXT_OOF_K7_FINDINGS),
        "weight": CONVNEXT_OOF_K7_WEIGHT,
        "docker": ANCHOR_DOCKER,
        "kernel_source": DINOSAUR_V5,
        "push_only_if_56921398_public_score_gte": 0.936,
    }
    (dst / "overlay_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dst}")
    print("DO NOT PUSH until 56921398 publicScore >= 0.936. This script does not push.")
    return dst


def main() -> None:
    build_kernel()


if __name__ == "__main__":
    main()
