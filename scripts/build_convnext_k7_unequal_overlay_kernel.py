"""Write the unequal-weight ConvNeXt TEST kernel. Does not push or Submit."""

from __future__ import annotations

import base64
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DST = ROOT / "kaggle_kernel_convnext_k7_unequal_overlay"
MEASURED = ROOT / "artifacts" / "convnext_oof870" / "k7_unequal_strict.json"
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
    "convnext_k7_unequal_overlay_kernel.py": ROOT / "rsna_knee" / "convnext_k7_unequal_overlay_kernel.py",
    "_mill_v10.py": MILL_SRC,
}

WRAPPER = '''#!/usr/bin/env python3
"""RSNA Knee unequal ConvNeXt TEST overlay. Does not read a report.

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
    from convnext_k7_unequal_overlay_kernel import main
    main()
'''


def _blob(path: Path) -> str:
    return base64.b64encode(zlib.compress(path.read_bytes(), level=9)).decode("ascii")


def build_kernel(dst: Path = DST) -> Path:
    from rsna_knee.fusion import convnext_oof_unequal_weights

    measured = json.loads(MEASURED.read_text(encoding="utf-8"))
    weights = convnext_oof_unequal_weights()
    if measured.get("weights") != weights:
        raise SystemExit("shipped unequal weights drifted from the 517-study search")
    if measured.get("do_not_submit") is not True:
        raise SystemExit("search record must keep do_not_submit true")
    if float(measured.get("macro_lo", 0.0)) <= 0.0:
        raise SystemExit("unequal macro CI lower bound must be > 0")
    for name, value in weights.items():
        if float(value) >= 0.50:
            raise SystemExit(f"{name} weight {value} is not under 0.50")
    for path in EMBED.values():
        if not path.is_file():
            raise SystemExit(f"missing embed {path}")
    blobs = {name: _blob(path) for name, path in EMBED.items()}
    dst.mkdir(parents=True, exist_ok=True)
    wrapper = WRAPPER.replace("__BLOBS__", repr(blobs))
    (dst / "main.py").write_text(wrapper, encoding="utf-8")
    compile(wrapper, str(dst / "main.py"), "exec")
    mill_meta = json.loads(MILL_META.read_text(encoding="utf-8"))
    meta = {
        "id": "vitaliecervinschi/rsna-knee-convnext-k7-unequal-ov",
        "title": "RSNA Knee convnext k7 unequal ov",
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
        "kernel_sources": [DINOSAUR_V5],
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
        "source": "convnext-oof870-unequal-strict",
        "weights": weights,
        "dropped": ["Lateral Meniscus", "Lateral OA", "PF OA", "Contusion", "Fracture"],
        "docker": ANCHOR_DOCKER,
        "kernel_source": DINOSAUR_V5,
    }
    (dst / "overlay_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dst}")
    print("DO NOT PUSH from this script.")
    return dst


def main() -> None:
    build_kernel()


if __name__ == "__main__":
    main()
