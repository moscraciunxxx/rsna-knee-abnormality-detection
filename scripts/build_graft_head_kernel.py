"""Write the CPU graft-label kernel. Does not push and does not Submit."""

from __future__ import annotations

import base64
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DST = ROOT / "kaggle_kernel_graft_head"
ANCHOR_DOCKER = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
EMBED = {
    "schema.py": ROOT / "rsna_knee" / "schema.py",
    "metrics.py": ROOT / "rsna_knee" / "metrics.py",
    "v4_split.py": ROOT / "rsna_knee" / "v4_split.py",
    "soft_labels.py": ROOT / "rsna_knee" / "soft_labels.py",
    "graft_head_export.py": ROOT / "rsna_knee" / "graft_head_export.py",
    "graft_head_kernel.py": ROOT / "rsna_knee" / "graft_head_kernel.py",
}

WRAPPER = '''#!/usr/bin/env python3
"""CPU export of grafted fit targets. GPU stays off. DO NOT SUBMIT.

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
    from graft_head_kernel import main
    main()
'''


def _blob(path: Path) -> str:
    return base64.b64encode(zlib.compress(path.read_bytes(), level=9)).decode("ascii")


def build_kernel(dst: Path = DST) -> Path:
    for path in EMBED.values():
        if not path.is_file():
            raise SystemExit(f"missing embed {path}")
    blobs = {name: _blob(path) for name, path in EMBED.items()}
    dst.mkdir(parents=True, exist_ok=True)
    wrapper = WRAPPER.replace("__BLOBS__", repr(blobs))
    (dst / "main.py").write_text(wrapper, encoding="utf-8")
    compile(wrapper, str(dst / "main.py"), "exec")
    meta = {
        "id": "vitaliecervinschi/rsna-knee-graft-head-labels",
        "title": "RSNA Knee graft head labels",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "stevenleehans/rsna-knee-llm-report-labels",
            "pilkwang/rsna-knee-llm-labels",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": ANCHOR_DOCKER,
    }
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    spec = {
        "decision": "package",
        "keep": "55779066",
        "do_not_submit_holdout_ids": True,
        "test_submit": False,
        "enable_gpu": False,
        "push": False,
    }
    (dst / "overlay_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return dst


if __name__ == "__main__":
    print(build_kernel())
