"""Generate 5 CPU-only cached-slice kernels, one per fold. Local only, no push.

Kernel f reads competition ``train_series`` DICOMs for its fold's studies with
the reader's ``preprocess.load_series`` (384 px, 0.4 mm, at most 64 slices
per series) and writes ``vols/{StudyInstanceUID}.npz``.
A later training kernel attaches these outputs as data sources.

Each kernel is one self-contained ``main.py`` (folds table + fast DICOM
reader embedded), CPU-only, internet off, docker 37c64f.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

MAIN_TEMPLATE = '''"""Cache fold-{fold} slice volumes. CPU-only, internet off, not a submission."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

FOLD = {fold}
FOLDS_B64 = "{folds_b64}"

IMG = 384


def log_versions() -> None:
    import sys
    print("python", sys.version.replace("\\n", " "), flush=True)
    try:
        import torch
        print("torch", torch.__version__, flush=True)
    except ImportError:
        print("torch missing", flush=True)
    try:
        import timm
        print("timm", timm.__version__, flush=True)
    except ImportError:
        print("timm missing", flush=True)


{preprocess}


def main() -> None:
    import base64
    import io
    log_versions()
    t0 = time.time()
    folds = pd.read_csv(
        io.BytesIO(base64.b64decode(FOLDS_B64)), dtype={{"StudyInstanceUID": str}}
    )
    ids = folds.loc[folds["fold"] == FOLD, "StudyInstanceUID"].astype(str).tolist()
    comp = None
    for base in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (base / "train.csv").is_file():
            comp = base
            break
    if comp is None:
        raise FileNotFoundError("competition train.csv is not mounted")
    images = comp / "train_series"
    if not images.is_dir():
        raise FileNotFoundError("competition train_series is not mounted")
    series = pd.read_csv(comp / "train_series.csv", dtype=str)
    ser = {{}}
    for row in series.to_dict("records"):
        ser.setdefault(row["StudyInstanceUID"], []).append(row)
    out = Path("/kaggle/working")
    vols = out / "vols"
    vols.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, sid in enumerate(ids):
        arrays = {{}}
        for row in ser.get(sid, []):
            sdir = images / sid / row["SeriesInstanceUID"]
            if not sdir.is_dir():
                continue
            vol, info = load_series(str(sdir), row.get("Anatomical_Plane"))
            if vol.ndim != 3 or vol.shape[1:] != (IMG, IMG) or vol.dtype != np.uint8:
                raise RuntimeError(f"{{sid}} {{row['SeriesInstanceUID']}} {{vol.shape}} {{vol.dtype}}")
            arrays[row["SeriesInstanceUID"]] = vol
        if not arrays:
            raise RuntimeError(f"{{sid}} produced no series")
        np.savez_compressed(vols / f"{{sid}}.npz", **arrays)
        manifest.append({{"StudyInstanceUID": sid, "n_series": len(arrays)}})
        if (i + 1) % 50 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"{{i + 1}}/{{len(ids)}} s/study {{rate:.2f}}", flush=True)
    with open(out / "manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["StudyInstanceUID", "n_series"])
        writer.writeheader()
        writer.writerows(manifest)
    timing = {{"fold": FOLD, "n": len(ids), "seconds": round(time.time() - t0, 1)}}
    (out / "timing.json").write_text(json.dumps(timing))
    print("timing", json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
'''


def build() -> list[Path]:
    folds_b64 = base64.b64encode((ROOT / "data" / "folds.csv").read_bytes()).decode("ascii")
    preprocess = (ROOT / "artifacts" / "provenance" / "reader_weights" / "preprocess.py").read_text(encoding="utf-8")
    cut = preprocess.find("\ndef _job(")
    if cut > 0:
        preprocess = preprocess[:cut]
    made = []
    for fold in range(5):
        kdir = ROOT / f"kaggle_kernel_cache_f{fold}"
        kdir.mkdir(exist_ok=True)
        (kdir / "main.py").write_text(
            MAIN_TEMPLATE.format(fold=fold, folds_b64=folds_b64, preprocess=preprocess),
            encoding="utf-8",
        )
        (kdir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": f"vitaliecervinschi/rsna-knee-cache-f{fold}",
                    "title": f"RSNA Knee cache f{fold} no-OAI",
                    "code_file": "main.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": True,
                    "enable_gpu": False,
                    "enable_tpu": False,
                    "enable_internet": False,
                    "keywords": ["cpu"],
                    "dataset_sources": [],
                    "kernel_sources": [],
                    "competition_sources": ["rsna-knee-abnormality-detection"],
                    "model_sources": [],
                    "docker_image": DOCKER,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        made.append(kdir)
    return made


if __name__ == "__main__":
    for path in build():
        print("wrote", path)
