"""Write the 8-study RadImageNet-DenseNet121 timing probe. Does not Submit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "kaggle_kernel_densenet121_probe"

RUNNER = r'''
# --- 8-study timing probe. Do not competitions-submit. Keep 55779066. ---
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
print("nvidia-smi:", subprocess.getoutput("nvidia-smi -L"), flush=True)
print(
    "[dn121-probe] 8-study frozen RadImageNet-DenseNet121 timing probe. "
    "New 12-way head. Gold-58 out of the loss. DO NOT SUBMIT. keep 55779066.",
    flush=True,
)

HARD_LIMIT_S = 1500


def _watchdog() -> None:
    time.sleep(HARD_LIMIT_S)
    print(f"[dn121-probe] WATCHDOG {HARD_LIMIT_S}s", flush=True)
    os._exit(99)


threading.Thread(target=_watchdog, daemon=True, name="dn121-watchdog").start()

FINDINGS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)


def _comp_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for path in candidates:
        if (path / "train.csv").is_file() and (path / "train_series").is_dir():
            return path
    base = Path("/kaggle/input")
    for path in base.rglob("train.csv"):
        if (path.parent / "train_series").is_dir():
            return path.parent
    raise FileNotFoundError("competition train_series not mounted")


def _find_h5() -> Path:
    base = Path("/kaggle/input")
    skip = {"train_series", "test_series", "train_images", "test_images"}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [name for name in dirnames if name not in skip and not name.startswith("1.2.")]
        if "RadImageNet-DenseNet121_notop.h5" in filenames:
            return Path(dirpath) / "RadImageNet-DenseNet121_notop.h5"
    raise FileNotFoundError("RadImageNet-DenseNet121_notop.h5")


def _study_batch(root: Path, uid: str, k: int = IMAGES_PER_STUDY) -> torch.Tensor | None:
    folder = root / "train_series" / uid
    files = sorted(path for path in folder.rglob("*") if path.suffix.lower() == ".dcm")
    if not files:
        return None
    if len(files) > k:
        pick = np.linspace(0, len(files) - 1, k).astype(int)
        files = [files[int(i)] for i in pick]
    images = []
    for path in files:
        array = np.asarray(pydicom.dcmread(str(path), force=True).pixel_array, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        low, high = np.percentile(array, [1, 99])
        scale = max(float(high - low), 1e-6)
        array = np.clip((array - float(low)) / scale, 0.0, 1.0).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        images.append(tensor.repeat(1, 3, 1, 1))
    if not images:
        return None
    return torch.cat(images, dim=0)


def main() -> None:
    comp = _comp_root()
    train = pd.read_csv(comp / "train.csv", usecols=["StudyInstanceUID", *FINDINGS])
    ids = probe_study_ids(train, n=PROBE_STUDIES, seed=2026)
    gold = set(expert_gold_ids(train))
    print(f"[dn121-probe] n={len(ids)} gold_in_probe={len(set(ids) & gold)}", flush=True)
    if set(ids) & gold:
        raise SystemExit("[dn121-probe] gold leaked into the probe sample")
    model, report = load_radimagenet_densenet121(_find_h5())
    print(
        f"[dn121-probe] loaded n={report['n_loaded']} unexpected={len(report['unexpected'])} "
        f"frozen={report['features_frozen']} head={report['classifier_out']}",
        flush=True,
    )
    if report["unexpected"] or not report["features_frozen"] or report["classifier_out"] != 12:
        raise SystemExit("[dn121-probe] weight load failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    started = time.perf_counter()
    n_images = 0
    for uid in ids:
        batch = _study_batch(comp, uid)
        if batch is None:
            print(f"[dn121-probe] no slices {uid}", flush=True)
            continue
        with torch.no_grad():
            model.features(batch.to(device).float())
        n_images += int(batch.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if n_images <= 0:
        raise SystemExit("[dn121-probe] encoded 0 images")
    projected = projected_encode_seconds(elapsed, n_images)
    print(
        f"[dn121-probe] images={n_images} elapsed_s={elapsed:.3f} "
        f"projected_s={projected:.1f} abort_limit_s={ENCODE_ABORT_S:.1f}",
        flush=True,
    )
    sub = Path("/kaggle/working/submission.csv")
    if sub.is_file():
        sub.unlink()
    if encode_probe_aborts(elapsed, n_images):
        print("[dn121-probe] ABORT projected encode > 2.8h. No full train. DO NOT SUBMIT.", flush=True)
        raise SystemExit(2)
    print(
        "[dn121-probe] PROBE_OK projected encode <= 2.8h. "
        "gold_macro=not_fit gold_in_loss=0. Full train not started. DO NOT SUBMIT. keep 55779066.",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def _embed(path: Path) -> str:
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("from __future__") or line.startswith("from rsna_knee"):
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def build_kernel(dst: Path = DST) -> Path:
    loader = _embed(ROOT / "rsna_knee" / "radimagenet_densenet.py")
    probe = _embed(ROOT / "rsna_knee" / "densenet_probe.py")
    header = '''"""8-study RadImageNet-DenseNet121 timing probe. Do not Submit.

Gold-58 stay out of the sample. Abort when the projected 4407-study encode
is above 2.8 h. A new 12-way head is created and not trained here.
"""
from __future__ import annotations

ID_COLUMN = "StudyInstanceUID"
FINDINGS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)


def expert_gold_ids(train):
    frame = train.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    mask = frame[list(FINDINGS)].notna().all(axis=1)
    ids = tuple(frame.loc[mask, ID_COLUMN].tolist())
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate gold StudyInstanceUID")
    return ids

'''
    text = header + "\n" + loader + "\n" + probe + "\n" + RUNNER
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "main.py").write_text(text, encoding="utf-8")
    compile(text, str(dst / "main.py"), "exec")
    meta = {
        "id": "vitaliecervinschi/rsna-knee-densenet121-8-study-probe",
        "title": "RSNA Knee densenet121 8-study probe",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["sjudal/radimagenet-densenet121"],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
    }
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dst}")
    print("DO NOT SUBMIT. Timing probe only.")
    return dst


if __name__ == "__main__":
    build_kernel()
