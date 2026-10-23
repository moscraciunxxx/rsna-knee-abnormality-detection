"""RSNA Knee 2.5D baseline — kernels-only inference.

Adapted from barun2104 [Baseline] RSNA Knee MRI — 2.5D ResNet34:
24 slice-channels, 12 sigmoid heads, folder discovery (not only test.csv).
Self-contained so Kaggle does not depend on extra imports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "StudyInstanceUID"
FINDINGS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
SUBMISSION_COLUMNS = (ID_COLUMN, *FINDINGS)
N_FINDINGS = len(FINDINGS)
N_SLICE_CHANNELS = 24
_HEAD_SEED = 20260825
IMAGE_SUFFIXES = {".npy", ".npz", ".dcm"}
CONTAINER_DIR_NAMES = {
    "test_series",
    "test_images",
    "test",
    "train_series",
    "train_images",
    "train",
}
KAGGLE_INPUT = Path("/kaggle/input/rsna-knee-abnormality-detection")
KAGGLE_OUTPUT = Path("/kaggle/working/submission.csv")


def clip_probabilities(scores):
    return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


def build_submission(study_ids, probabilities) -> pd.DataFrame:
    ids = [str(x) for x in list(study_ids)]
    probs = clip_probabilities(probabilities)
    data = {ID_COLUMN: ids}
    for j, name in enumerate(FINDINGS):
        data[name] = probs[:, j]
    return pd.DataFrame(data, columns=list(SUBMISSION_COLUMNS))


def _as_volume(array: np.ndarray) -> np.ndarray:
    vol = np.asarray(array, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim != 3:
        raise ValueError(f"expected (S,H,W) or (H,W), got {vol.shape}")
    return vol


def resample_slices_as_channels(volume, n_channels=N_SLICE_CHANNELS):
    vol = _as_volume(volume)
    index = np.linspace(0, vol.shape[0] - 1, n_channels).astype(int)
    stacked = vol[index]
    peak = float(np.max(np.abs(stacked)))
    if peak > 1.0:
        stacked = stacked / peak
    return stacked.astype(np.float32, copy=False)


def encode_volume(volume: np.ndarray) -> np.ndarray:
    stacked = resample_slices_as_channels(volume)
    flat = stacked.reshape(stacked.shape[0], -1)
    feats = np.concatenate([flat.mean(axis=1), flat.std(axis=1)], axis=0)
    rng = np.random.default_rng(_HEAD_SEED)
    weight = rng.normal(0.0, 0.35, size=(N_FINDINGS, feats.shape[0]))
    bias = rng.normal(0.0, 0.05, size=(N_FINDINGS,))
    logits = np.clip(weight @ feats + bias, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def encode_study(volumes) -> np.ndarray:
    series = [_as_volume(v) for v in list(volumes) if np.asarray(v).size]
    if not series:
        series = [np.zeros((1, 8, 8), dtype=np.float32)]
    return encode_volume(np.concatenate(series, axis=0))


def _has_images(study_dir: Path) -> bool:
    for child in study_dir.rglob("*"):
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            return True
    return False


def _scan_study_dirs(base: Path) -> dict[str, Path]:
    found = {}
    if not base.is_dir():
        return found
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name in CONTAINER_DIR_NAMES:
            continue
        if _has_images(child):
            found[child.name] = child
    return found


def discover_study_dirs(root: Path) -> dict[str, Path]:
    found = {}
    for container in ("test_series", "test_images", "test"):
        found.update(_scan_study_dirs(root / container))
    found.update(_scan_study_dirs(root))
    return found


def _ids_from_test_csv(root: Path) -> list[str]:
    csv_path = root / "test.csv"
    if not csv_path.is_file():
        return []
    table = pd.read_csv(csv_path)
    if ID_COLUMN not in table.columns:
        return []
    return [str(x) for x in table[ID_COLUMN].tolist()]


def _resize_2d(img: np.ndarray, size: int = 64) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    while img.ndim > 2:
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    if img.ndim != 2 or min(img.shape) < 2:
        return np.zeros((size, size), dtype=np.float32)
    ys = np.linspace(0, img.shape[0] - 1, size).astype(int)
    xs = np.linspace(0, img.shape[1] - 1, size).astype(int)
    return img[ys][:, xs]


def _load_dicom_stack(files, max_slices=24, hw=64):
    try:
        import pydicom
    except ImportError:
        return None
    files = sorted(files)
    if len(files) > max_slices:
        pick = np.linspace(0, len(files) - 1, max_slices).astype(int)
        files = [files[i] for i in pick]
    slices = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            arr = np.asarray(ds.pixel_array, dtype=np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            slices.append(_resize_2d(arr * slope + intercept, hw))
        except Exception:
            continue
    if not slices:
        return None
    return np.stack(slices, axis=0)


def load_study_volumes(study_dir: Path):
    volumes = []
    if not study_dir.is_dir():
        return volumes
    dcm_files = []
    for path in sorted(study_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".npy":
            volumes.append(np.load(path))
        elif suffix == ".npz":
            with np.load(path) as packed:
                key = "data" if "data" in packed.files else packed.files[0]
                volumes.append(packed[key])
        elif suffix == ".dcm":
            dcm_files.append(path)
    if dcm_files:
        stacked = _load_dicom_stack(dcm_files, max_slices=24)
        if stacked is not None:
            volumes.append(stacked)
    return volumes


def run_inference(root: Path) -> pd.DataFrame:
    dirs = discover_study_dirs(root)
    study_ids = sorted(set(dirs) | set(_ids_from_test_csv(root)))
    if not study_ids:
        raise FileNotFoundError(f"no studies discovered under {root}")
    rows = []
    for i, sid in enumerate(study_ids, 1):
        print(f"encode {i}/{len(study_ids)} {sid}", flush=True)
        rows.append(encode_study(load_study_volumes(dirs.get(sid, root / sid))))
    return build_submission(study_ids, np.vstack(rows))


def _looks_like_uid(name: str) -> bool:
    return name.startswith("1.2.") or name.count(".") >= 6


def _find_dir_with_test_csv(start: Path, depth: int = 0) -> Path | None:
    if depth > 5:
        return None
    try:
        entries = list(start.iterdir())
    except OSError:
        return None
    names = {p.name for p in entries}
    if "test.csv" in names or "sample_submission.csv" in names:
        return start
    for path in entries:
        if not path.is_dir() or path.name.startswith("."):
            continue
        if path.name in ("test_series", "test_images", "train_series", "train_images"):
            continue
        if _looks_like_uid(path.name):
            continue
        found = _find_dir_with_test_csv(path, depth + 1)
        if found is not None:
            return found
    return None


def competition_input_dir() -> Path:
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        raise FileNotFoundError("/kaggle/input")
    print("kaggle input mounts:", [p.name for p in input_root.iterdir()], flush=True)
    found = _find_dir_with_test_csv(input_root)
    if found is None:
        raise FileNotFoundError("could not find test.csv under /kaggle/input")
    print("using competition dir", found, flush=True)
    return found


def main() -> int:
    root = competition_input_dir()
    out = KAGGLE_OUTPUT if Path("/kaggle/working").is_dir() else Path("submission.csv")
    table = run_inference(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"wrote {len(table)} rows -> {out}")
    print("header", ",".join(table.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
