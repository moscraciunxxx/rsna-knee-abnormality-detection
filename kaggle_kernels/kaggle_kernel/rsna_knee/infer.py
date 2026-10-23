"""Smoke / notebook inference: discover studies from folders, emit 12 probs.

Hidden test sets in kernels-only competitions swap in extra study folders.
Never trust test.csv alone and never hard-code the three public StudyInstanceUIDs.
Reports are train-only (leakage if used at test). Imaging path is required.

Official Kaggle layout (downloaded file listing):
  {comp}/test.csv
  {comp}/test_series.csv
  {comp}/test_series/{StudyInstanceUID}/{SeriesInstanceUID}/{SOPInstanceUID}.dcm
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.encoder import encode_study
from rsna_knee.schema import ID_COLUMN
from rsna_knee.submission import build_submission, write_submission

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


def _has_images(study_dir: Path) -> bool:
    for child in study_dir.rglob("*"):
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            return True
    return False


def _scan_study_dirs(base: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
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


def discover_study_dirs(root: str | Path) -> dict[str, Path]:
    """Map StudyInstanceUID → folder, scanning test_series/ then the root."""
    root = Path(root)
    found: dict[str, Path] = {}
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


def discover_study_ids(root: str | Path) -> list[str]:
    """Union of study folders on disk and optional test.csv (folders always kept)."""
    root = Path(root)
    from_folders = list(discover_study_dirs(root))
    from_csv = _ids_from_test_csv(root)
    return sorted(set(from_folders) | set(from_csv))


def _resize_2d(img: np.ndarray, size: int = 64) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    while img.ndim > 2:
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    if img.ndim != 2 or min(img.shape) < 2:
        return np.zeros((size, size), dtype=np.float32)
    ys = np.linspace(0, img.shape[0] - 1, size).astype(int)
    xs = np.linspace(0, img.shape[1] - 1, size).astype(int)
    return img[ys][:, xs]


def _load_dicom_stack(files: list[Path], max_slices: int = 24, hw: int = 64) -> np.ndarray | None:
    try:
        import pydicom
    except ImportError:
        return None
    files = sorted(files)
    if len(files) > max_slices:
        pick = np.linspace(0, len(files) - 1, max_slices).astype(int)
        files = [files[i] for i in pick]
    slices: list[np.ndarray] = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            arr = np.asarray(ds.pixel_array, dtype=np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            arr = arr * slope + intercept
            slices.append(_resize_2d(arr, hw))
        except Exception:
            continue
    if not slices:
        return None
    return np.stack(slices, axis=0)


def load_study_volumes(study_dir: str | Path) -> list[np.ndarray]:
    study_dir = Path(study_dir)
    volumes: list[np.ndarray] = []
    if not study_dir.is_dir():
        return volumes
    dcm_by_series: dict[Path, list[Path]] = defaultdict(list)
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
            dcm_by_series[path.parent].append(path)
    for files in dcm_by_series.values():
        stacked = _load_dicom_stack(files)
        if stacked is not None:
            volumes.append(stacked)
    return volumes


def run_inference(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    dirs = discover_study_dirs(root)
    study_ids = sorted(set(dirs) | set(_ids_from_test_csv(root)))
    if not study_ids:
        raise FileNotFoundError(f"no studies discovered under {root}")
    rows = []
    for study_id in study_ids:
        volumes = load_study_volumes(dirs.get(study_id, root / study_id))
        rows.append(encode_study(volumes))
    return build_submission(study_ids, np.vstack(rows))


def write_inference_csv(root: str | Path, output: str | Path) -> Path:
    return write_submission(run_inference(root), output)


def competition_input_dir() -> Path:
    """Kaggle input mount, else a local data/ folder with test.csv."""
    if KAGGLE_INPUT.is_dir():
        return KAGGLE_INPUT
    here = Path(__file__).resolve().parent.parent
    local = here / "data"
    if local.is_dir() and (
        (local / "test.csv").is_file()
        or (local / "test_series").is_dir()
        or (local / "sample_submission.csv").is_file()
    ):
        return local
    raise FileNotFoundError(
        "competition input not found (expected /kaggle/input/rsna-knee-abnormality-detection)"
    )


def make_smoke_fixture(root: str | Path, n_studies: int = 4, seed: int = 0) -> Path:
    """Tiny synthetic volumes — enough for a local smoke without DICOM."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [f"1.2.826.0.1.3680043.8.498.smoke{i:02d}" for i in range(n_studies)]
    for i, study_id in enumerate(ids):
        for series in ("axial", "sagittal"):
            dest = root / study_id / series
            dest.mkdir(parents=True, exist_ok=True)
            volume = rng.random((8 + i, 16, 16)).astype(np.float32) * (0.3 + 0.2 * i)
            np.save(dest / "stack.npy", volume)
    pd.DataFrame({ID_COLUMN: ids[: max(3, n_studies - 1)]}).to_csv(
        root / "test.csv", index=False
    )
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="2.5D RSNA knee inference → submission.csv"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Competition directory (default: Kaggle input mount)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination submission CSV (default: /kaggle/working/submission.csv or ./submission.csv)",
    )
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="Create synthetic studies under --input before running",
    )
    parser.add_argument("--n-studies", type=int, default=4)
    args = parser.parse_args(argv)
    if args.write_fixture:
        if args.input is None:
            raise SystemExit("--input is required with --write-fixture")
        input_dir = Path(args.input)
        make_smoke_fixture(input_dir, n_studies=args.n_studies)
    else:
        input_dir = Path(args.input) if args.input else competition_input_dir()
    if args.output:
        output = Path(args.output)
    elif Path("/kaggle/working").is_dir():
        output = KAGGLE_OUTPUT
    else:
        output = Path("submission.csv")
    out = write_inference_csv(input_dir, output)
    table = pd.read_csv(out)
    print(f"wrote {len(table)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
