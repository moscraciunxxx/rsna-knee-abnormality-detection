"""Stream canonical volumes out of the competition zip. Never writes a DICOM.

One compressed npz per study under D:\\rsna_cache. Each array is a series
volume from the reader's preprocess.load_series path: uint8 [n, 384, 384].
"""

from __future__ import annotations

import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom

READER = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\provenance\reader_weights")
sys.path.insert(0, str(READER))
import preprocess  # noqa: E402

CACHE = Path(r"D:\rsna_cache")
ZIP_CANDIDATES = [
    Path(r"D:\rsna_data\rsna-knee-abnormality-detection.zip"),
    Path(r"D:\rsna_data\rsna-knee-abnormality-detection.tar.gz"),
]


def read_blob(blob: bytes):
    ds = pydicom.dcmread(io.BytesIO(blob))
    arr = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    slope, inter = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is not None and inter is not None:
        arr = arr * float(slope) + float(inter)
    iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    inst = float(ds.get("InstanceNumber") or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]


def load_series_bytes(blobs: list[bytes], plane_name: str | None = None):
    slices, bad = [], 0
    for blob in blobs:
        try:
            slices.extend(read_blob(blob))
        except Exception:
            bad += 1
    info = {"n_raw": len(slices), "n_bad": bad}
    if not slices:
        return np.zeros((1, preprocess.SIZE, preprocess.SIZE), np.uint8), info
    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
        plane = int(np.abs(normal).argmax())
    else:
        normal, plane = None, preprocess.PLANES.index(plane_name) if plane_name in preprocess.PLANES else 0
    info["plane"] = preprocess.PLANES[plane]
    if normal is not None and all(s[2] is not None for s in slices):
        sgn = preprocess.CANON[plane][0] * np.sign(normal[plane])
        slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
    else:
        slices.sort(key=lambda s: s[4])
    if len(slices) > preprocess.MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, preprocess.MAX_SLICES).round().astype(int)
        slices = [slices[i] for i in keep]
    vol = np.stack([
        preprocess._resample(*preprocess._canon_slice(s[0], s[1], s[3], plane))
        for s in slices
    ])
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    info["n"] = len(vol)
    return (vol * 255).round().astype(np.uint8), info


def series_key(name: str) -> tuple[str, str, str] | None:
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if "train_series" not in parts and "test_series" not in parts:
        return None
    if not name.lower().endswith(".dcm"):
        return None
    kind_at = parts.index("train_series") if "train_series" in parts else parts.index("test_series")
    if len(parts) < kind_at + 4:
        return None
    return parts[kind_at], parts[kind_at + 1], parts[kind_at + 2]


def groups(zf: zipfile.ZipFile) -> dict[tuple[str, str], dict[str, list[str]]]:
    out: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for info in zf.infolist():
        if info.is_dir():
            continue
        key = series_key(info.filename)
        if key is None:
            continue
        kind, study, series = key
        out[(kind, study)][series].append(info.filename)
    return out


def cache_study(zf: zipfile.ZipFile, kind: str, study: str, series_map: dict[str, list[str]], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / kind / f"{study}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for series, members in series_map.items():
        blobs = [zf.read(name) for name in members]
        vol, _info = load_series_bytes(blobs)
        arrays[series] = vol
    if path.exists():
        path.unlink()
    np.savez_compressed(path, **arrays)
    return path.stat().st_size


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    zpath = next((p for p in ZIP_CANDIDATES if p.is_file()), None)
    if zpath is None:
        raise SystemExit("zip is not on disk yet")
    with zipfile.ZipFile(zpath) as zf:
        grouped = groups(zf)
        if not grouped:
            raise SystemExit("no train_series or test_series DICOM members")
        picked = list(grouped.items())[:limit]
        sizes = []
        for (kind, study), series_map in picked:
            nbytes = cache_study(zf, kind, study, series_map, CACHE)
            sizes.append(nbytes)
            print(f"study {study} series {len(series_map)} mb {nbytes / 1e6:.2f}", flush=True)
    mean = sum(sizes) / len(sizes)
    print(f"mb_per_study {mean / 1e6:.2f} projected_4407_gb {mean * 4407 / 1e9:.1f}", flush=True)


if __name__ == "__main__":
    main()
