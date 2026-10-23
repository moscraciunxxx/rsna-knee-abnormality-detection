"""One DICOM header per train series. CPU only. Do not submit.

Reads pixels nowhere. One file per series, pydicom stop_before_pixels.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pydicom

FIELDS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "SoftwareVersions",
    "ReceiveCoilName",
    "PatientSex",
    "PatientAge",
    "Laterality",
    "SeriesDescription",
    "PatientID",
]


def find_comp() -> Path:
    for base in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (base / "train_series.csv").is_file():
            return base
    root = Path("/kaggle/input")
    if root.is_dir():
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in {"train_series", "test_series"}]
            if "train_series.csv" in files:
                return Path(dirpath)
    raise FileNotFoundError("train_series.csv is not mounted")


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(item) for item in value)
    return str(value).strip()


def header_row(study: str, series: str, path: Path | None) -> dict:
    row = {name: "" for name in FIELDS}
    row["StudyInstanceUID"] = study
    row["SeriesInstanceUID"] = series
    if path is None:
        return row
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    for name in FIELDS[2:]:
        if name == "Laterality" and "Laterality" not in ds and "ImageLaterality" in ds:
            row[name] = as_text(ds.get("ImageLaterality"))
            continue
        if name not in ds:
            continue
        value = ds.get(name)
        if name == "MagneticFieldStrength":
            try:
                row[name] = f"{float(value):g}"
            except (TypeError, ValueError):
                row[name] = as_text(value)
        else:
            row[name] = as_text(value)
    return row


def first_file(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    files = sorted(path for path in folder.iterdir() if path.is_file())
    return files[0] if files else None


def main() -> None:
    comp = find_comp()
    series = pd.read_csv(comp / "train_series.csv", dtype=str)
    root = comp / "train_series"
    rows = []
    for study, series_id in zip(series.StudyInstanceUID, series.SeriesInstanceUID):
        folder = root / str(study) / str(series_id)
        try:
            rows.append(header_row(str(study), str(series_id), first_file(folder)))
        except Exception as exc:
            row = header_row(str(study), str(series_id), None)
            row["SeriesDescription"] = f"READ_FAILED {type(exc).__name__}"
            rows.append(row)
    frame = pd.DataFrame(rows)[FIELDS]
    if frame.SeriesInstanceUID.duplicated().any():
        raise RuntimeError("duplicate series in train_headers.csv")
    if frame.SeriesInstanceUID.astype(str).tolist() != series.SeriesInstanceUID.astype(str).tolist():
        raise RuntimeError("header rows do not match train_series.csv")
    out = Path("/kaggle/working/train_headers.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out} rows {len(frame)} cols {frame.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
