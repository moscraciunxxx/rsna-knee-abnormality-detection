"""ConvNeXt-nano student, 224 px, 24 slices, no TTA. Do not submit.

Study and series discovery, and DICOM reads, follow kaggle_kernel_clean_blend_t4.
A study that fails or has no usable series is filled with the report-label prior.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import timm

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"
PLANES = ["Sagittal", "Coronal", "Axial"]
N_SLOTS = 6
K = 24
RES = 224
SIZE = 384
MM_PER_PX = 0.4
MAX_SLICES = 64
CANON = {
    0: (+1, (1, +1), (2, -1)),
    1: (+1, (0, +1), (2, -1)),
    2: (-1, (0, +1), (1, +1)),
}
# Mean of steven v4 report labels on the 4,349 non-gold studies.
PRIOR = np.array(
    [0.232398, 0.178999, 0.459530, 0.223309, 0.406923, 0.325536,
     0.483248, 0.583739, 0.338169, 0.356775, 0.260040, 0.182033],
    dtype=np.float32,
)
WEIGHT = Path("/kaggle/input/rsna-knee-student-ckpt/student_resume224.pt")


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = timm.create_model("convnext_nano", pretrained=False, num_classes=0)
        self.head = nn.Linear(self.enc.num_features, 12)
        self.chunk = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        parts = [self.enc(chunk) for chunk in x.split(self.chunk)]
        return self.head(torch.cat(parts, 0).mean(0))


def find_test_root() -> Path:
    for base in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (base / "test.csv").is_file():
            return base
    root = Path("/kaggle/input")
    if root.is_dir():
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in {"train_series", "test_series"}]
            if "test.csv" in files:
                return Path(dirpath)
    raise FileNotFoundError("test.csv is not mounted")


def slot_of(plane, fat_sup):
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))


def build_study_table(series_df, n_slices):
    """Same slot table as the reader's knee.build_study_table."""
    table = {}
    for st, sr, fs, pl in zip(
        series_df.StudyInstanceUID,
        series_df.SeriesInstanceUID,
        series_df.Fat_Suppression,
        series_df.Anatomical_Plane,
    ):
        if n_slices.get(sr, 0) < 3:
            continue
        table.setdefault(st, [[] for _ in range(N_SLOTS)])[slot_of(pl, fs)].append(sr)
    for slots in table.values():
        for cands in slots:
            cands.sort(key=lambda series_id: -n_slices[series_id])
    return table


def _read(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    slope, intercept = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is not None and intercept is not None:
        arr = arr * float(slope) + float(intercept)
    iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    spacing = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    inst = float(ds.get("InstanceNumber") or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(frame, iop, ipp, spacing, inst + i * 1e-3) for i, frame in enumerate(frames)]


def _canon_slice(img, iop, spacing, plane):
    spacing = spacing or [MM_PER_PX * img.shape[0] / SIZE] * 2
    if iop is None:
        return img, spacing
    row, col = np.array(iop[:3]), np.array(iop[3:])
    _, (col_axis, col_sign), (row_axis, row_sign) = CANON[plane]
    if abs(row[col_axis]) < abs(row[row_axis]):
        img, row, col, spacing = img.T, col, row, spacing[::-1]
    if row[col_axis] * col_sign < 0:
        img = img[:, ::-1]
    if col[row_axis] * row_sign < 0:
        img = img[::-1]
    return img, spacing


def _resample(img, spacing):
    out = np.zeros((SIZE, SIZE), np.float32)
    height, width = img.shape
    crop_h = min(height, int(round(SIZE * MM_PER_PX / spacing[0])))
    crop_w = min(width, int(round(SIZE * MM_PER_PX / spacing[1])))
    y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
    crop = np.ascontiguousarray(img[y0:y0 + crop_h, x0:x0 + crop_w])
    new_h = max(1, min(SIZE, int(round(crop_h * spacing[0] / MM_PER_PX))))
    new_w = max(1, min(SIZE, int(round(crop_w * spacing[1] / MM_PER_PX))))
    interp = cv2.INTER_AREA if new_h < crop_h else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    oy, ox = (SIZE - new_h) // 2, (SIZE - new_w) // 2
    out[oy:oy + new_h, ox:ox + new_w] = resized
    return out


def load_series(series_dir, plane_name=None):
    """Reader load_series. Unreadable files, including truncated DICOMs, are skipped."""
    slices, bad = [], 0
    for name in os.listdir(series_dir):
        try:
            slices.extend(_read(os.path.join(series_dir, name)))
        except Exception:
            bad += 1
    if not slices:
        return np.zeros((1, SIZE, SIZE), np.uint8), {"n_raw": 0, "n_bad": bad}
    iop = next((item[1] for item in slices if item[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:])
        plane = int(np.abs(normal).argmax())
    else:
        normal, plane = None, PLANES.index(plane_name) if plane_name in PLANES else 0
    if normal is not None and all(item[2] is not None for item in slices):
        sign = CANON[plane][0] * np.sign(normal[plane])
        slices.sort(key=lambda item: float(np.dot(item[2], normal)) * sign)
    else:
        slices.sort(key=lambda item: item[4])
    if len(slices) > MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
        slices = [slices[int(i)] for i in keep]
    frames = []
    for image, orientation, _position, spacing, _inst in slices:
        try:
            frames.append(_resample(*_canon_slice(image, orientation, spacing, plane)))
        except Exception:
            continue
    if not frames:
        return np.zeros((0, SIZE, SIZE), np.uint8), {"n_raw": 0, "n_bad": bad + len(slices)}
    vol = np.stack(frames)
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (vol * 255).round().astype(np.uint8), {"n_raw": len(frames), "n_bad": bad}


def to_student(vol: np.ndarray) -> np.ndarray | None:
    """24 slices at 224. Fewer than 24 slices are kept as they are."""
    n = len(vol)
    if n < 1:
        return None
    if n < K:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, K).round().astype(int)
    picked = vol[idx]
    out = np.empty((len(picked), RES, RES), np.uint8)
    for i, image in enumerate(picked):
        if image.shape != (SIZE, SIZE):
            image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out[i] = cv2.resize(image, (RES, RES), interpolation=cv2.INTER_AREA)
    return np.repeat(out[:, None, :, :], 3, axis=1)


def study_images(study: str, slots, root: Path) -> np.ndarray | None:
    parts = []
    for cands in slots:
        if not cands:
            continue
        folder = root / str(study) / str(cands[0])
        if not folder.is_dir() or not any(folder.iterdir()):
            continue
        try:
            vol, info = load_series(str(folder))
        except Exception as exc:
            print(f"skip series {study} {cands[0]} {type(exc).__name__}: {exc}", flush=True)
            continue
        if info["n_raw"] < 1:
            continue
        images = to_student(vol)
        if images is not None:
            parts.append(images)
    if not parts:
        return None
    return np.concatenate(parts, 0)


def prior_row(study: str) -> dict:
    return {ID_COL: study, **{name: float(PRIOR[i]) for i, name in enumerate(LABELS)}}


def score_study(study: str, table, root: Path, model, device) -> tuple[dict, str | None]:
    slots = table.get(study, [[] for _ in range(N_SLOTS)])
    images = study_images(study, slots, root)
    if images is None:
        return prior_row(study), "missing series"
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        prob = torch.sigmoid(model(x).float()).cpu().numpy()
    if not np.isfinite(prob).all() or prob.shape != (12,):
        return prior_row(study), "non-finite score"
    return {ID_COL: study, **{name: float(prob[i]) for i, name in enumerate(LABELS)}}, None


def write_submission(comp: Path, weight: Path, out: Path) -> pd.DataFrame:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    if not weight.is_file():
        raise FileNotFoundError(f"missing checkpoint {weight}")
    ck = torch.load(weight, map_location="cpu", weights_only=False)
    print("checkpoint epoch", ck.get("epoch"), "k", ck.get("k"), "res", ck.get("res"), flush=True)
    model = Student()
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    studies = pd.read_csv(comp / "test.csv", dtype={ID_COL: str})[ID_COL].tolist()
    se = pd.read_csv(comp / "test_series.csv")
    se = se[se.StudyInstanceUID.isin(set(studies))]
    root = comp / "test_series"
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        folder = root / str(st) / str(sr)
        n_files[sr] = min(len(list(folder.iterdir())), MAX_SLICES) if folder.is_dir() else 0
    table = build_study_table(se, n_files)

    rows = []
    t_loop = time.time()
    with torch.no_grad():
        for study in studies:
            try:
                row, reason = score_study(study, table, root, model, device)
            except Exception as exc:
                row, reason = prior_row(study), f"{type(exc).__name__}: {exc}"
            if reason:
                print(f"prior {study} {reason}", flush=True)
            rows.append(row)
    print(f"seconds/study {(time.time() - t_loop) / max(len(studies), 1):.3f} n {len(studies)}", flush=True)
    frame = pd.DataFrame(rows)[[ID_COL, *LABELS]]
    if frame[LABELS].isna().any().any():
        raise RuntimeError("submission has NaNs")
    if frame[ID_COL].astype(str).tolist() != [str(study) for study in studies]:
        raise RuntimeError("submission IDs do not match the reader")
    if list(frame.columns) != [ID_COL, *LABELS] or len(frame) != len(studies):
        raise RuntimeError("submission shape does not match the reader")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote submission.csv rows {len(frame)} cols {frame.shape[1]} wall_s {time.time() - t0:.1f}", flush=True)
    return frame


def main() -> None:
    write_submission(find_test_root(), WEIGHT, Path("/kaggle/working/submission.csv"))


if __name__ == "__main__":
    main()
