"""Cached volumes in the reader's preprocess format.

One compressed file per study: ``{StudyInstanceUID}.npz``. Each array is one
series, uint8 ``[n, 384, 384]``, ``1 <= n <= 64``, produced by
``preprocess.load_series`` (0.4 mm pixels, 153.6 mm field of view). This is
not the CoAtNet 96-slice, 5-slot, 140 mm stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMG = 384
MAX_SLICES = 64


def vol_path(cache_dir: Path, study_id: str) -> Path:
    return Path(cache_dir) / f"{study_id}.npz"


def save_study(cache_dir: Path, study_id: str, series: dict[str, np.ndarray]) -> Path:
    """Write one study. ``series`` maps SeriesInstanceUID to a uint8 volume."""
    if not series:
        raise ValueError("study has no series")
    for name, vol in series.items():
        if vol.dtype != np.uint8:
            raise ValueError(f"{name} dtype {vol.dtype} is not uint8")
        if vol.ndim != 3 or vol.shape[1:] != (IMG, IMG):
            raise ValueError(f"{name} shape {vol.shape} is not (n, {IMG}, {IMG})")
        if not 1 <= vol.shape[0] <= MAX_SLICES:
            raise ValueError(f"{name} has {vol.shape[0]} slices")
    path = vol_path(cache_dir, study_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **series)
    return path


def load_study(cache_dir: Path, study_id: str) -> dict[str, np.ndarray]:
    """Return series uid -> uint8 volume."""
    with np.load(vol_path(cache_dir, study_id)) as blob:
        series = {name: blob[name] for name in blob.files}
    for name, vol in series.items():
        if vol.dtype != np.uint8 or vol.ndim != 3 or vol.shape[1:] != (IMG, IMG):
            raise ValueError(f"cached {name} is {vol.dtype} {vol.shape}")
    return series
