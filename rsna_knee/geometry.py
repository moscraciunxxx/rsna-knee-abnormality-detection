"""Physical MRI frame helpers. No DICOM reads and no Report text.

ImageOrientationPatient is six direction cosines: row, then column.
The slice normal is their cross product. LPS x is the patient's left.
"""

from __future__ import annotations

import numpy as np

def slice_normal(image_orientation: np.ndarray) -> np.ndarray:
    """Unit normal from a 6-vector ImageOrientationPatient."""
    iop = np.asarray(image_orientation, dtype=np.float64).reshape(-1)
    if iop.size != 6 or not np.isfinite(iop).all():
        raise ValueError("ImageOrientationPatient must be 6 finite numbers")
    normal = np.cross(iop[:3], iop[3:])
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("ImageOrientationPatient is degenerate")
    return normal / norm


def sort_along_normal(positions: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Indices that order slice positions along the increasing normal.

    ``positions`` is (n, 3) in the same frame as ``normal``. Ties keep input order.
    """
    pos = np.asarray(positions, dtype=np.float64)
    direction = np.asarray(normal, dtype=np.float64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3 or direction.size != 3:
        raise ValueError("positions must be (n, 3) and normal must be length 3")
    if not np.isfinite(pos).all() or not np.isfinite(direction).all():
        raise ValueError("positions and normal must be finite")
    projection = pos @ direction
    return np.argsort(projection, kind="mergesort")


def laterality_from_lps_x(x_mm: float) -> str:
    """Patient left is positive LPS x. Zero is unknown."""
    value = float(x_mm)
    if not np.isfinite(value) or value == 0.0:
        return "U"
    return "L" if value > 0.0 else "R"
