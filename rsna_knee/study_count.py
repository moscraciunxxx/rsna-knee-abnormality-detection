"""Count hidden-test studies from folders and test.csv.

Same union as ``rsna_knee.infer.discover_study_ids``, without importing the
encoder. Kernels embed this file so the clock check can run before inference.
"""

from __future__ import annotations

import csv
from pathlib import Path

IMAGE_SUFFIXES = {".npy", ".npz", ".dcm"}
CONTAINER_DIR_NAMES = {"test_series", "test_images", "test", "train_series"}
ID_COLUMN = "StudyInstanceUID"


def _has_images(study_dir: Path) -> bool:
    for child in study_dir.rglob("*"):
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            return True
    return False


def _scan_study_dirs(base: Path) -> set[str]:
    found: set[str] = set()
    if not base.is_dir():
        return found
    for child in base.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name in CONTAINER_DIR_NAMES:
            continue
        if _has_images(child):
            found.add(child.name)
    return found


def _ids_from_test_csv(root: Path) -> set[str]:
    path = root / "test.csv"
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or ID_COLUMN not in reader.fieldnames:
            return set()
        return {str(row[ID_COLUMN]) for row in reader if row.get(ID_COLUMN)}


def discovered_study_ids(root: str | Path) -> list[str]:
    """Study IDs from test_series / test_images / test folders, plus test.csv."""
    root = Path(root)
    found: set[str] = set()
    for container in ("test_series", "test_images", "test"):
        found.update(_scan_study_dirs(root / container))
    found.update(_scan_study_dirs(root))
    found.update(_ids_from_test_csv(root))
    return sorted(found)


def count_discovered_studies(root: str | Path) -> int:
    return len(discovered_study_ids(root))
