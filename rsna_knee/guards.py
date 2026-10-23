"""Kernel / submission sanity checks used by tests and teacher notes.

These do not score the leaderboard. They catch the traps that already burned
a P100 commit (dummy ranks in {0, 0.5, 1}) and a report-leak kernel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rsna_knee.schema import SUBMISSION_COLUMNS

DUMMY_RANK_VALUES = frozenset({0.0, 0.5, 1.0})


def is_dummy_rank_trap(values: np.ndarray, atol: float = 1e-8) -> bool:
    """True when every entry is 0, 0.5, or 1 — the P100 CUDA-fallback pattern."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return True
    rounded = np.round(arr / 0.5) * 0.5
    if not np.allclose(arr, rounded, atol=atol):
        return False
    uniq = {float(x) for x in np.unique(rounded)}
    return uniq.issubset(DUMMY_RANK_VALUES)


def kernel_source_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def kernel_has_t4_gpu_probe(source: str) -> bool:
    return "nvidia-smi" in source and "CUDA_VISIBLE_DEVICES" in source


def kernel_discovers_competitions_mount(source: str) -> bool:
    return "/kaggle/input/competitions/rsna-knee-abnormality-detection" in source


def kernel_keeps_bakers_apostrophe(source: str) -> bool:
    return "Baker's" in source


def kernel_reads_test_report_column(source: str) -> bool:
    """True if the source indexes a Report column (train-only leakage at test)."""
    needles = ("['Report']", '["Report"]', "['report']", '["report"]')
    return any(n in source for n in needles)


def header_is_official(columns) -> bool:
    return list(columns) == list(SUBMISSION_COLUMNS)
