"""Seed-2026 holdout shim for the public CoAtNet/d4 mill-870 gate.

The Kaggle script cannot import this package. ``SHIM_SOURCE`` is copied into
``kaggle_kernel_coatnet870_gate/main.py`` by the build script. Local tests exec
that block and check it against ``holdout_ids_and_y``. No weights. No Submit.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS

HOLD_SEED = 2026
HOLD_FRAC = 0.2
EXPECTED_HOLDOUT_N = 870

SERIES_COLUMNS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)

# Pasted into the kernel. Keep this block free of rsna_knee imports.
SHIM_SOURCE = r'''
HOLD_SEED = 2026
HOLD_FRAC = 0.2
EXPECTED_HOLDOUT_N = 870
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SUBMISSION_HEADER = ["StudyInstanceUID"] + TARGETS
SERIES_COLUMNS = ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]

def _gate_gold_ids(train):
    mask = train[list(TARGETS)].notna().all(axis=1)
    gold = set(train.loc[mask, "StudyInstanceUID"].astype(str))
    if len(gold) != 58:
        raise RuntimeError("gold count %s != 58; refusing coatnet holdout" % (len(gold),))
    return gold

def _gate_holdout_ids(train_csv, v4_csv):
    """Same seed-2026 LLM-soft split as mill-holdout. Gold 58 never included."""
    train = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    v4 = pd.read_csv(v4_csv, dtype={"StudyInstanceUID": str})
    gold = _gate_gold_ids(train)
    labels = v4.copy()
    labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)
    labels = labels.drop_duplicates("StudyInstanceUID")
    train_frame = labels[~labels["StudyInstanceUID"].isin(gold)].copy()
    train_ids = tuple(train_frame["StudyInstanceUID"].tolist())
    if set(train_ids) & gold:
        raise RuntimeError("gold 58 leaked into coatnet-gate train ids")
    rng = np.random.default_rng(int(HOLD_SEED))
    n = len(train_ids)
    perm = rng.permutation(n)
    n_ho = max(1, min(n - 1, int(round(n * float(HOLD_FRAC)))))
    holdout = np.sort(perm[:n_ho])
    ids = tuple(train_ids[int(i)] for i in holdout.tolist())
    leaked = set(ids) & gold
    if leaked:
        raise RuntimeError("LLM-soft holdout overlapped gold 58")
    if len(ids) != EXPECTED_HOLDOUT_N:
        raise RuntimeError("coatnet gate expected n=%s, got %s" % (EXPECTED_HOLDOUT_N, len(ids)))
    print(
        "[coatnet870] llm-soft holdout n=%s gold=58 excluded leaked=%s seed=%s"
        % (len(ids), len(leaked), HOLD_SEED),
        flush=True,
    )
    return ids

def _gate_stage_comp(real_comp, ids, dest):
    """Fake competition root: holdout test.csv + symlinks into train_series only."""
    real_comp = Path(real_comp)
    dest = Path(dest)
    id_list = [str(x) for x in ids]
    if len(id_list) != len(set(id_list)):
        raise RuntimeError("duplicate holdout StudyInstanceUID")
    series_csv = real_comp / "train_series.csv"
    series_root = real_comp / "train_series"
    if not series_csv.is_file() or not series_root.is_dir():
        raise FileNotFoundError("real competition train_series missing at %s" % (real_comp,))
    ser = pd.read_csv(series_csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    missing_cols = [c for c in SERIES_COLUMNS if c not in ser.columns]
    if missing_cols:
        raise RuntimeError("train_series.csv missing %s" % (missing_cols,))
    id_set = set(id_list)
    kept = ser[ser["StudyInstanceUID"].isin(id_set)].copy()
    have = set(kept["StudyInstanceUID"].astype(str))
    missing_rows = [i for i in id_list if i not in have]
    if missing_rows:
        raise RuntimeError("train_series.csv missing %s/%s holdout studies" % (len(missing_rows), len(id_list)))
    missing_dirs = [i for i in id_list if not (series_root / i).is_dir()]
    if missing_dirs:
        raise RuntimeError("train_series dirs missing %s/%s holdout studies" % (len(missing_dirs), len(id_list)))
    if dest.exists():
        raise RuntimeError("refusing to delete existing %s" % (dest,))
    (dest / "test_series").mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": id_list}).to_csv(dest / "test.csv", index=False)
    kept[list(SERIES_COLUMNS)].to_csv(dest / "test_series.csv", index=False)
    sample = pd.DataFrame({"StudyInstanceUID": id_list})
    for name in TARGETS:
        sample[name] = 0.5
    sample = sample[SUBMISSION_HEADER]
    sample.to_csv(dest / "sample_submission.csv", index=False)
    # plan_cache only reads the row count. A stub avoids sizing the cache for all of train.
    pd.DataFrame({"StudyInstanceUID": ["stub-train-0", "stub-train-1"]}).to_csv(dest / "train.csv", index=False)
    for uid in id_list:
        os.symlink(series_root / uid, dest / "test_series" / uid, target_is_directory=True)
    print(
        "[coatnet870] staged holdout comp %s studies=%s series_rows=%s"
        % (dest, len(id_list), len(kept)),
        flush=True,
    )
    return dest
'''


def holdout_study_ids(train: pd.DataFrame, v4: pd.DataFrame) -> tuple[str, ...]:
    """Library split. Must match the pasted ``_gate_holdout_ids`` block."""
    ids, _y = holdout_ids_and_y(train, v4, frac=HOLD_FRAC, seed=HOLD_SEED)
    if len(ids) != EXPECTED_HOLDOUT_N:
        raise RuntimeError(f"coatnet gate expected n={EXPECTED_HOLDOUT_N}, got {len(ids)}")
    return ids


def shim_namespace() -> dict:
    """Exec the pasted kernel block with local pandas/numpy."""
    ns: dict = {
        "pd": pd,
        "np": np,
        "Path": Path,
        "os": os,
        "__builtins__": __builtins__,
    }
    exec(SHIM_SOURCE, ns)
    return ns


def assert_shim_matches_schema() -> None:
    ns = shim_namespace()
    if tuple(ns["TARGETS"]) != tuple(FINDINGS):
        raise RuntimeError("coatnet shim TARGETS drifted from schema.FINDINGS")
    if tuple(ns["SUBMISSION_HEADER"]) != tuple(SUBMISSION_COLUMNS):
        raise RuntimeError("coatnet shim header drifted from schema.SUBMISSION_COLUMNS")
