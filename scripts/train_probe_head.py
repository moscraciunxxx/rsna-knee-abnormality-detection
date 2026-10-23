#!/usr/bin/env python
"""CPU dry-run for the frozen-encoder probe-head protocol.

Loads train / series / LLM CSVs, prints the split, mill attach recipe, time
estimate, and go/no-go. Does not read DICOMs and does not pretend they exist.

Usage:
    python scripts/train_probe_head.py
    python scripts/train_probe_head.py --data-root data --llm-csv .cache/llm-labels/report_labels_v2.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.schema import FINDINGS, ID_COLUMN, N_FINDINGS
from rsna_knee.v4_backbone import (
    BACKBONE_CROP_MM,
    BACKBONE_IMG,
    BACKBONE_NAME,
    FORBIDDEN_SCORED_CHECKPOINTS,
    UNFREEZE_LAST,
)
from rsna_knee.v4_split import GOLD_AUC_ABORT_THRESHOLD, make_steven_v4_split

MISSING_LLM_GOLD_UID = (
    "1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220"
)

# Mill V10 DINO (kaggle_kernel_blend_script/main.py). Do not steal mill config.json.
MILL_DINO_VARIANT = "small"
MILL_HIDDEN = 384
MILL_POOL = "cls_mean"
MILL_POOL_PARTS = 2
MILL_SLOT_IN = MILL_HIDDEN * MILL_POOL_PARTS  # 768
MILL_HEAD_HIDDEN = 256
MILL_IMG = 336
MILL_GROUP = 3
MILL_N_SLOT = 6
MILL_CROP_MM = 130.0

# Preferred probe encoder: public DINOv2-B (nursrijan r336 size), frozen.
PROBE_VARIANT = "base"
PROBE_HIDDEN = 768
PROBE_SLOT_IN = PROBE_HIDDEN * MILL_POOL_PARTS  # 1536
PROBE_IMG = 336
NURSRIJAN_R336_PT_BYTES = 348_014_603
NURSRIJAN_DATASET = "nursrijan/rsna-knee-dinov2-base-llm-models"

LINEAR_PARAMS = PROBE_SLOT_IN * N_FINDINGS + N_FINDINGS  # LN scale not counted
TIME_BUDGET_H = 4.0
UNFREEZE_LAST = 0


def _find_llm_csv(explicit: Path | None) -> Path | None:
    cands = []
    if explicit is not None:
        cands.append(explicit)
    cands.extend(
        [
            ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv",
            ROOT / ".cache" / "llm-labels" / "report_labels_v2.csv",
            ROOT / "data" / "derived" / "report_labels_v2.csv",
            ROOT / "data" / "report_labels_v2.csv",
        ]
    )
    for p in cands:
        if p.is_file():
            return p
    return None


def _train_dicom_present(data_root: Path) -> bool:
    series = data_root / "train_series"
    if not series.is_dir():
        return False
    for p in series.rglob("*.dcm"):
        return True
    return False


def audit(data_root: Path, llm_csv: Path | None) -> dict:
    train = pd.read_csv(data_root / "train.csv")
    series = pd.read_csv(data_root / "train_series.csv")
    test = pd.read_csv(data_root / "test.csv")

    missing_findings = [c for c in FINDINGS if c not in train.columns]
    if missing_findings:
        raise SystemExit(f"train.csv missing findings: {missing_findings}")

    gold_mask = train[list(FINDINGS)].notna().all(axis=1)
    any_mask = train[list(FINDINGS)].notna().any(axis=1)
    gold = train.loc[gold_mask].copy()
    gold_ids = set(gold[ID_COLUMN])

    report_n = int(train["Report"].fillna("").str.strip().ne("").sum()) if "Report" in train.columns else 0

    llm = None
    llm_n = 0
    llm_overlap = 0
    llm_non_gold = 0
    gold_in_llm = 0
    missing_llm = []
    if llm_csv is not None:
        llm = pd.read_csv(llm_csv)
        llm_n = int(llm[ID_COLUMN].nunique()) if ID_COLUMN in llm.columns else 0
        tr_ids = set(train[ID_COLUMN])
        llm_ids = set(llm[ID_COLUMN]) if ID_COLUMN in llm.columns else set()
        llm_overlap = len(tr_ids & llm_ids)
        missing_llm = sorted(tr_ids - llm_ids)
        gold_in_llm = len(gold_ids & llm_ids)
        llm_non_gold = len((llm_ids & tr_ids) - gold_ids)

    gold_counts = {}
    for name in FINDINGS:
        col = gold[name].astype(float)
        gold_counts[name] = {
            "n": int(col.notna().sum()),
            "pos": int((col == 1).sum()),
            "neg": int((col == 0).sum()),
            "prev": float(col.mean()) if len(col) else float("nan"),
        }

    gold_series = series[series[ID_COLUMN].isin(gold_ids)]
    planes = (
        gold_series.groupby(ID_COLUMN)["Anatomical_Plane"].nunique()
        if len(gold_series)
        else pd.Series(dtype=int)
    )

    return {
        "n_train": int(train[ID_COLUMN].nunique()),
        "n_train_rows": int(len(train)),
        "n_series": int(series["SeriesInstanceUID"].nunique())
        if "SeriesInstanceUID" in series.columns
        else int(len(series)),
        "n_studies_in_series": int(series[ID_COLUMN].nunique()),
        "series_per_study_mean": float(series.groupby(ID_COLUMN).size().mean()),
        "planes": series["Anatomical_Plane"].value_counts().to_dict()
        if "Anatomical_Plane" in series.columns
        else {},
        "n_report": report_n,
        "n_gold": int(gold_mask.sum()),
        "n_partial_label": int((any_mask & ~gold_mask).sum()),
        "gold_counts": gold_counts,
        "gold_all_three_planes": int((planes == 3).sum()) if len(planes) else 0,
        "n_test": int(len(test)),
        "test_has_report": "Report" in test.columns,
        "llm_csv": str(llm_csv) if llm_csv else None,
        "llm_n": llm_n,
        "llm_overlap": llm_overlap,
        "llm_non_gold": llm_non_gold,
        "gold_in_llm": gold_in_llm,
        "missing_llm": missing_llm,
        "train_dicoms": _train_dicom_present(data_root),
    }


def mill_attach_recipe() -> dict:
    return {
        "mill_backbone": f"transformers.AutoModel DINOv2-{MILL_DINO_VARIANT}",
        "mill_hidden_size": MILL_HIDDEN,
        "mill_pool": MILL_POOL,
        "mill_slot_in": MILL_SLOT_IN,
        "mill_head": f"SlotHead hidden={MILL_HEAD_HIDDEN} n_out={N_FINDINGS}",
        "do_not_steal": [
            "mill member config.json / adopt_config_globals",
            "mill member state_dict (SlotHead + unfrozen blocks)",
            "directories that have config.json but no 'dinov2' in the path",
            f"{NURSRIJAN_DATASET} r336_best.pt as the scored head "
            f"({NURSRIJAN_R336_PT_BYTES} B = already-trained public mill-like B)",
        ],
        "probe_backbone": f"public DINOv2-{PROBE_VARIANT} HF config.json (hidden={PROBE_HIDDEN})",
        "probe_slot_in": PROBE_SLOT_IN,
        "probe_head": f"Linear({PROBE_SLOT_IN}, {N_FINDINGS}) or SlotHead hidden<=256",
        "unfreeze_last": UNFREEZE_LAST,
        "img": PROBE_IMG,
        "group": MILL_GROUP,
        "n_slot": MILL_N_SLOT,
        "images_per_study": MILL_N_SLOT * MILL_GROUP,
        "linear_params": LINEAR_PARAMS,
        "nursrijan_r336": NURSRIJAN_DATASET,
    }


def time_and_go(facts: dict) -> dict:
    images = int(facts["n_train"]) * MILL_N_SLOT * MILL_GROUP
    # Co-plane ViT-S@336 including DICOM decode ~14 img/s on T4. ViT-B is ~4x
    # FLOP; T4x2 recovers ~2x. 8 img/s is the conservative B@336 T4x2 rate.
    imgs_s_t4x2_b = 8.0
    encode_h = images / imgs_s_t4x2_b / 3600.0
    mill58_h = 0.5
    head_h = 0.15
    total_h = encode_h + mill58_h + head_h
    freeze_ok = UNFREEZE_LAST == 0
    dicom_ok = bool(facts["train_dicoms"])
    # This process is the dry-run: never GO for GPU spend from here.
    session = "NO-GO"
    later = (
        "GO"
        if freeze_ok and total_h <= TIME_BUDGET_H
        else "NO-GO"
    )
    return {
        "images_if_full_train": images,
        "projected_encode_h_t4x2_B336": round(encode_h, 2),
        "projected_mill58_h": mill58_h,
        "projected_head_h": head_h,
        "projected_total_h": round(total_h, 2),
        "time_budget_h": TIME_BUDGET_H,
        "abort_if_unfreeze": True,
        "abort_if_projected_gt_h": TIME_BUDGET_H,
        "session_go": session,
        "later_gpu_session_go": later,
        "local_dicoms": dicom_ok,
        "reason": (
            "CPU dry-run; no kernel push; no 24h finetune; "
            + (
                "train DICOMs absent locally"
                if not dicom_ok
                else "train DICOMs present but this script will not train"
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--llm-csv", type=Path, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object instead of the text protocol dump",
    )
    args = parser.parse_args(argv)

    data_root = args.data_root
    if not (data_root / "train.csv").is_file():
        raise SystemExit(f"missing {data_root / 'train.csv'}")

    llm_csv = _find_llm_csv(args.llm_csv)
    facts = audit(data_root, llm_csv)
    recipe = mill_attach_recipe()
    clock = time_and_go(facts)
    payload = {"facts": facts, "encoder": recipe, "clock": clock}

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print("=== RSNA probe-head CPU dry-run (no DICOMs, no train) ===")
    print(f"data_root          : {data_root}")
    print(f"n_train            : {facts['n_train']}")
    print(f"n_series           : {facts['n_series']}")
    print(f"n_gold (all 12)    : {facts['n_gold']}")
    print(f"n_partial labels   : {facts['n_partial_label']}")
    print(f"n_report           : {facts['n_report']}")
    print(f"n_test (no Report) : {facts['n_test']}  has_report={facts['test_has_report']}")
    print(f"gold 3-plane       : {facts['gold_all_three_planes']}/{facts['n_gold']}")
    print(f"train DICOMs local : {facts['train_dicoms']}")
    print()
    print("--- gold-58 prevalence ---")
    for name in FINDINGS:
        row = facts["gold_counts"][name]
        print(
            f"  {name:20s} pos={row['pos']:2d} neg={row['neg']:2d} prev={row['prev']:.3f}"
        )
    print()
    print("--- steven v4 train / gold 58 val-only ---")
    v4_path = facts["llm_csv"]
    print(f"  csv              : {v4_path or '(steven v4 not found)'}")
    if v4_path and Path(v4_path).is_file():
        train_df = pd.read_csv(data_root / "train.csv", dtype={ID_COLUMN: str})
        v4_df = pd.read_csv(v4_path, dtype={ID_COLUMN: str})
        split = make_steven_v4_split(train_df, v4_df)
        print(f"  train labels     : steven v4  n={split.n_train}")
        print(f"  gold=58 val-only : n={split.n_val}  never train")
        print(f"  gold in train    : {len(set(split.train_ids) & set(split.val_ids))}")
        print(f"  abort if gold AUC > {GOLD_AUC_ABORT_THRESHOLD:.2f}")
    else:
        print("  train labels     : steven v4 MISSING")
        print("  gold=58 val-only : (split not built)")
    print()
    print("--- frozen backbone (NOT mill 6-slot 336/130 mm) ---")
    print(f"  backbone         : {BACKBONE_NAME} img={BACKBONE_IMG} crop_mm={BACKBONE_CROP_MM}")
    print(f"  unfreeze_last    : {UNFREEZE_LAST}  (must stay 0)")
    print("  forbidden scored :")
    for item in FORBIDDEN_SCORED_CHECKPOINTS:
        print(f"    - {item}")
    print("  never load r336_best.pt as the scored head")
    print()
    print("--- time / go-no-go ---")
    print(f"  train DICOMs local : {facts['train_dicoms']}")
    print(f"  THIS SESSION dry-run: NO-GO GPU (no local DICOMs to fit)")
    print(f"  reason           : {clock['reason']}")
    print()
    print("Train labels = steven v4. Gold 58 = val only never train. Overlay << 0.50, fail-closed.")
    rng = np.random.default_rng(2026)
    feat = 512
    w = rng.normal(0, 0.01, size=(N_FINDINGS, feat))
    b = np.zeros(N_FINDINGS)
    x = rng.normal(0, 1, size=(8, feat))
    logits = x @ w.T + b
    print(f"dummy linear logits shape {logits.shape} (not a trained model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
