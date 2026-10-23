"""Screen multi-finding ConvNeXt OOF rank-blends. Does not Submit.

Uses the shipped rank blend on the pinned 517-row table. A config is legal
only when it is not an all-12 mix and not a ruled-out finding set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna_knee.fusion import (
    is_ruled_out_dino3s_replace_family,
    is_ruled_out_jointline_mill_radimagenet_family,
    is_ruled_out_public_native384_family,
    is_ruled_out_r336ov_family,
    is_uniform_all_finding_overlay,
    per_finding_rank_blend,
)
from rsna_knee.metrics import macro_roc_auc
from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS

PRED = ROOT / "artifacts" / "convnext_oof870" / "convnext_oof870.csv"
OUT = ROOT / "artifacts" / "convnext_oof870" / "multifind_screen.json"

# Standalone OOF deltas, largest first. PF OA is ~0 and is not included.
CANDIDATES = [
    ("ACL", "Effusion"),
    ("ACL", "Effusion", "Synovitis", "MCL"),
    ("ACL", "Effusion", "Synovitis", "MCL", "Medial OA"),
    (
        "ACL",
        "Effusion",
        "Synovitis",
        "MCL",
        "Medial OA",
        "Medial Meniscus",
        "Baker's",
    ),
]
WEIGHTS = (0.15, 0.20)


def _frame(ids: list[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(FINDINGS))
    frame.insert(0, ID_COLUMN, ids)
    return frame[list(SUBMISSION_COLUMNS)]


def _legal(findings: tuple[str, ...], weight: float) -> bool:
    if len(findings) >= len(FINDINGS):
        return False
    weights = {name: (weight if name in findings else 0.0) for name in FINDINGS}
    if is_uniform_all_finding_overlay(weights, min_weight=0.5):
        return False
    if is_ruled_out_r336ov_family(findings, weight=weight):
        return False
    if is_ruled_out_public_native384_family(findings):
        return False
    if is_ruled_out_dino3s_replace_family(findings, weight=weight):
        return False
    if is_ruled_out_jointline_mill_radimagenet_family(findings):
        return False
    return True


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    mill = pd.read_csv(ROOT / "data" / "mill_v10_holdout870.csv", dtype={ID_COLUMN: str})
    pred = pd.read_csv(PRED, dtype={ID_COLUMN: str})
    holdout, y_hold = holdout_ids_and_y(train, v4)
    y_map = {uid: y_hold[i] for i, uid in enumerate(holdout)}
    ids = pred[ID_COLUMN].astype(str).tolist()
    y = np.stack([y_map[uid] for uid in ids])
    mill_v = mill.set_index(ID_COLUMN).loc[ids, list(FINDINGS)].to_numpy(np.float64)
    new_v = pred.set_index(ID_COLUMN).loc[ids, list(FINDINGS)].to_numpy(np.float64)
    n = len(ids)
    rng = np.random.default_rng(2026)
    draws = [rng.integers(0, n, size=n) for _ in range(400)]
    rows = []
    for findings in CANDIDATES:
        for weight in WEIGHTS:
            if not _legal(findings, weight):
                rows.append({"findings": list(findings), "weight": weight, "legal": False})
                continue
            weights = {name: (weight if name in findings else 0.0) for name in FINDINGS}
            point = per_finding_rank_blend(_frame(ids, mill_v), _frame(ids, new_v), weights)
            point_delta = float(
                macro_roc_auc(y, point[list(FINDINGS)].to_numpy(np.float64)) - macro_roc_auc(y, mill_v)
            )
            deltas = np.empty(len(draws), dtype=np.float64)
            for i, take in enumerate(draws):
                stub = [str(j) for j in range(n)]
                mixed = per_finding_rank_blend(
                    _frame(stub, mill_v[take]), _frame(stub, new_v[take]), weights
                )
                deltas[i] = macro_roc_auc(y[take], mixed[list(FINDINGS)].to_numpy(np.float64)) - macro_roc_auc(
                    y[take], mill_v[take]
                )
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            rows.append(
                {
                    "findings": list(findings),
                    "n_findings": len(findings),
                    "weight": weight,
                    "legal": True,
                    "point_macro_delta": point_delta,
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                    "clears_0.003": bool(lo >= 0.003),
                }
            )
            print(
                f"k={len(findings)} w={weight:.2f} point={point_delta:+.5f} "
                f"ci={lo:+.5f}..{hi:+.5f} clears={bool(lo >= 0.003)}",
                flush=True,
            )
    payload = {"n": n, "n_boot": 400, "seed": 2026, "do_not_submit": True, "rows": rows}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
