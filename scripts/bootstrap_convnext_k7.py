"""1000-draw CI for the 7-finding ConvNeXt OOF blend at weight 0.15.

Does not Submit. Drops a finding whose blended-delta CI lower bound is <= 0
and rescores that smaller set with the shipped rank blend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna_knee.fusion import per_finding_rank_blend
from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS

PRED = ROOT / "artifacts" / "convnext_oof870" / "convnext_oof870.csv"
OUT = ROOT / "artifacts" / "convnext_oof870" / "bootstrap_k7.json"
SEVEN = (
    "ACL",
    "Effusion",
    "Synovitis",
    "MCL",
    "Medial OA",
    "Medial Meniscus",
    "Baker's",
)


def _frame(ids: list[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(FINDINGS))
    frame.insert(0, ID_COLUMN, ids)
    return frame[list(SUBMISSION_COLUMNS)]


def _boot(y, mill_v, new_v, findings: tuple[str, ...], draws: list[np.ndarray]) -> dict:
    n = len(y)
    weights = {name: (0.15 if name in findings else 0.0) for name in FINDINGS}
    ids = [str(i) for i in range(n)]
    point = per_finding_rank_blend(_frame(ids, mill_v), _frame(ids, new_v), weights)
    point_v = point[list(FINDINGS)].to_numpy(np.float64)
    point_macro = float(macro_roc_auc(y, point_v) - macro_roc_auc(y, mill_v))
    point_each = per_finding_roc_auc(y, point_v) - per_finding_roc_auc(y, mill_v)
    macros = np.empty(len(draws), dtype=np.float64)
    each = np.empty((len(draws), len(FINDINGS)), dtype=np.float64)
    for i, take in enumerate(draws):
        stub = [str(j) for j in range(n)]
        mixed = per_finding_rank_blend(_frame(stub, mill_v[take]), _frame(stub, new_v[take]), weights)
        mv = mixed[list(FINDINGS)].to_numpy(np.float64)
        macros[i] = macro_roc_auc(y[take], mv) - macro_roc_auc(y[take], mill_v[take])
        each[i] = per_finding_roc_auc(y[take], mv) - per_finding_roc_auc(y[take], mill_v[take])
    lo, hi = np.percentile(macros, [2.5, 97.5])
    per = {}
    for name in findings:
        j = FINDINGS.index(name)
        plo, phi = np.percentile(each[:, j], [2.5, 97.5])
        per[name] = {
            "point": float(point_each[j]),
            "lo": float(plo),
            "hi": float(phi),
            "lo_gt_0": bool(plo > 0.0),
        }
    return {
        "findings": list(findings),
        "weight": 0.15,
        "point_macro_delta": point_macro,
        "macro_lo": float(lo),
        "macro_hi": float(hi),
        "clears_0.003": bool(lo >= 0.003),
        "per_finding": per,
    }


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
    rng = np.random.default_rng(2026)
    n = len(ids)
    draws = [rng.integers(0, n, size=n) for _ in range(1000)]
    full = _boot(y, mill_v, new_v, SEVEN, draws)
    keep = tuple(name for name, row in full["per_finding"].items() if row["lo_gt_0"])
    trimmed = _boot(y, mill_v, new_v, keep, draws) if keep and keep != SEVEN else None
    payload = {
        "n": n,
        "n_boot": 1000,
        "seed": 2026,
        "do_not_submit": True,
        "keep_anchor": "55779066",
        "seven": full,
        "trimmed_to_positive_ci": trimmed,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"seven point={full['point_macro_delta']:+.5f} "
        f"ci={full['macro_lo']:+.5f}..{full['macro_hi']:+.5f} clears={full['clears_0.003']}"
    )
    for name, row in full["per_finding"].items():
        print(f"  {name} point={row['point']:+.5f} ci={row['lo']:+.5f}..{row['hi']:+.5f} lo>0={row['lo_gt_0']}")
    if trimmed is not None:
        print(
            f"trimmed k={len(trimmed['findings'])} point={trimmed['point_macro_delta']:+.5f} "
            f"ci={trimmed['macro_lo']:+.5f}..{trimmed['macro_hi']:+.5f} clears={trimmed['clears_0.003']}"
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
