"""Joint CI after dropping the two razor-thin findings. Does not Submit.

Lateral Meniscus (CI lo +0.00019) and PF OA (CI lo +0.00061) stay on mill.
The remaining clearly positive findings keep weight 0.49.
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
OUT = ROOT / "artifacts" / "convnext_oof870" / "k7_unequal_strict.json"
KEPT = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Medial OA",
    "Effusion",
    "Synovitis",
    "Baker's",
)


def _frame(ids: list[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(FINDINGS))
    frame.insert(0, ID_COLUMN, ids)
    return frame[list(SUBMISSION_COLUMNS)]


def _delta(y, mill_v, new_v, weights: dict[str, float]) -> tuple[float, np.ndarray]:
    stub = [str(i) for i in range(len(y))]
    mixed = per_finding_rank_blend(_frame(stub, mill_v), _frame(stub, new_v), weights)
    mv = mixed[list(FINDINGS)].to_numpy(np.float64)
    each = per_finding_roc_auc(y, mv) - per_finding_roc_auc(y, mill_v)
    macro = float(macro_roc_auc(y, mv) - macro_roc_auc(y, mill_v))
    return macro, each


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
    weights = {name: (0.49 if name in KEPT else 0.0) for name in FINDINGS}
    rng = np.random.default_rng(2026)
    draws = [rng.integers(0, len(ids), size=len(ids)) for _ in range(400)]
    point_macro, point_each = _delta(y, mill_v, new_v, weights)
    macros = np.empty(len(draws), dtype=np.float64)
    each = np.empty((len(draws), len(FINDINGS)), dtype=np.float64)
    for i, take in enumerate(draws):
        macros[i], each[i] = _delta(y[take], mill_v[take], new_v[take], weights)
    lo, hi = np.percentile(macros, [2.5, 97.5])
    per = {}
    for name in KEPT:
        j = FINDINGS.index(name)
        plo, phi = np.percentile(each[:, j], [2.5, 97.5])
        per[name] = {
            "weight": 0.49,
            "point": float(point_each[j]),
            "lo": float(plo),
            "hi": float(phi),
        }
        print(f"{name} point={float(point_each[j]):+.5f} ci={float(plo):+.5f}..{float(phi):+.5f}", flush=True)
    dropped = [name for name in FINDINGS if name not in KEPT]
    payload = {
        "n": len(ids),
        "n_boot": 400,
        "seed": 2026,
        "do_not_submit": True,
        "weights": weights,
        "kept": list(KEPT),
        "dropped": dropped,
        "point_macro_delta": point_macro,
        "macro_lo": float(lo),
        "macro_hi": float(hi),
        "per_finding": per,
        "all_kept_lo_gt_0": all(row["lo"] > 0.0 for row in per.values()),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"macro={point_macro:+.5f} ci={float(lo):+.5f}..{float(hi):+.5f} "
        f"all_kept_lo_gt_0={payload['all_kept_lo_gt_0']}"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
