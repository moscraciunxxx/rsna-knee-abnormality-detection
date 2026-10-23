"""Bootstrap uniform high weights on the seven OOF findings. Does not Submit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna_knee.fusion import CONVNEXT_OOF_K7_FINDINGS, per_finding_rank_blend
from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS

PRED = ROOT / "artifacts" / "convnext_oof870" / "convnext_oof870.csv"
OUT = ROOT / "artifacts" / "convnext_oof870" / "k7_high_weight.json"
LEVELS = (0.15, 0.35, 0.45, 0.49)


def _frame(ids: list[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(FINDINGS))
    frame.insert(0, ID_COLUMN, ids)
    return frame[list(SUBMISSION_COLUMNS)]


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
    stub = [str(i) for i in range(len(ids))]
    rng = np.random.default_rng(2026)
    draws = [rng.integers(0, len(ids), size=len(ids)) for _ in range(400)]
    rows = []
    for weight in LEVELS:
        weights = {name: (weight if name in CONVNEXT_OOF_K7_FINDINGS else 0.0) for name in FINDINGS}
        point_mix = per_finding_rank_blend(_frame(stub, mill_v), _frame(stub, new_v), weights)
        point_v = point_mix[list(FINDINGS)].to_numpy(np.float64)
        point_macro = float(macro_roc_auc(y, point_v) - macro_roc_auc(y, mill_v))
        point_each = per_finding_roc_auc(y, point_v) - per_finding_roc_auc(y, mill_v)
        macros = np.empty(len(draws), dtype=np.float64)
        each = np.empty((len(draws), len(FINDINGS)), dtype=np.float64)
        for i, take in enumerate(draws):
            mixed = per_finding_rank_blend(
                _frame(stub, mill_v[take]), _frame(stub, new_v[take]), weights
            )
            mv = mixed[list(FINDINGS)].to_numpy(np.float64)
            macros[i] = macro_roc_auc(y[take], mv) - macro_roc_auc(y[take], mill_v[take])
            each[i] = per_finding_roc_auc(y[take], mv) - per_finding_roc_auc(y[take], mill_v[take])
        lo, hi = np.percentile(macros, [2.5, 97.5])
        per = {}
        for name in CONVNEXT_OOF_K7_FINDINGS:
            j = FINDINGS.index(name)
            plo, phi = np.percentile(each[:, j], [2.5, 97.5])
            per[name] = {"point": float(point_each[j]), "lo": float(plo), "hi": float(phi)}
        row = {
            "weight": weight,
            "point_macro_delta": point_macro,
            "macro_lo": float(lo),
            "macro_hi": float(hi),
            "per_finding": per,
            "all_lo_gt_0": all(item["lo"] > 0.0 for item in per.values()),
        }
        rows.append(row)
        print(
            f"w={weight:.2f} point={point_macro:+.5f} ci={lo:+.5f}..{hi:+.5f} "
            f"all_lo_gt_0={row['all_lo_gt_0']}",
            flush=True,
        )
    payload = {"n": len(ids), "n_boot": 400, "seed": 2026, "do_not_submit": True, "rows": rows}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
