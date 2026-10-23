"""Unequal per-finding weights on the pinned 517-row ConvNeXt table.

Each column is blended on its own, so the search is independent per finding.
A finding is dropped when its blended-delta 95% CI lower bound is <= 0.
Every kept weight is < 0.50. Does not push and does not Submit.
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
OUT = ROOT / "artifacts" / "convnext_oof870" / "k7_unequal_weights.json"
GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.49)


def _frame(ids: list[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(FINDINGS))
    frame.insert(0, ID_COLUMN, ids)
    return frame[list(SUBMISSION_COLUMNS)]


def _blend_delta(y, mill_v, new_v, weights: dict[str, float]) -> tuple[float, np.ndarray]:
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
    rng = np.random.default_rng(2026)
    draws = [rng.integers(0, len(ids), size=len(ids)) for _ in range(400)]

    chosen: dict[str, float] = {name: 0.0 for name in FINDINGS}
    detail = {}
    for name in FINDINGS:
        j = FINDINGS.index(name)
        best_w, best_point = 0.0, -1.0
        for weight in GRID:
            if weight >= 0.50:
                continue
            trial = {col: 0.0 for col in FINDINGS}
            trial[name] = weight
            _macro, each = _blend_delta(y, mill_v, new_v, trial)
            if float(each[j]) > best_point:
                best_w, best_point = weight, float(each[j])
        boots = np.empty(len(draws), dtype=np.float64)
        trial = {col: 0.0 for col in FINDINGS}
        trial[name] = best_w
        for i, take in enumerate(draws):
            _macro, each = _blend_delta(y[take], mill_v[take], new_v[take], trial)
            boots[i] = float(each[j])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        keep = bool(best_w > 0.0 and float(lo) > 0.0)
        weight_out = best_w if keep else 0.0
        chosen[name] = weight_out
        detail[name] = {
            "argmax_weight": best_w,
            "point": best_point if keep else 0.0,
            "lo": float(lo),
            "hi": float(hi),
            "kept": keep,
            "weight": weight_out,
        }
        print(
            f"{name} argmax={best_w:.2f} point={best_point:+.5f} "
            f"ci={float(lo):+.5f}..{float(hi):+.5f} kept={keep}",
            flush=True,
        )

    macro, each = _blend_delta(y, mill_v, new_v, chosen)
    joint = np.empty(len(draws), dtype=np.float64)
    for i, take in enumerate(draws):
        joint[i], _each = _blend_delta(y[take], mill_v[take], new_v[take], chosen)
    jlo, jhi = np.percentile(joint, [2.5, 97.5])
    payload = {
        "n": len(ids),
        "n_boot": 400,
        "seed": 2026,
        "do_not_submit": True,
        "weights": chosen,
        "findings": detail,
        "point_macro_delta": macro,
        "macro_lo": float(jlo),
        "macro_hi": float(jhi),
        "kept": [name for name, row in detail.items() if row["kept"]],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"macro={macro:+.5f} ci={float(jlo):+.5f}..{float(jhi):+.5f} kept={payload['kept']}"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
