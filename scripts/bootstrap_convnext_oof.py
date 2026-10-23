"""Bootstrap the ACL+Effusion @0.15 rank-blend on the pinned 517 OOF rows.

Does not Submit. Prints the blended-macro delta and its percentile interval.
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
OUT = ROOT / "artifacts" / "convnext_oof870" / "bootstrap_ci.json"
WEIGHTS = {name: (0.15 if name in ("ACL", "Effusion") else 0.0) for name in FINDINGS}


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
    blended = per_finding_rank_blend(_frame(ids, mill_v), _frame(ids, new_v), WEIGHTS)
    blend_v = blended[list(FINDINGS)].to_numpy(np.float64)
    point_macro = float(macro_roc_auc(y, blend_v) - macro_roc_auc(y, mill_v))
    point_each = per_finding_roc_auc(y, blend_v) - per_finding_roc_auc(y, mill_v)
    rng = np.random.default_rng(2026)
    n = len(ids)
    macros = np.empty(1000, dtype=np.float64)
    acl = np.empty(1000, dtype=np.float64)
    effusion = np.empty(1000, dtype=np.float64)
    acl_i = FINDINGS.index("ACL")
    eff_i = FINDINGS.index("Effusion")
    for b in range(1000):
        take = rng.integers(0, n, size=n)
        yb = y[take]
        mb = mill_v[take]
        nb = new_v[take]
        # fresh ids so rank-blend does not see duplicate index labels as a problem
        stub = [str(i) for i in range(n)]
        mixed = per_finding_rank_blend(_frame(stub, mb), _frame(stub, nb), WEIGHTS)
        mv = mixed[list(FINDINGS)].to_numpy(np.float64)
        macros[b] = macro_roc_auc(yb, mv) - macro_roc_auc(yb, mb)
        each = per_finding_roc_auc(yb, mv) - per_finding_roc_auc(yb, mb)
        acl[b] = each[acl_i]
        effusion[b] = each[eff_i]
    def band(arr: np.ndarray) -> dict[str, float]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"lo": float(lo), "hi": float(hi), "mean": float(arr.mean())}

    payload = {
        "n": n,
        "n_boot": 1000,
        "seed": 2026,
        "findings": ["ACL", "Effusion"],
        "weight": 0.15,
        "point_blended_macro_delta": point_macro,
        "point_acl_delta": float(point_each[acl_i]),
        "point_effusion_delta": float(point_each[eff_i]),
        "macro_delta_ci95": band(macros),
        "acl_delta_ci95": band(acl),
        "effusion_delta_ci95": band(effusion),
        "bar_macro_lo_ge_0.003": bool(band(macros)["lo"] >= 0.003),
        "do_not_submit": True,
        "keep": "55779066",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
