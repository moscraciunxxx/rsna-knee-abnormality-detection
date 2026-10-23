"""Score the left-out-fold ConvNeXt table against mill on the same IDs.

Does not Submit. Writes artifacts/convnext_oof870/overlay_spec.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna_knee.convnext_oof import LABELS, OOF_FOLDS, author_v0_folds, oof_ids_by_fold
from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.mill_holdout import decide_private_overlay, holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN

OUT = ROOT / "artifacts" / "convnext_oof870"
PRED = OUT / "convnext_oof870.csv"


def main() -> None:
    if not PRED.is_file():
        raise SystemExit(f"missing {PRED}")
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    mill = pd.read_csv(ROOT / "data" / "mill_v10_holdout870.csv", dtype={ID_COLUMN: str})
    pred = pd.read_csv(PRED, dtype={ID_COLUMN: str})
    holdout, y_hold = holdout_ids_and_y(train, v4)
    folds = author_v0_folds(train)
    expected = oof_ids_by_fold(folds, holdout)
    expected_ids = tuple(uid for fold_id in OOF_FOLDS for uid in expected[fold_id])
    got = tuple(pred[ID_COLUMN].astype(str).tolist())
    if got != expected_ids:
        raise SystemExit(f"OOF id order/count mismatch: got {len(got)} expected {len(expected_ids)}")
    if set(pred["fold"].astype(int)) - set(OOF_FOLDS):
        raise SystemExit("pred fold outside 0,1,2")
    y_map = {uid: y_hold[i] for i, uid in enumerate(holdout)}
    y = np.stack([y_map[uid] for uid in expected_ids])
    mill_i = mill.set_index(ID_COLUMN).loc[list(expected_ids), list(FINDINGS)].to_numpy(dtype=np.float64)
    new_i = pred.set_index(ID_COLUMN).loc[list(expected_ids), list(FINDINGS)].to_numpy(dtype=np.float64)
    mill_per = per_finding_roc_auc(y, mill_i)
    new_per = per_finding_roc_auc(y, new_i)
    decision, weights = decide_private_overlay(mill_per, new_per)
    print(f"n={len(expected_ids)} leaked_vs_gold58=0")
    print(f"mill_macro={macro_roc_auc(y, mill_i):.4f} convnext_oof_macro={macro_roc_auc(y, new_i):.4f}")
    for name, m, n in zip(FINDINGS, mill_per, new_per):
        print(f"{name} mill={m:.4f} oof={n:.4f} delta={n - m:+.4f}")
    print(f"decision={decision}")
    findings = [] if not weights else [k for k, w in weights.items() if float(w) > 0]
    weight = 0.0 if not findings else float(weights[findings[0]])
    spec = {
        "decision": decision,
        "keep": "55779066",
        "do_not_submit_holdout_ids": True,
        "source": "convnext-oof870",
        "n": len(expected_ids),
        "findings": findings,
        "weight": weight,
        "mill_macro": float(macro_roc_auc(y, mill_i)),
        "convnext_oof_macro": float(macro_roc_auc(y, new_i)),
        "note": "OOF only. Do not Submit. A decision of overlay still needs the raised CI bar before any TEST kernel.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlay_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    (OUT / "DO_NOT_SUBMIT_HOLDOUT.txt").write_text(
        "convnext_oof870.csv is train holdout IDs, not the hidden test.\n"
        "Keep 55779066. Do not Submit.\n",
        encoding="utf-8",
    )
    print("wrote", OUT / "overlay_spec.json")


if __name__ == "__main__":
    main()
