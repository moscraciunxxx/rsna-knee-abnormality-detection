"""After vitb870 COMPLETE: join mill-870 vs ViT-B; write overlay_spec.json. Never Submit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.mill_holdout import (
    MillHoldoutMissing,
    decide_private_overlay,
    holdout_ids_and_y,
    join_mill_holdout,
)
from rsna_knee.schema import FINDINGS, ID_COLUMN


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mill", type=Path, default=ROOT / "data" / "mill_v10_holdout870.csv")
    p.add_argument("--vitb", type=Path, required=True)
    p.add_argument("--spec", type=Path, default=ROOT / "kaggle_kernel_mill870_ov" / "overlay_spec.json")
    args = p.parse_args()
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(
        ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv",
        dtype={ID_COLUMN: str},
    )
    ids, y = holdout_ids_and_y(train, v4)
    mill = pd.read_csv(args.mill, dtype={ID_COLUMN: str}) if args.mill.is_file() else None
    mill_join = join_mill_holdout(ids, mill, y)
    vitb = pd.read_csv(args.vitb, dtype={ID_COLUMN: str}) if args.vitb.is_file() else None
    if vitb is None:
        raise MillHoldoutMissing("vitb_holdout870.csv missing")
    new_join = join_mill_holdout(ids, vitb, y)
    print(f"mill n={len(mill_join.ids)} macro={mill_join.macro:.4f}")
    print(f"vitb n={len(new_join.ids)} macro={new_join.macro:.4f}")
    print(f"{'finding':22} {'mill':7} {'vitb':7} {'delta':7}")
    for name, m, n in zip(FINDINGS, mill_join.per_auc, new_join.per_auc):
        dlt = (n - m) if (m == m and n == n) else float("nan")
        print(f"{name:22} {m:7.4f} {n:7.4f} {dlt:7.4f}")
    decision, weights = decide_private_overlay(mill_join.per_auc, new_join.per_auc)
    spec = {
        "decision": decision,
        "keep": "55779066",
        "do_not_submit_holdout_ids": True,
        "findings": [f for f, w in (weights or {}).items() if w > 0],
        "weight": (max((weights or {}).values()) if weights else 0.0),
    }
    args.spec.parent.mkdir(parents=True, exist_ok=True)
    args.spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"decision={decision} spec={args.spec}")
    if decision == "stop":
        print("STOP_PUBLIC_TRAINS: vitb < mill on every finding; keep 55779066; do not Submit")
    elif decision == "hold":
        print("HOLD: no legal overlay; keep 55779066; do not Submit")
    else:
        print(
            f"OVERLAY licensed findings={spec['findings']} w={spec['weight']:.2f} "
            "then push kaggle_kernel_mill870_ov — still fail-closed; Submit only that TEST kernel"
        )


if __name__ == "__main__":
    main()
