"""Join mill V10 holdout encodings vs a new head on the 870. Never Submit here."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.fusion import FusionError, per_finding_rank_blend
from rsna_knee.mill_holdout import (
    MillHoldoutMissing,
    holdout_ids_and_y,
    join_mill_holdout,
    licensed_overlay_weights,
    vitb_loses_every_finding,
)
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mill", type=Path, required=True)
    p.add_argument("--new", type=Path, default=None)
    p.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    p.add_argument("--v4", type=Path, default=ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv")
    args = p.parse_args()
    train = pd.read_csv(args.train, dtype={ID_COLUMN: str})
    v4 = pd.read_csv(args.v4, dtype={ID_COLUMN: str})
    ids, y = holdout_ids_and_y(train, v4)
    mill_frame = pd.read_csv(args.mill, dtype={ID_COLUMN: str}) if args.mill.is_file() else None
    try:
        mill_join = join_mill_holdout(ids, mill_frame, y)
    except MillHoldoutMissing as exc:
        print(str(exc))
        raise SystemExit(2)
    print(f"mill vs llm-soft holdout n={len(mill_join.ids)} macro={mill_join.macro:.4f}")
    for name, auc in zip(FINDINGS, mill_join.per_auc):
        print(f"mill {name:22s} {auc:.4f}")
    if args.new is None or not args.new.is_file():
        print("new head CSV missing; no overlay license")
        return
    new_frame = pd.read_csv(args.new, dtype={ID_COLUMN: str})
    new_join = join_mill_holdout(ids, new_frame, y)
    print(f"new vs llm-soft holdout n={len(new_join.ids)} macro={new_join.macro:.4f}")
    print(f"{'finding':22} {'mill':7} {'new':7} {'delta':7}")
    for name, m, n in zip(FINDINGS, mill_join.per_auc, new_join.per_auc):
        dlt = (n - m) if (m == m and n == n) else float("nan")
        print(f"{name:22} {m:7.4f} {n:7.4f} {dlt:7.4f}")
    if vitb_loses_every_finding(mill_join.per_auc, new_join.per_auc):
        print("STOP_PUBLIC_TRAINS: new < mill on every finding; keep 55779066; do not Submit")
        return
    try:
        weights = licensed_overlay_weights(mill_join.per_auc, new_join.per_auc)
    except FusionError as exc:
        print(f"overlay illegal: {exc}; keep 55779066; do not Submit")
        return
    active = [f for f, w in weights.items() if w > 0]
    if not active:
        print("no finding beats mill on holdout by >0.01; do not overlay / do not submit")
        return
    print(f"licensed overlay findings={active} weights={ {k:v for k,v in weights.items() if v>0} }")


if __name__ == "__main__":
    main()
