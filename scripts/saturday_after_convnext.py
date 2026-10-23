"""After convnext-mil870-gate COMPLETE: join mill-870 vs ConvNeXt reader. Never Submit.

Writes artifacts/convnext_mil870/overlay_spec.json.
Does not touch coatnet870 or vitb870 overlay specs.
"""
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

PROTECTED = {
    (ROOT / "kaggle_kernel_mill870_ov" / "overlay_spec.json").resolve(),
    (ROOT / "artifacts" / "coatnet870" / "overlay_spec.json").resolve(),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mill", type=Path, default=ROOT / "data" / "mill_v10_holdout870.csv")
    parser.add_argument("--convnext", type=Path, required=True)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "artifacts" / "convnext_mil870" / "overlay_spec.json",
    )
    args = parser.parse_args()
    if args.spec.resolve() in PROTECTED:
        raise SystemExit("refusing to overwrite coatnet/vitb overlay_spec; pass a convnext spec path")
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(
        ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv",
        dtype={ID_COLUMN: str},
    )
    ids, y = holdout_ids_and_y(train, v4)
    mill = pd.read_csv(args.mill, dtype={ID_COLUMN: str}) if args.mill.is_file() else None
    mill_join = join_mill_holdout(ids, mill, y)
    new = pd.read_csv(args.convnext, dtype={ID_COLUMN: str}) if args.convnext.is_file() else None
    if new is None:
        raise MillHoldoutMissing("convnext_holdout870.csv missing")
    new_join = join_mill_holdout(ids, new, y)
    print(f"mill n={len(mill_join.ids)} macro={mill_join.macro:.4f}")
    print(f"convnext n={len(new_join.ids)} macro={new_join.macro:.4f}")
    print(f"{'finding':22} {'mill':7} {'cnxt':7} {'delta':7}")
    for name, left, right in zip(FINDINGS, mill_join.per_auc, new_join.per_auc):
        delta = (right - left) if (left == left and right == right) else float("nan")
        print(f"{name:22} {left:7.4f} {right:7.4f} {delta:7.4f}")
    decision, weights = decide_private_overlay(mill_join.per_auc, new_join.per_auc)
    spec = {
        "decision": decision,
        "keep": "55779066",
        "source": "convnext-mil870-gate",
        "do_not_submit_holdout_ids": True,
        "author_claim_public": 0.944,
        "author_reader_alone_public": 0.929,
        "author_claim_is_not_our_cli": True,
        "findings": [name for name, weight in (weights or {}).items() if weight > 0],
        "weight": (max((weights or {}).values()) if weights else 0.0),
    }
    args.spec.parent.mkdir(parents=True, exist_ok=True)
    args.spec.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"decision={decision} spec={args.spec}")
    print("keep 55779066. DO NOT SUBMIT vitaliecervinschi/rsna-knee-convnext-mil870-gate.")


if __name__ == "__main__":
    main()
