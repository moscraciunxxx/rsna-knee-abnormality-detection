"""Write data/folds.csv from the shipped study-fold assignment. Does not train."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna_knee.folds import assign_study_folds
from rsna_knee.schema import ID_COLUMN


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    folds = assign_study_folds(train)
    out = ROOT / "data" / "folds.csv"
    folds.to_csv(out, index=False)
    print(f"wrote {out} n={len(folds)} gold={int(folds['is_gold'].sum())}")


if __name__ == "__main__":
    main()
