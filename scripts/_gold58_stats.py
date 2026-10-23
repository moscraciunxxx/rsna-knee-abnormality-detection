"""Print 58-expert-label stats from local train.csv. No invented scores."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
SERIES = ROOT / "data" / "train_series.csv"
FINDINGS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def main() -> None:
    tr = pd.read_csv(TRAIN)
    any_label = tr[FINDINGS].notna().any(axis=1)
    all12 = tr[FINDINGS].notna().all(axis=1)
    gold = tr.loc[all12].copy()
    print(f"train rows={len(tr)} unique={tr['StudyInstanceUID'].nunique()}")
    print(f"any-label={int(any_label.sum())} all-12-present={int(all12.sum())}")
    print(f"{'finding':22} n_pos n_neg n_nan   prev")
    for col in FINDINGS:
        s = tr[col]
        n_pos = int((s == 1).sum())
        n_neg = int((s == 0).sum())
        n_nan = int(s.isna().sum())
        prev = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else float("nan")
        print(f"{col:22} {n_pos:5d} {n_neg:5d} {n_nan:5d}  {prev:.3f}")

    ts = pd.read_csv(SERIES)
    print(f"\ntrain_series rows={len(ts)} studies={ts['StudyInstanceUID'].nunique()}")
    print("planes", ts["Anatomical_Plane"].value_counts().to_dict())
    gold_ids = set(gold["StudyInstanceUID"])
    gser = ts[ts["StudyInstanceUID"].isin(gold_ids)]
    print(f"gold series={len(gser)} gold studies in series={gser['StudyInstanceUID'].nunique()}")
    print("gold planes", gser["Anatomical_Plane"].value_counts().to_dict())
    per = gser.groupby("StudyInstanceUID").size()
    print("series/study gold", per.describe().to_dict())
    out = ROOT / "data" / "gold58_ids.csv"
    gold[["StudyInstanceUID", *FINDINGS]].to_csv(out, index=False)
    print(f"wrote {out} rows={len(gold)}")


if __name__ == "__main__":
    main()
