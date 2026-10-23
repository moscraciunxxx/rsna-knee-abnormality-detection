"""Recompute mill vs r336 AUCs on 58 gold from downloaded kernel CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache" / "goldfold_out"
GOLD = ROOT / "data" / "gold58_ids.csv"
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


def auc(y, s) -> float:
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> None:
    gold = pd.read_csv(GOLD, dtype={"StudyInstanceUID": str})
    mill = pd.read_csv(OUT / "mill_v10_gold58.csv", dtype={"StudyInstanceUID": str})
    r336 = pd.read_csv(OUT / "r336_gold58.csv", dtype={"StudyInstanceUID": str})
    print("rows gold/mill/r336", len(gold), len(mill), len(r336))
    print("overlap mill", len(set(gold.StudyInstanceUID) & set(mill.StudyInstanceUID)))
    print("overlap r336", len(set(gold.StudyInstanceUID) & set(r336.StudyInstanceUID)))
    g = gold.set_index("StudyInstanceUID")
    m = mill.set_index("StudyInstanceUID").reindex(g.index)
    r = r336.set_index("StudyInstanceUID").reindex(g.index)
    print(f"{'finding':22} mill   r336   dlt    n_pos n_neg  mill_pos mill_neg  r336_pos r336_neg  spearman")
    mill_aucs, r_aucs = [], []
    for f in FINDINGS:
        y = g[f].to_numpy(dtype=float)
        sm = m[f].to_numpy(dtype=float)
        sr = r[f].to_numpy(dtype=float)
        am, ar = auc(y, sm), auc(y, sr)
        mill_aucs.append(am)
        r_aucs.append(ar)
        pos, neg = y == 1, y == 0
        rho = float(pd.Series(sm).corr(pd.Series(sr), method="spearman"))
        print(
            f"{f:22} {am:6.4f} {ar:6.4f} {ar-am:6.4f} {int(pos.sum()):5d} {int(neg.sum()):5d}  "
            f"{sm[pos].mean():7.3f} {sm[neg].mean():7.3f}  {sr[pos].mean():7.3f} {sr[neg].mean():7.3f}  {rho:6.3f}"
        )
    print(f"macro mill {np.nanmean(mill_aucs):.4f} r336 {np.nanmean(r_aucs):.4f}")
    # copy mill table into data for the fold script
    dest = ROOT / "data" / "mill_v10_gold58.csv"
    mill.to_csv(dest, index=False)
    r336.to_csv(ROOT / "data" / "r336_gold58.csv", index=False)
    print("wrote", dest, ROOT / "data" / "r336_gold58.csv")


if __name__ == "__main__":
    main()
