"""AUC of pilkwang LLM-soft labels vs 58 expert gold. No DICOMs required."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
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
    gold = pd.read_csv(ROOT / "data" / "gold58_ids.csv")
    llm = pd.read_csv(ROOT / ".cache" / "llm_labels" / "report_labels_v2.csv")
    print(f"gold={len(gold)} llm={len(llm)} llm_unique={llm['StudyInstanceUID'].nunique()}")
    merged = gold.merge(llm, on="StudyInstanceUID", suffixes=("_gold", "_llm"), how="left")
    print(f"merged={len(merged)}")
    aucs = []
    print(f"{'finding':22} llm_auc n_pos n_neg mean_llm")
    for col in FINDINGS:
        y = merged[f"{col}_gold"].to_numpy(dtype=float)
        s = merged[f"{col}_llm"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(s)
        y, s = y[mask], s[mask]
        auc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
        aucs.append(auc)
        print(f"{col:22} {auc:7.4f} {int((y == 1).sum()):5d} {int((y == 0).sum()):5d} {float(s.mean()):.3f}")
    print(f"macro {float(np.nanmean(aucs)):.4f}")


if __name__ == "__main__":
    main()
