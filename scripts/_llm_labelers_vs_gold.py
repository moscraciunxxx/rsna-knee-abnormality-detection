"""Compare public LLM label CSVs against 58 expert gold."""
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
SOURCES = {
    "pilkwang_v2": ROOT / ".cache" / "llm_labels" / "report_labels_v2.csv",
    "steven_full": ROOT / ".cache" / "llm_steven" / "llm_labels_full.csv",
    "steven_v2": ROOT / ".cache" / "llm_steven" / "llm_labels_v2.csv",
    "steven_v4": ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv",
    "lixin_sol56": ROOT / ".cache" / "llm_lixin" / "labels_llm_gpt56sol.csv",
    "lixin_report": ROOT / ".cache" / "llm_lixin" / "report_labels_gpt56sol.csv",
}


def main() -> None:
    gold = pd.read_csv(ROOT / "data" / "gold58_ids.csv")
    gold_ids = gold.set_index("StudyInstanceUID")
    print("gold", len(gold))
    hdr = f"{'source':16} {'n':5} {'ov':4} {'macro':7}  " + " ".join(f"{f[:6]:>7}" for f in FINDINGS)
    print(hdr)
    for name, path in SOURCES.items():
        if not path.is_file():
            print(name, "MISSING")
            continue
        df = pd.read_csv(path)
        uid = "StudyInstanceUID" if "StudyInstanceUID" in df.columns else df.columns[0]
        df = df.rename(columns={uid: "StudyInstanceUID"}).set_index("StudyInstanceUID")
        common = gold_ids.index.intersection(df.index)
        per = []
        for f in FINDINGS:
            y = gold_ids.loc[common, f].to_numpy(dtype=float)
            s = pd.to_numeric(df.loc[common, f], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(s)
            yy, ss = y[mask], s[mask]
            auc = float(roc_auc_score(yy, ss)) if len(np.unique(yy)) == 2 else float("nan")
            per.append(auc)
        macro = float(np.nanmean(per))
        print(
            f"{name:16} {len(df):5d} {len(common):4d} {macro:7.4f}  "
            + " ".join(f"{v:7.3f}" if np.isfinite(v) else f"{'nan':>7}" for v in per)
        )


if __name__ == "__main__":
    main()
