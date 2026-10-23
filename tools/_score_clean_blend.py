"""Score the strict-clean pool on the existing seed-2026 holdout870 files. CPU only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
sys.path.insert(0, str(ROOT))

from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN

VIEWS = ("maxspan-v5", "native384dense-v10", "maxspan-v5-reverse", "native384-v8")


def align(frame: pd.DataFrame, ids: tuple[str, ...]) -> np.ndarray:
    frame = frame.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    frame = frame.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    missing = [uid for uid in ids if uid not in frame.index]
    if missing:
        raise SystemExit(f"missing {len(missing)} holdout ids")
    return frame.loc[list(ids), list(FINDINGS)].to_numpy(dtype=np.float64)


def ranks(scores: np.ndarray) -> np.ndarray:
    out = np.empty_like(scores)
    for j in range(scores.shape[1]):
        out[:, j] = rankdata(scores[:, j], method="average")
    return out


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(
        ROOT / "data" / "external_labels" / "llm_labels_v4_blend.csv",
        dtype={ID_COLUMN: str},
    )
    ids, y = holdout_ids_and_y(train, v4)
    gold = set(pd.read_csv(ROOT / "data" / "gold58_ids.csv", dtype={ID_COLUMN: str})[ID_COLUMN])
    leak = set(ids) & gold
    print("holdout", len(ids), "gold_leak", len(leak))

    reader = align(pd.read_csv(ROOT / "artifacts" / "convnext_mil870" / "convnext_holdout870.csv"), ids)
    raw = np.load(ROOT / "artifacts" / "coatnet870" / "raptor_raw.npz")
    raw_ids = [str(x) for x in raw["study_uids"].tolist()]
    if raw_ids != list(ids):
        order = {uid: i for i, uid in enumerate(raw_ids)}
        missing = [uid for uid in ids if uid not in order]
        if missing:
            raise SystemExit(f"raptor_raw missing {len(missing)} holdout ids")
        idx = np.array([order[uid] for uid in ids])
    else:
        idx = np.arange(len(ids))
    probs = raw["raw_probabilities"][:, idx, :]
    maxspan = probs[0].astype(np.float64)

    models = {
        "goodpjw2008_reader": reader,
        "dread_maxspan_v5": maxspan,
    }
    ranked = [ranks(scores) for scores in models.values()]
    blend = np.mean(ranked, axis=0)

    rows = []
    for name, scores in {**models, "equal_rank_reader_maxspan": blend}.items():
        per = per_finding_roc_auc(y, scores)
        macro = macro_roc_auc(y, scores)
        row = {"model": name, "macro": macro}
        row.update({FINDINGS[i]: float(per[i]) for i in range(12)})
        rows.append(row)
        print(f"{name} macro {macro:.4f}")

    mill = align(pd.read_csv(ROOT / "data" / "mill_v10_holdout870.csv"), ids)
    print("mill_v10_EXCLUDED macro", f"{macro_roc_auc(y, mill):.4f}")
    print("widedense UNSCORED: no local holdout870 or OOF file")

    out = ROOT / "artifacts" / "clean_blend"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "holdout870_scores.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps({"n": len(ids), "gold_leak": len(leak), "views": list(VIEWS), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
