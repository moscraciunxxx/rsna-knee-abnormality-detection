"""Per-finding blend weights on the pinned 517-row ConvNeXt OOF table.

Uses the shipped rank blend. Does not Submit and does not push.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna_knee.fusion import CONVNEXT_OOF_K7_FINDINGS, per_finding_rank_blend
from rsna_knee.metrics import macro_roc_auc
from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS

PRED = ROOT / "artifacts" / "convnext_oof870" / "convnext_oof870.csv"
OUT = ROOT / "artifacts" / "convnext_oof870" / "k7_weight_search.json"
GRID = (0.0, 0.10, 0.15, 0.25, 0.35, 0.45)


def _frame(ids: list[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(FINDINGS))
    frame.insert(0, ID_COLUMN, ids)
    return frame[list(SUBMISSION_COLUMNS)]


def _delta(y, mill_v, new_v, weights: dict[str, float], ids: list[str]) -> float:
    mixed = per_finding_rank_blend(_frame(ids, mill_v), _frame(ids, new_v), weights)
    return float(
        macro_roc_auc(y, mixed[list(FINDINGS)].to_numpy(np.float64))
        - macro_roc_auc(y, mill_v)
    )


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    mill = pd.read_csv(ROOT / "data" / "mill_v10_holdout870.csv", dtype={ID_COLUMN: str})
    pred = pd.read_csv(PRED, dtype={ID_COLUMN: str})
    holdout, y_hold = holdout_ids_and_y(train, v4)
    y_map = {uid: y_hold[i] for i, uid in enumerate(holdout)}
    ids = pred[ID_COLUMN].astype(str).tolist()
    y = np.stack([y_map[uid] for uid in ids])
    mill_v = mill.set_index(ID_COLUMN).loc[ids, list(FINDINGS)].to_numpy(np.float64)
    new_v = pred.set_index(ID_COLUMN).loc[ids, list(FINDINGS)].to_numpy(np.float64)
    stub = [str(i) for i in range(len(ids))]

    base = {name: (0.15 if name in CONVNEXT_OOF_K7_FINDINGS else 0.0) for name in FINDINGS}
    base_delta = _delta(y, mill_v, new_v, base, stub)
    chosen = dict(base)
    trace = []
    for _pass in range(2):
        for name in CONVNEXT_OOF_K7_FINDINGS:
            best_w, best_d = chosen[name], _delta(y, mill_v, new_v, chosen, stub)
            for weight in GRID:
                trial = dict(chosen)
                trial[name] = weight
                got = _delta(y, mill_v, new_v, trial, stub)
                if got > best_d + 1e-6:
                    best_w, best_d = weight, got
            chosen[name] = best_w
            trace.append({"pass": _pass, "finding": name, "weight": best_w, "macro_delta": best_d})
            print(f"pass {_pass} {name} w={best_w:.2f} delta={best_d:+.5f}", flush=True)

    chosen_delta = _delta(y, mill_v, new_v, chosen, stub)
    rng = np.random.default_rng(2026)
    draws = [rng.integers(0, len(ids), size=len(ids)) for _ in range(400)]

    def _boot(weights: dict[str, float]) -> np.ndarray:
        out = np.empty(len(draws), dtype=np.float64)
        for i, take in enumerate(draws):
            out[i] = _delta(y[take], mill_v[take], new_v[take], weights, stub)
        return out

    base_boot = _boot(base)
    chosen_boot = _boot(chosen)
    gain = chosen_boot - base_boot
    def band(arr: np.ndarray) -> dict[str, float]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"lo": float(lo), "hi": float(hi), "mean": float(arr.mean())}

    payload = {
        "n": len(ids),
        "n_boot": 400,
        "seed": 2026,
        "do_not_submit": True,
        "uniform_0.15_point": base_delta,
        "uniform_0.15_ci95": band(base_boot),
        "chosen_point": chosen_delta,
        "chosen_ci95": band(chosen_boot),
        "gain_vs_0.15_ci95": band(gain),
        "chosen_weights": {name: chosen[name] for name in CONVNEXT_OOF_K7_FINDINGS},
        "trace": trace,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"uniform={base_delta:+.5f} chosen={chosen_delta:+.5f} "
        f"gain_ci={band(gain)['lo']:+.5f}..{band(gain)['hi']:+.5f}"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
