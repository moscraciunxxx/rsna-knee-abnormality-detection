"""Copy goldfold mill V10, retarget test→seed-2026 LLM-soft holdout (~870).

Save Version only. Do not Submit: output IDs are train holdout, not hidden test.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "kaggle_kernel_goldfold"
DST_DIR = ROOT / "kaggle_kernel_mill_holdout"

HOLD_HELPERS = r'''
GOLD_ONLY = True  # use train_series encode path
HOLDOUT_ONLY = True  # LLM-soft holdout seed 2026; NOT gold-58 folklore. DO NOT SUBMIT.
HOLD_SEED = 2026
HOLD_FRAC = 0.2
_HOLD_CACHE = None

def _find_v4_csv():
    skip = {"train_series", "test_series", "test_images"}
    base = Path("/kaggle/input")
    hits = []
    if base.is_dir():
        stack = [base]
        depth = 0
        while stack and depth < 80:
            cur = stack.pop(0)
            depth += 1
            try:
                kids = list(cur.iterdir())
            except OSError:
                continue
            names = {k.name for k in kids}
            if "llm_labels_v4_blend.csv" in names:
                hits.append(cur / "llm_labels_v4_blend.csv")
            for k in kids:
                if k.is_dir() and k.name not in skip and not k.name.startswith("1.2."):
                    stack.append(k)
    if not hits:
        raise FileNotFoundError("steven v4 llm_labels_v4_blend.csv not mounted")
    return hits[0]

def _holdout_id_tuple():
    global _HOLD_CACHE
    if _HOLD_CACHE is not None:
        return _HOLD_CACHE
    train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
    v4 = pd.read_csv(_find_v4_csv(), dtype={"StudyInstanceUID": str})
    gold_mask = train[list(TARGETS)].notna().all(axis=1)
    gold = set(train.loc[gold_mask, "StudyInstanceUID"].astype(str))
    labels = v4.copy()
    labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)
    labels = labels.drop_duplicates("StudyInstanceUID")
    train_frame = labels[~labels["StudyInstanceUID"].isin(gold)].copy()
    train_ids = tuple(train_frame["StudyInstanceUID"].tolist())
    if set(train_ids) & gold:
        raise ValueError("gold 58 leaked into mill-holdout train ids")
    rng = np.random.default_rng(int(HOLD_SEED))
    n = len(train_ids)
    perm = rng.permutation(n)
    n_ho = max(1, min(n - 1, int(round(n * float(HOLD_FRAC)))))
    holdout = np.sort(perm[:n_ho])
    ids = tuple(train_ids[int(i)] for i in holdout.tolist())
    if set(ids) & gold:
        raise ValueError("LLM-soft holdout overlapped gold 58")
    print(
        f"[mill-holdout] llm-soft holdout n={len(ids)} gold=58 excluded leaked={len(set(ids)&gold)} seed={HOLD_SEED}",
        flush=True,
    )
    _HOLD_CACHE = ids
    return ids

def _goldfold_mask(frame):
    cols = [c for c in TARGETS if c in frame.columns]
    if not cols:
        return frame.index[0:0]
    return frame[cols].notna().all(axis=1)

def _goldfold_id_set():
    return set(_holdout_id_tuple())

def _goldfold_test_df():
    return pd.DataFrame({"StudyInstanceUID": list(_holdout_id_tuple())})

def _goldfold_test_series():
    ser = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    return ser[ser["StudyInstanceUID"].isin(_goldfold_id_set())].copy()

'''

HOLD_STAGE = r'''
# mill-holdout: mill V10 on seed-2026 LLM-soft holdout (~870). NOT gold-58 folklore.
# Save Version only. DO NOT Submit to Competition (IDs are train holdout).
from pathlib import Path as _GFPath
import pandas as _gf_pd
import numpy as _gf_np
from sklearn.metrics import roc_auc_score as _gf_auc

_GF_FINDINGS = TARGETS
_GF_ID = "StudyInstanceUID"
_gf_work = _GFPath("/kaggle/working")
_gf_primary = _gf_work / "submission.csv"
print("[mill-holdout] mill encode of LLM-soft holdout finished; scoring vs v4>=0.5. DO NOT SUBMIT.", flush=True)

def _gf_macro(y, s):
    per = []
    for j, name in enumerate(_GF_FINDINGS):
        yt = y[:, j]
        if len(_gf_np.unique(yt)) < 2:
            per.append(float("nan"))
            continue
        try:
            per.append(float(_gf_auc(yt, s[:, j])))
        except Exception:
            per.append(float("nan"))
    vals = [v for v in per if v == v]
    return per, (float(_gf_np.mean(vals)) if vals else float("nan"))

try:
    if not _gf_primary.is_file():
        raise RuntimeError("mill-holdout: mill submission.csv missing")
    mill = _gf_pd.read_csv(_gf_primary, dtype={_GF_ID: str})
    mill.to_csv(_gf_work / "mill_v10_holdout870.csv", index=False)
    ho_ids = list(_holdout_id_tuple())
    gold = set(
        _gf_pd.read_csv(ROOT / "train.csv", dtype={_GF_ID: str})
        .loc[lambda t: t[list(_GF_FINDINGS)].notna().all(axis=1), _GF_ID]
        .astype(str)
    )
    leaked = set(ho_ids) & gold
    print(f"[mill-holdout] n={len(ho_ids)} leaked_vs_gold58={len(leaked)}", flush=True)
    if leaked:
        raise RuntimeError("gold 58 leaked into mill-holdout IDs")
    v4 = _gf_pd.read_csv(_find_v4_csv(), dtype={_GF_ID: str})
    v4[_GF_ID] = v4[_GF_ID].astype(str)
    v4 = v4.drop_duplicates(_GF_ID).set_index(_GF_ID)
    missing_v4 = [i for i in ho_ids if i not in v4.index]
    if missing_v4:
        raise RuntimeError(f"v4 labels missing for {len(missing_v4)} holdout UIDs")
    y = (v4.loc[ho_ids, list(_GF_FINDINGS)].to_numpy(dtype=float) >= 0.5).astype(float)
    mill = mill.drop_duplicates(_GF_ID).set_index(_GF_ID)
    missing_mill = [i for i in ho_ids if i not in mill.index]
    if missing_mill:
        print(
            f"[mill-holdout] MILL V10 PREDS: MISSING — cannot compute mill-vs-holdout AUC; missing {len(missing_mill)}/{len(ho_ids)}",
            flush=True,
        )
        raise RuntimeError("MILL V10 PREDS: MISSING")
    mill_s = mill.loc[ho_ids, list(_GF_FINDINGS)].to_numpy(dtype=float)
    mill_per, mill_macro = _gf_macro(y, mill_s)
    print(f"[mill-holdout] mill vs llm-soft holdout n={len(ho_ids)} macro={mill_macro:.4f}", flush=True)
    for name, auc in zip(_GF_FINDINGS, mill_per):
        print(f"[mill-holdout] mill {name:22s} {auc:.4f}", flush=True)
    print(
        f"[mill-holdout] mill_holdout_macro={mill_macro:.4f} n={len(ho_ids)} "
        f"| not gold-58 folklore | not overlay | not submit | Baker's ok | shape={mill_s.shape}",
        flush=True,
    )
except Exception as _gf_err:
    import traceback as _gf_tb
    print(f"[mill-holdout] failed: {type(_gf_err).__name__}: {_gf_err}", flush=True)
    _gf_tb.print_exc()
'''


def main() -> None:
    DST_DIR.mkdir(exist_ok=True)
    text = (SRC_DIR / "main.py").read_text(encoding="utf-8")
    old_helpers = """GOLD_ONLY = True  # 58 expert train studies; DO NOT submit this kernel to the LB

def _goldfold_mask(frame):
    cols = [c for c in TARGETS if c in frame.columns]
    if not cols:
        return frame.index[0:0]
    return frame[cols].notna().all(axis=1)

def _goldfold_id_set():
    tr = pd.read_csv(ROOT / 'train.csv', dtype={'StudyInstanceUID': str})
    return set(tr.loc[_goldfold_mask(tr), 'StudyInstanceUID'].astype(str))

def _goldfold_test_df():
    return pd.DataFrame({'StudyInstanceUID': sorted(_goldfold_id_set())})

def _goldfold_test_series():
    ser = pd.read_csv(ROOT / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    return ser[ser['StudyInstanceUID'].isin(_goldfold_id_set())].copy()
"""
    if old_helpers not in text:
        raise SystemExit("goldfold helpers block not found")
    text = text.replace(old_helpers, HOLD_HELPERS, 1)
    marker = "# goldfold: mill V10 + nursrijan r336 on the 58 expert TRAIN studies."
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit("goldfold stage marker not found")
    text = text[:idx] + HOLD_STAGE
    (DST_DIR / "main.py").write_text(text, encoding="utf-8")
    meta = (SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8")
    meta = meta.replace(
        "vitaliecervinschi/rsna-knee-goldfold-v1",
        "vitaliecervinschi/rsna-knee-mill-holdout-870",
    )
    meta = meta.replace("RSNA Knee goldfold v1", "RSNA Knee mill holdout 870")
    if "stevenleehans/rsna-knee-llm-report-labels" not in meta:
        meta = meta.replace(
            '"dataset_sources": [',
            '"dataset_sources": [\n    "stevenleehans/rsna-knee-llm-report-labels",',
            1,
        )
    (DST_DIR / "kernel-metadata.json").write_text(meta, encoding="utf-8")
    print(f"wrote {DST_DIR / 'main.py'} bytes={len(text)}")


if __name__ == "__main__":
    main()
