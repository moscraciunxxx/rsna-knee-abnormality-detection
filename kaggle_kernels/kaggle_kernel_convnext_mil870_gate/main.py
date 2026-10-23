"""PRIVATE GATE: ConvNeXt MIL reader alone on seed-2026 LLM-soft holdout n=870.

Save Version only. DO NOT SUBMIT. Train-holdout IDs must never hit the LB.
Reader assets: goodpjw2008/rsna-knee-2-5d-convnext-reader (cnxt_v0_fold*.pt).
Author claim: reader alone ~0.929 public; stack+30% = 0.944 (not our CLI).
Hard guards: TIME_BUDGET_S + daemon os._exit watchdog + subprocess timeout.
Push with: kaggle kernels push -p kaggle_kernel_convnext_mil870_gate -t 5400
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

TIME_BUDGET_S = int(1.5 * 3600)  # 5400s; match kaggle -t
HARD_LIMIT_S = TIME_BUDGET_S
T0 = time.perf_counter()
EXPECTED_HOLDOUT_N = 870


def _watchdog() -> None:
    time.sleep(HARD_LIMIT_S)
    print(f"[cnxt870] WATCHDOG hard kill after {HARD_LIMIT_S}s", flush=True)
    os._exit(99)


threading.Thread(target=_watchdog, daemon=True, name="hard-watchdog").start()

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"
HOLD_SEED = 2026
HOLD_FRAC = 0.2


def _budget_ok(tag: str) -> bool:
    elapsed = time.perf_counter() - T0
    print(f"[cnxt870] {tag} elapsed={elapsed:.1f}s budget={TIME_BUDGET_S}s", flush=True)
    return elapsed < TIME_BUDGET_S


def _find_real_comp() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for path in candidates:
        if (
            (path / "train.csv").is_file()
            and (path / "train_series.csv").is_file()
            and (path / "train_series").is_dir()
        ):
            print(f"[cnxt870] competition root {path}", flush=True)
            return path
    # last-resort scan
    base = Path("/kaggle/input")
    if base.is_dir():
        for path in base.rglob("train_series.csv"):
            root = path.parent
            if (root / "train.csv").is_file() and (root / "train_series").is_dir():
                print(f"[cnxt870] competition root (scan) {root}", flush=True)
                return root
    raise FileNotFoundError("competition train_series not mounted")


def _find_file(name: str) -> Path:
    skip = {"train_series", "test_series", "test_images", "train_images"}
    base = Path("/kaggle/input")
    hits: list[Path] = []
    if base.is_dir():
        stack = [base]
        depth = 0
        while stack and depth < 120:
            cur = stack.pop(0)
            depth += 1
            try:
                kids = list(cur.iterdir())
            except OSError:
                continue
            names = {k.name for k in kids}
            if name in names:
                hits.append(cur / name)
            for kid in kids:
                if kid.is_dir() and kid.name not in skip and not kid.name.startswith("1.2."):
                    stack.append(kid)
    if not hits:
        raise FileNotFoundError(name)
    print(f"[cnxt870] found {name} at {hits[0]}", flush=True)
    return hits[0]


def _gold_ids(train: pd.DataFrame) -> set[str]:
    cols = [c for c in LABELS if c in train.columns]
    if not cols:
        raise RuntimeError("no label columns in train.csv")
    mask = train[cols].notna().all(axis=1)
    gold = set(train.loc[mask, ID_COL].astype(str))
    if len(gold) != 58:
        raise RuntimeError(f"gold count {len(gold)} != 58; refusing holdout")
    return gold


def _holdout_ids(train: pd.DataFrame, v4: pd.DataFrame) -> tuple[str, ...]:
    """Same seed-2026 LLM-soft split as mill / coatnet870 / effnet870."""
    gold = _gold_ids(train)
    labels = v4.drop_duplicates(ID_COL).copy()
    labels[ID_COL] = labels[ID_COL].astype(str)
    train_ids = [uid for uid in labels[ID_COL].tolist() if uid not in gold]
    if set(train_ids) & gold:
        raise RuntimeError("gold 58 leaked into train ids")
    rng = np.random.default_rng(int(HOLD_SEED))
    perm = rng.permutation(len(train_ids))
    n_ho = max(1, min(len(train_ids) - 1, int(round(len(train_ids) * float(HOLD_FRAC)))))
    chosen = np.sort(perm[:n_ho])
    ids = tuple(train_ids[int(i)] for i in chosen.tolist())
    if set(ids) & gold:
        raise RuntimeError("holdout overlapped gold 58")
    if len(ids) != EXPECTED_HOLDOUT_N:
        raise RuntimeError(f"expected n={EXPECTED_HOLDOUT_N}, got {len(ids)}")
    return ids


def _stage_holdout_comp(ids: tuple[str, ...], real: Path) -> Path:
    dest = Path("/kaggle/working/_holdout_comp")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    ser = pd.read_csv(real / "train_series.csv", dtype={ID_COL: str, "SeriesInstanceUID": str})
    ser = ser[ser[ID_COL].isin(ids)].copy()
    pd.DataFrame({ID_COL: list(ids)}).to_csv(dest / "test.csv", index=False)
    ser.to_csv(dest / "test_series.csv", index=False)
    rows = [{ID_COL: uid, **{f: 0.5 for f in LABELS}} for uid in ids]
    pd.DataFrame(rows)[[ID_COL, *LABELS]].to_csv(dest / "sample_submission.csv", index=False)
    # stub train so infer.py that peeks train does not explode
    pd.DataFrame({ID_COL: ["stub-train-0", "stub-train-1"]}).to_csv(dest / "train.csv", index=False)
    src_root = real / "train_series"
    dst_root = dest / "test_series"
    dst_root.mkdir()
    for uid in ids:
        src = src_root / uid
        dst = dst_root / uid
        if src.is_dir():
            os.symlink(src, dst, target_is_directory=True)
    print(f"[cnxt870] staged holdout comp studies={len(ids)} series_rows={len(ser)}", flush=True)
    return dest


def main() -> None:
    print("[cnxt870] PRIVATE GATE ConvNeXt MIL reader on holdout. DO NOT SUBMIT.", flush=True)
    print(f"[cnxt870] Hard watchdog {HARD_LIMIT_S}s. Keep 55779066. Never Submit this gate.", flush=True)
    if not _budget_ok("start"):
        raise SystemExit(0)

    real = _find_real_comp()
    train = pd.read_csv(real / "train.csv", dtype={ID_COL: str})
    v4_path = _find_file("llm_labels_v4_blend.csv")
    v4 = pd.read_csv(v4_path, dtype={ID_COL: str})
    ids = _holdout_ids(train, v4)
    print(f"[cnxt870] holdout n={len(ids)} leaked_vs_gold58=0 seed={HOLD_SEED}", flush=True)

    hold = _stage_holdout_comp(ids, real)
    os.environ["RSNA_COMP_ROOT"] = str(hold)

    ckpt0 = _find_file("cnxt_v0_fold0.pt")
    assets = ckpt0.parent
    ckpts = sorted(glob.glob(str(assets / "cnxt_v0_fold*.pt")))
    assert len(ckpts) >= 3, f"need 3 fold ckpts, found {ckpts}"

    own = Path("/kaggle/working/own_src")
    own.mkdir(parents=True, exist_ok=True)
    for name in ("infer.py", "knee.py", "preprocess.py"):
        src = assets / name
        if src.is_file():
            shutil.copy2(src, own / name)
    assert (own / "infer.py").is_file(), "infer.py missing from reader dataset"

    target = Path("/kaggle/working/_own_env")
    wheels = list(assets.glob("*.whl"))
    if wheels:
        target.mkdir(exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
             "--target", str(target), "--find-links", str(assets)] + [w.name for w in wheels],
            check=False,
        )
        sys.path.insert(0, str(target))

    out_csv = Path("/kaggle/working/_own.csv")
    cmd = [sys.executable, str(own / "infer.py"), str(hold), "test", str(out_csv), *ckpts]
    print("[cnxt870] running reader:", " ".join(cmd[:6]), f"... +{len(ckpts)} ckpts", flush=True)
    if not _budget_ok("pre-infer"):
        print("[cnxt870] budget exhausted before infer; abort without license", flush=True)
        raise SystemExit(0)
    remaining = max(30, int(TIME_BUDGET_S - (time.perf_counter() - T0) - 60))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(own) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        rc = subprocess.call(cmd, env=env, timeout=remaining)
    except subprocess.TimeoutExpired:
        print(f"[cnxt870] infer subprocess timeout after {remaining}s; abort without license", flush=True)
        raise SystemExit(0)
    if rc != 0 or not out_csv.is_file():
        raise RuntimeError(f"reader infer failed rc={rc}")

    pred = pd.read_csv(out_csv, dtype={ID_COL: str})
    if list(pred.columns) != [ID_COL, *LABELS]:
        pred = pred[[ID_COL, *LABELS]]
    if "Baker's" not in pred.columns:
        raise RuntimeError("Baker's column missing")
    pred = pred.set_index(ID_COL).reindex(list(ids)).reset_index()
    if pred[LABELS].isna().any().any():
        raise RuntimeError("non-finite / missing preds after reindex")
    gate = Path("/kaggle/working/convnext_holdout870.csv")
    pred.to_csv(gate, index=False)
    pd.DataFrame({ID_COL: list(ids)}).to_csv("/kaggle/working/convnext870_ids.csv", index=False)
    for p in (Path("/kaggle/working/submission.csv"), out_csv):
        if p.exists():
            p.unlink()
            print(f"[cnxt870] removed {p.name}. DO NOT SUBMIT.", flush=True)
    Path("/kaggle/working/DO_NOT_SUBMIT_HOLDOUT.txt").write_text(
        "convnext_holdout870.csv is the seed-2026 train holdout, not the hidden test.\n"
        "Keep 55779066 until scripts/saturday_after_convnext.py licenses a TEST overlay.\n",
        encoding="utf-8",
    )
    print(f"[cnxt870] wrote {gate} rows={len(pred)}. DO NOT SUBMIT.", flush=True)
    _budget_ok("done")


if __name__ == "__main__":
    main()