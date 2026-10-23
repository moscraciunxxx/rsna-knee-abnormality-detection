"""Write the private ConvNeXt OOF gate. Does not push and does not Submit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from rsna_knee.convnext_oof import OOF_FOLDS, author_v0_folds
from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import ID_COLUMN

DEST = ROOT / "kaggle_kernel_convnext_oof870"
DOCKER = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
)

def _pinned_holdout_folds() -> dict[str, int]:
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(
        ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str}
    )
    folds = author_v0_folds(train)
    ids, _y = holdout_ids_and_y(train, v4)
    pinned = {uid: int(folds.loc[uid]) for uid in ids}
    counts = {k: sum(v == k for v in pinned.values()) for k in OOF_FOLDS}
    if counts != {0: 174, 1: 183, 2: 160}:
        raise SystemExit(f"pinned OOF counts changed: {counts}")
    if any(v == -1 for v in pinned.values()):
        raise SystemExit("gold leaked into pinned holdout folds")
    return pinned


PINNED_JSON = json.dumps(_pinned_holdout_folds(), indent=None, separators=(",", ":"))

RUNNER = r'''
TIME_BUDGET_S = int(45 * 60)
HARD_LIMIT_S = TIME_BUDGET_S
T0 = time.perf_counter()
EXPECTED_HOLDOUT_N = 870


def _watchdog() -> None:
    time.sleep(HARD_LIMIT_S)
    print(f"[cnxt-oof] WATCHDOG hard kill after {HARD_LIMIT_S}s", flush=True)
    os._exit(99)


threading.Thread(target=_watchdog, daemon=True, name="hard-watchdog").start()


def _budget_ok(tag: str) -> bool:
    elapsed = time.perf_counter() - T0
    print(f"[cnxt-oof] {tag} elapsed={elapsed:.1f}s budget={TIME_BUDGET_S}s", flush=True)
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
            print(f"[cnxt-oof] competition root {path}", flush=True)
            return path
    base = Path("/kaggle/input")
    if base.is_dir():
        for path in base.rglob("train_series.csv"):
            root = path.parent
            if (root / "train.csv").is_file() and (root / "train_series").is_dir():
                print(f"[cnxt-oof] competition root (scan) {root}", flush=True)
                return root
    raise FileNotFoundError("competition train_series not mounted")


def _find_file(name: str) -> Path:
    skip = {"train_series", "test_series", "test_images", "train_images"}
    base = Path("/kaggle/input")
    hits: list[Path] = []
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
    print(f"[cnxt-oof] found {name} at {hits[0]}", flush=True)
    return hits[0]


def _gold_ids(train: pd.DataFrame) -> set[str]:
    cols = [c for c in LABELS if c in train.columns]
    mask = train[cols].notna().all(axis=1)
    gold = set(train.loc[mask, ID_COL].astype(str))
    if len(gold) != 58:
        raise RuntimeError(f"gold count {len(gold)} != 58; refusing holdout")
    return gold


def _holdout_ids(train: pd.DataFrame, v4: pd.DataFrame) -> tuple[str, ...]:
    gold = _gold_ids(train)
    labels = v4.drop_duplicates(ID_COL).copy()
    labels[ID_COL] = labels[ID_COL].astype(str)
    train_ids = [uid for uid in labels[ID_COL].tolist() if uid not in gold]
    if set(train_ids) & gold:
        raise RuntimeError("gold 58 leaked into train ids")
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(train_ids))
    n_ho = max(1, min(len(train_ids) - 1, int(round(len(train_ids) * 0.2))))
    chosen = np.sort(perm[:n_ho])
    ids = tuple(train_ids[int(i)] for i in chosen.tolist())
    if set(ids) & gold:
        raise RuntimeError("holdout overlapped gold 58")
    if len(ids) != EXPECTED_HOLDOUT_N:
        raise RuntimeError(f"expected n={EXPECTED_HOLDOUT_N}, got {len(ids)}")
    return ids


def _stage(ids: tuple[str, ...], real: Path) -> Path:
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
    pd.DataFrame({ID_COL: ["stub-train-0", "stub-train-1"]}).to_csv(dest / "train.csv", index=False)
    src_root = real / "train_series"
    dst_root = dest / "test_series"
    dst_root.mkdir()
    for uid in ids:
        src = src_root / uid
        dst = dst_root / uid
        if src.is_dir():
            os.symlink(src, dst, target_is_directory=True)
    print(f"[cnxt-oof] staged studies={len(ids)} series_rows={len(ser)}", flush=True)
    return dest


def _prepare_reader(assets: Path) -> Path:
    own = Path("/kaggle/working/own_src")
    own.mkdir(parents=True, exist_ok=True)
    for name in ("infer.py", "knee.py", "preprocess.py"):
        src = assets / name
        if src.is_file():
            shutil.copy2(src, own / name)
    if not (own / "infer.py").is_file():
        raise FileNotFoundError("infer.py missing from reader dataset")
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
    return own


def main() -> None:
    print("[cnxt-oof] PRIVATE OOF gate. One left-out fold checkpoint. DO NOT SUBMIT.", flush=True)
    print("[cnxt-oof] Keep 55779066. This file is train IDs, not the hidden test.", flush=True)
    sub = Path("/kaggle/working/submission.csv")
    if sub.exists():
        sub.unlink()
    if not _budget_ok("start"):
        raise SystemExit(0)
    real = _find_real_comp()
    train = pd.read_csv(real / "train.csv", dtype={ID_COL: str})
    v4 = pd.read_csv(_find_file("llm_labels_v4_blend.csv"), dtype={ID_COL: str})
    holdout = _holdout_ids(train, v4)
    missing = [uid for uid in holdout if uid not in PINNED_HOLDOUT_FOLDS]
    if missing:
        raise RuntimeError(f"pinned fold map missing {len(missing)} holdout ids")
    extra = [uid for uid in PINNED_HOLDOUT_FOLDS if uid not in set(holdout)]
    if extra:
        raise RuntimeError(f"pinned fold map has {len(extra)} ids outside the holdout")
    by_fold = {
        fold_id: tuple(uid for uid in holdout if int(PINNED_HOLDOUT_FOLDS[uid]) == fold_id)
        for fold_id in OOF_FOLDS
    }
    if [len(by_fold[k]) for k in OOF_FOLDS] != [174, 183, 160]:
        raise RuntimeError(
            "pinned OOF counts drifted: " + str({k: len(by_fold[k]) for k in OOF_FOLDS})
        )
    for fold_id, fold_ids in by_fold.items():
        print(f"[cnxt-oof] fold={fold_id} n={len(fold_ids)}", flush=True)
    oof_ids = tuple(uid for fold_id in OOF_FOLDS for uid in by_fold[fold_id])
    print(f"[cnxt-oof] oof_n={len(oof_ids)} leaked_vs_gold58=0 holdout_n={len(holdout)}", flush=True)
    if len(oof_ids) < 300:
        raise RuntimeError(f"OOF subset too small: {len(oof_ids)}")
    ckpt0 = _find_file("cnxt_v0_fold0.pt")
    assets = ckpt0.parent
    own = _prepare_reader(assets)
    parts: list[pd.DataFrame] = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(own) + os.pathsep + env.get("PYTHONPATH", "")
    for fold_id in OOF_FOLDS:
        fold_ids = by_fold[fold_id]
        if not fold_ids:
            continue
        if not _budget_ok(f"pre-fold-{fold_id}"):
            print("[cnxt-oof] budget exhausted; abort without a license file", flush=True)
            raise SystemExit(0)
        hold = _stage(fold_ids, real)
        ckpt = assets / f"cnxt_v0_fold{fold_id}.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        out_csv = Path(f"/kaggle/working/_fold{fold_id}.csv")
        cmd = [sys.executable, str(own / "infer.py"), str(hold), "test", str(out_csv), str(ckpt)]
        remaining = max(30, int(TIME_BUDGET_S - (time.perf_counter() - T0) - 30))
        print(f"[cnxt-oof] infer fold={fold_id} ckpt={ckpt.name} n={len(fold_ids)}", flush=True)
        try:
            rc = subprocess.call(cmd, env=env, timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[cnxt-oof] fold {fold_id} timeout; abort without a license file", flush=True)
            raise SystemExit(0)
        if rc != 0 or not out_csv.is_file():
            raise RuntimeError(f"fold {fold_id} infer failed rc={rc}")
        pred = pd.read_csv(out_csv, dtype={ID_COL: str})
        pred = pred[[ID_COL, *LABELS]].set_index(ID_COL).reindex(list(fold_ids))
        if pred[LABELS].isna().any().any():
            raise RuntimeError(f"fold {fold_id} missing preds")
        pred.insert(1, "fold", fold_id)
        parts.append(pred.reset_index())
        out_csv.unlink()
    pred = pd.concat(parts, ignore_index=True)
    if list(pred[ID_COL]) != list(oof_ids):
        pred = pred.set_index(ID_COL).reindex(list(oof_ids)).reset_index()
    if pred["fold"].isna().any() or pred[LABELS].isna().any().any():
        raise RuntimeError("OOF table has missing rows")
    gate = Path("/kaggle/working/convnext_oof870.csv")
    pred[[ID_COL, "fold", *LABELS]].to_csv(gate, index=False)
    for p in (sub, Path("/kaggle/working/submission.csv")):
        if p.exists():
            p.unlink()
            print(f"[cnxt-oof] removed {p.name}. DO NOT SUBMIT.", flush=True)
    Path("/kaggle/working/DO_NOT_SUBMIT_HOLDOUT.txt").write_text(
        "convnext_oof870.csv is the left-out-fold slice of the seed-2026 train holdout, "
        "not the hidden test.\nKeep 55779066. Do not Submit this kernel.\n",
        encoding="utf-8",
    )
    print(f"[cnxt-oof] wrote {gate} rows={len(pred)}. DO NOT SUBMIT.", flush=True)
    _budget_ok("done")


if __name__ == "__main__":
    main()
'''

HEADER = '''"""PRIVATE GATE: ConvNeXt v0 left-out folds on the seed-2026 holdout.

Scores each mill-870 study in author folds 0, 1, or 2 with only that fold's
checkpoint. Fold ids are pinned from the local sklearn 1.8.0 run of the
published fold script. Version 1 recomputed folds inside the Kaggle image and
agreed on 65 of 513 rows, so it is not a license. This file does not recompute
folds and does not read a report. Save Version only. DO NOT SUBMIT.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"
OOF_FOLDS = (0, 1, 2)
N_FOLDS = 5
FOLD_SEED = 0

'''


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    pinned_block = "PINNED_HOLDOUT_FOLDS = json.loads(r'''" + PINNED_JSON + "''')\n"
    (DEST / "main.py").write_text(HEADER + pinned_block + "\n" + RUNNER, encoding="utf-8")
    meta = {
        "id": "vitaliecervinschi/rsna-knee-convnext-oof870",
        "title": "RSNA Knee convnext oof870",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "goodpjw2008/rsna-knee-2-5d-convnext-reader",
            "stevenleehans/rsna-knee-llm-report-labels",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": DOCKER,
    }
    (DEST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
