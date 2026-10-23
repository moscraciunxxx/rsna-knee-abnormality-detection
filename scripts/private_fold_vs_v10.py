#!/usr/bin/env python
"""Private 58-row expert fold: mill V10 vs any new head.

The only honest overlay gate. 58 rows is high-variance; still better than the
47-row dino3s folklore (Baker's 0.947 / MCL 0.914) that was not mill-vs-head
and produced public 0.908 after a weight-1.0 replace.

Fail-closed if mill predictions for the gold studies are missing.
Never invents AUC scores. Does not submit a Kaggle kernel.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsna_knee.guards import is_dummy_rank_trap
from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS

TRAIN_CSV = ROOT / "data" / "train.csv"
SERIES_CSV = ROOT / "data" / "train_series.csv"
GOLD_EXPORT = ROOT / "data" / "gold58_ids.csv"
PUBLIC_TEST_CSV = ROOT / "data" / "test.csv"

MILL_CANDIDATES = (
    ROOT / "data" / "mill_v10_gold58.csv",
    ROOT / "data" / "mill_v10_preds.csv",
    ROOT / "artifacts" / "mill_v10_gold58.csv",
    ROOT / "kaggle_kernel_blend_script" / "mill_v10_gold58.csv",
    ROOT / "kaggle_kernel_blend_script" / "submission.csv",
    ROOT / ".cache" / "dinov2b" / "submission.csv",
    ROOT / ".cache" / "dinov2b" / "submission_rankmean.csv",
    ROOT / ".cache" / "dinov2b" / "submission_r336.csv",
)

OTHER_GLOBS = (
    "data/*_gold58.csv",
    "artifacts/*_gold58.csv",
)

# Filenames that are labels or mill, not a new head.
_NOT_A_HEAD = frozenset(
    {
        "gold58_ids.csv",
        "train.csv",
        "mill_v10_gold58.csv",
        "mill_v10_preds.csv",
        "sample_submission.csv",
    }
)

MISSING_MILL_BANNER = "MILL V10 PREDS: MISSING — cannot compute mill-vs-new-head AUC"


def load_expert_gold(train_csv: Path = TRAIN_CSV) -> pd.DataFrame:
    if not train_csv.is_file():
        raise FileNotFoundError(f"train table not found: {train_csv}")
    train = pd.read_csv(train_csv, dtype={ID_COLUMN: str})
    missing_cols = [c for c in (ID_COLUMN, *FINDINGS) if c not in train.columns]
    if missing_cols:
        raise ValueError(f"{train_csv} missing columns {missing_cols}")
    mask = train[list(FINDINGS)].notna().all(axis=1)
    gold = train.loc[mask, [ID_COLUMN, *FINDINGS]].copy()
    gold[ID_COLUMN] = gold[ID_COLUMN].astype(str)
    for col in FINDINGS:
        gold[col] = pd.to_numeric(gold[col], errors="coerce")
    if gold[list(FINDINGS)].isna().any().any():
        raise ValueError("expert rows still contain NaN after the all-12 filter")
    uniq = set(gold[ID_COLUMN])
    if len(uniq) != len(gold):
        raise ValueError("duplicate StudyInstanceUID in expert gold")
    return gold.reset_index(drop=True)


def class_counts(gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in FINDINGS:
        y = gold[name].to_numpy(np.float64)
        n_pos = int((y == 1.0).sum())
        n_neg = int((y == 0.0).sum())
        other = int(((y != 0.0) & (y != 1.0)).sum())
        n_nan = int(np.isnan(y).sum())
        denom = n_pos + n_neg
        prev = n_pos / denom if denom else float("nan")
        rows.append(
            {
                "finding": name,
                "n": int(len(y)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_other": other,
                "n_nan": n_nan,
                "prevalence": prev,
            }
        )
    return pd.DataFrame(rows)


def _read_pred_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={ID_COLUMN: str})
    if ID_COLUMN not in table.columns:
        raise ValueError(f"{path} has no {ID_COLUMN}")
    missing = [c for c in FINDINGS if c not in table.columns]
    if missing:
        raise ValueError(f"{path} missing findings {missing} (Baker's apostrophe required)")
    out = table.loc[:, [ID_COLUMN, *FINDINGS]].copy()
    out[ID_COLUMN] = out[ID_COLUMN].astype(str)
    for col in FINDINGS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.drop_duplicates(ID_COLUMN, keep="first")
    return out.reset_index(drop=True)


def describe_candidate(path: Path, gold_ids: set[str], public_test_ids: set[str]) -> dict[str, object]:
    rec: dict[str, object] = {
        "path": str(path),
        "exists": path.is_file(),
        "n_rows": 0,
        "n_overlap_gold": 0,
        "n_overlap_public_test": 0,
        "usable_as_mill": False,
        "reason": "not found",
    }
    if not path.is_file():
        return rec
    blocked = {GOLD_EXPORT.resolve()}
    if TRAIN_CSV.is_file():
        blocked.add(TRAIN_CSV.resolve())
    if path.resolve() in blocked:
        rec["n_rows"] = 0
        rec["reason"] = "label table, not mill predictions"
        return rec
    try:
        table = _read_pred_table(path)
    except Exception as exc:
        rec["reason"] = f"unreadable as 12-finding table: {exc}"
        return rec
    ids = set(table[ID_COLUMN])
    rec["n_rows"] = int(len(table))
    rec["n_overlap_gold"] = int(len(ids & gold_ids))
    rec["n_overlap_public_test"] = int(len(ids & public_test_ids))
    if rec["n_overlap_gold"] == 0:
        if rec["n_overlap_public_test"] and rec["n_rows"] <= 3:
            rec["reason"] = "public-test submission only (0 gold overlap)"
        else:
            rec["reason"] = "0 gold StudyInstanceUID overlap"
        return rec
    rec["usable_as_mill"] = True
    rec["reason"] = f"overlaps {rec['n_overlap_gold']}/58 gold studies"
    return rec


def align_scores(
    gold: pd.DataFrame, preds: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    merged = gold.merge(preds, on=ID_COLUMN, how="inner", suffixes=("_true", "_pred"))
    y_true = np.column_stack(
        [merged[f"{name}_true"].to_numpy(np.float64) for name in FINDINGS]
    )
    y_score = np.column_stack(
        [merged[f"{name}_pred"].to_numpy(np.float64) for name in FINDINGS]
    )
    return merged, y_true, y_score


def print_label_stats(gold: pd.DataFrame) -> None:
    counts = class_counts(gold)
    print("=== 58 expert labels ===")
    print(f"source: {TRAIN_CSV}")
    print(f"export: {GOLD_EXPORT} (exists={GOLD_EXPORT.is_file()})")
    print(f"n_studies={len(gold)} unique={gold[ID_COLUMN].nunique()}")
    print(f"schema: {list(SUBMISSION_COLUMNS)}")
    print(f"{'finding':22} {'n_pos':>5} {'n_neg':>5} {'n_other':>7} {'n_nan':>5} {'prev':>6}")
    for row in counts.itertuples(index=False):
        print(
            f"{row.finding:22} {row.n_pos:5d} {row.n_neg:5d} {row.n_other:7d} "
            f"{row.n_nan:5d} {row.prevalence:6.3f}"
        )
    print(
        "missingness: 0/58 on all 12 findings among expert rows; "
        "4349/4407 train studies have all 12 as NaN."
    )
    if SERIES_CSV.is_file():
        series = pd.read_csv(SERIES_CSV, dtype={ID_COLUMN: str})
        gold_ids = set(gold[ID_COLUMN])
        gser = series[series[ID_COLUMN].isin(gold_ids)]
        print(
            f"train_series: {len(series)} rows / {series[ID_COLUMN].nunique()} studies; "
            f"gold coverage {gser[ID_COLUMN].nunique()}/58 studies, {len(gser)} series."
        )
        if "Anatomical_Plane" in gser.columns:
            print("gold planes", gser["Anatomical_Plane"].value_counts().to_dict())
    train_dicom = ROOT / "data" / "train_series"
    print(f"local train_series DICOM tree: {train_dicom.is_dir()}")
    test_dicom = ROOT / "data" / "test_series"
    n_dcm = sum(1 for _ in test_dicom.rglob("*.dcm")) if test_dicom.is_dir() else 0
    print(f"local test_series DICOM files: {n_dcm}")
    if PUBLIC_TEST_CSV.is_file():
        public = pd.read_csv(PUBLIC_TEST_CSV, dtype={ID_COLUMN: str})
        overlap = set(gold[ID_COLUMN]) & set(public[ID_COLUMN].astype(str))
        print(f"gold overlap with public test.csv: {len(overlap)}")
    print()


def print_mill_scan(records: list[dict[str, object]]) -> None:
    print("=== mill V10 prediction scan ===")
    if not records:
        print("no candidate paths")
        print()
        return
    for rec in records:
        flag = "USABLE" if rec["usable_as_mill"] else "NO"
        print(
            f"[{flag}] {rec['path']}\n"
            f"        exists={rec['exists']} n_rows={rec['n_rows']} "
            f"gold_overlap={rec['n_overlap_gold']} "
            f"public_test_overlap={rec['n_overlap_public_test']}\n"
            f"        {rec['reason']}"
        )
    print()


def print_auc_table(
    gold: pd.DataFrame,
    mill: pd.DataFrame,
    others: list[tuple[str, pd.DataFrame]],
) -> None:
    mill_merged, y_true, y_mill = align_scores(gold, mill)
    n = len(mill_merged)
    if n == 0:
        print("mill table overlaps 0 gold studies — no AUC")
        return
    if not np.isfinite(y_mill).all():
        print("WARNING: mill scores contain NaN/inf on the overlap; those findings will be skipped.")
    mill_auc = per_finding_roc_auc(y_true, y_mill)
    other_aucs: list[tuple[str, np.ndarray, int]] = []
    for name, table in others:
        merged, y_t, y_s = align_scores(gold, table)
        # Compare on the mill overlap (inner of gold ∩ mill ∩ other).
        keep = mill_merged[ID_COLUMN].isin(set(merged[ID_COLUMN]))
        mill_ids = mill_merged.loc[keep, ID_COLUMN].tolist()
        if not mill_ids:
            print(f"WARNING: {name} has 0 overlap with mill∩gold; skipped (no invented scores).")
            continue
        order = {uid: i for i, uid in enumerate(merged[ID_COLUMN])}
        idx = [order[uid] for uid in mill_ids]
        y_true_o = y_t[idx]
        y_score_o = y_s[idx]
        other_aucs.append((name, per_finding_roc_auc(y_true_o, y_score_o), len(idx)))

    dummy = is_dummy_rank_trap(y_mill)
    print("=== per-finding ROC AUC (overlap only; no imputation) ===")
    print(f"n_overlap mill∩gold = {n} / {len(gold)}")
    if n < len(gold):
        print("WARNING: not the full 58-row gate.")
    if dummy:
        print("WARNING: mill scores look like the dummy-rank trap {0, 0.5, 1}.")
    headers = ["finding", "n_pos", "n_neg", "mill"]
    for name, _, n_o in other_aucs:
        headers.append(f"{name}")
        headers.append(f"d_{name}")
    print(" ".join(f"{h:>16}" if i else f"{h:22}" for i, h in enumerate(headers)))

    for i, finding in enumerate(FINDINGS):
        y = mill_merged[f"{finding}_true"].to_numpy(np.float64)
        n_pos = int((y == 1.0).sum())
        n_neg = int((y == 0.0).sum())
        mill_v = mill_auc[i]
        mill_s = f"{mill_v:.4f}" if np.isfinite(mill_v) else "undef"
        line = f"{finding:22} {n_pos:16d} {n_neg:16d} {mill_s:>16}"
        for _, aucs, _ in other_aucs:
            ov = aucs[i]
            if not np.isfinite(mill_v) or not np.isfinite(ov):
                line += f" {'undef':>16} {'n/a':>16}"
            else:
                line += f" {ov:16.4f} {ov - mill_v:16.4f}"
        print(line)

    mill_macro = macro_roc_auc(y_true, y_mill)
    print(f"macro (nanmean of 12) mill={mill_macro:.4f}")
    for name, aucs, n_o in other_aucs:
        finite = aucs[np.isfinite(aucs)]
        other_macro = float(np.mean(finite)) if len(finite) else float("nan")
        delta = other_macro - mill_macro if np.isfinite(other_macro) else float("nan")
        print(f"macro (nanmean of 12) {name}={other_macro:.4f}  delta={delta:+.4f}  n={n_o}")
    print()
    print("Overlay rule: 1–2 findings only if Δ vs mill is quoted and positive; weight ≪ 0.50.")
    print("Never weight-1.0 replace without that quoted Δ. dino3s MCL+Baker's @1.00 is ruled out.")
    print()


def print_missing_mill_kernel() -> None:
    print(MISSING_MILL_BANNER)
    print(
        "Why a local mill-vs-new-head AUC cannot be computed yet:\n"
        "  1. Mill V10 (55779066, kernel vitaliecervinschi/rsna-knee-dino-coatnet-script-v1)\n"
        "     is a scored TEST inference kernel. It walks test_series / test.csv, not the\n"
        "     58 expert TRAIN studies.\n"
        "  2. This repo has no mill oof, cached logits, or gold-aligned submission for those\n"
        "     58 StudyInstanceUIDs. Public-test caches under .cache/dinov2b/ are 3 rows with\n"
        "     0 gold overlap. LLM report labels are not mill encodings.\n"
        "  3. Train DICOMs are not local (data/train_series/ is absent). Metadata for all 58\n"
        "     gold studies exists in data/train_series.csv (336 series). Images live on the\n"
        "     Kaggle competition mount only.\n"
    )
    print("=== exact next kernel (Save Version — DO NOT Submit to Competition) ===")
    print(
        "Smallest job: encode the 58 gold studies with mill V10, write mill_v10_gold58.csv.\n"
        "\n"
        "Fork: kaggle_kernel_blend_script/main.py\n"
        "  slug     vitaliecervinschi/rsna-knee-dino-coatnet-script-v1\n"
        "  GPU      NvidiaTeslaT4 (T4×2), internet off, same docker pin as mill V10\n"
        "  sources  competition rsna-knee-abnormality-detection\n"
        "           datasets dreaddevelopment/raptor-knee-maxspan,\n"
        "                    dreaddevelopment/raptor-knee-widedense,\n"
        "                    tonylica/rsna-knee-bend-dinov3-0917-repro-assets\n"
        "           kernel   romantamrazov/rsna-knee-dinosaur-v3-train\n"
        "           model    metaresearch/dinov2/PyTorch/small/1\n"
        "\n"
        "GOLD_ONLY redirect (do not retrain, do not clone ruled-out families):\n"
        "  gold = train.csv rows where all 12 findings are notna  # n=58\n"
        "  Retarget every test.csv / test_series.csv / walk('test_series') /\n"
        "  SERIES_ROOT=test_series site to train_series filtered to those 58 UIDs.\n"
        "  Mill hardcodes test at four blocks in main.py:\n"
        "    infer_from_package  ~test.csv + walk('test_series')\n"
        "    DINOv2 cache        SERIES_ROOT / ser_csv  (test_series fallback)\n"
        "    RadImageNet e10/e13 _rad_main  test.csv + walk('test_series')\n"
        "    dual CoAtNet        find_test_root test_ids + test_series.csv\n"
        "  Keep mill fusion: 20-member DINOv2-S + RadImageNet (Baker's/Fracture excluded\n"
        "  inside mill) + dual CoAtNet rank blend. TIME_BUDGET 8h is enough for 58 studies\n"
        "  (336 series) — this is not the hidden-test wall.\n"
        "\n"
        "Write /kaggle/working/mill_v10_gold58.csv with the official header including Baker's.\n"
        "Download to data/mill_v10_gold58.csv and re-run this script.\n"
        "Do NOT click Submit to Competition: that scores hidden test and burns GPU quota.\n"
        "Do NOT native384 / mill-RadImageNet crop / CNN-tri all-12 / dino3s 1.0 replace /\n"
        "tta-max mill TTA in this kernel.\n"
    )
    print("Overlay-candidate findings: none until mill encodings exist.")
    print()


def discover_other_heads(explicit: list[tuple[str, Path]], mill_path: Path | None) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = list(explicit)
    seen = {p.resolve() for _, p in found}
    if mill_path is not None:
        seen.add(mill_path.resolve())
    for pattern in OTHER_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.name in _NOT_A_HEAD:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append((path.stem, path))
    return found


def parse_other(values: list[str] | None) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"--other expects name=path, got {raw!r}")
        name, path_s = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--other missing name in {raw!r}")
        out.append((name, Path(path_s)))
    return out


def pick_mill(
    gold_ids: set[str],
    public_ids: set[str],
    explicit: Path | None,
) -> tuple[Path | None, list[dict[str, object]]]:
    env = os.environ.get("MILL_V10_PREDS", "").strip()
    ordered: list[Path] = []
    if explicit is not None:
        ordered.append(explicit)
    if env:
        ordered.append(Path(env))
    ordered.extend(MILL_CANDIDATES)
    seen: set[Path] = set()
    uniq: list[Path] = []
    for path in ordered:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)

    recs = [describe_candidate(p, gold_ids, public_ids) for p in uniq]
    usable = [r for r in recs if r["usable_as_mill"]]
    mill_path = Path(str(usable[0]["path"])) if usable else None
    return mill_path, recs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_CSV)
    parser.add_argument("--mill", type=Path, default=None, help="mill V10 pred CSV aligned by StudyInstanceUID")
    parser.add_argument(
        "--other",
        action="append",
        default=[],
        help="repeatable name=path for a new head CSV",
    )
    args = parser.parse_args(argv)

    gold = load_expert_gold(args.train)
    print_label_stats(gold)

    public_ids: set[str] = set()
    if PUBLIC_TEST_CSV.is_file():
        public_ids = set(
            pd.read_csv(PUBLIC_TEST_CSV, dtype={ID_COLUMN: str})[ID_COLUMN].astype(str)
        )
    mill_path, scan = pick_mill(set(gold[ID_COLUMN]), public_ids, args.mill)
    print_mill_scan(scan)

    if mill_path is None:
        print_missing_mill_kernel()
        others = discover_other_heads(parse_other(args.other), mill_path=None)
        if others:
            print("=== other head files (not scored; mill missing) ===")
            for name, path in others:
                print(f"  {name}: {path} exists={path.is_file()}")
            print("Heads are not scored against gold without mill — that is the 47-row folklore trap.")
            print()
        return 2

    mill = _read_pred_table(mill_path)
    print(f"MILL V10 PREDS: {mill_path}")
    other_tables: list[tuple[str, pd.DataFrame]] = []
    for name, path in discover_other_heads(parse_other(args.other), mill_path):
        if not path.is_file():
            print(f"WARNING: other head {name} not found at {path}")
            continue
        try:
            table = _read_pred_table(path)
        except Exception as exc:
            print(f"WARNING: other head {name} unreadable: {exc}")
            continue
        other_tables.append((name, table))
        print(f"OTHER {name}: {path} n={len(table)}")
    print()
    print_auc_table(gold, mill, other_tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
