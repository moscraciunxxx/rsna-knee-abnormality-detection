"""CPU label export for the frozen head. Does not train and does not submit.

enable_gpu stays false. A blank 56926403 score is not a push.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from rsna_knee.graft_head_export import graft_head_frames
    from rsna_knee.schema import ID_COLUMN
except ImportError:
    from graft_head_export import graft_head_frames
    from schema import ID_COLUMN


def _find_named(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        if path.is_file():
            return path
    return None


def export_directory(data_root: Path, work: Path) -> tuple[int, int]:
    """Write fit targets and weights. Delete submission.csv if it appears.

    Returns fit row count and holdout row count.
    """
    train_path = data_root / "train.csv"
    if not train_path.is_file():
        found = _find_named(data_root, "train.csv")
        if found is None:
            raise FileNotFoundError("train.csv missing")
        train_path = found
    v4_path = _find_named(data_root, "llm_labels_v4_blend.csv")
    pw_path = _find_named(data_root, "report_labels_v2.csv")
    if v4_path is None or pw_path is None:
        raise FileNotFoundError("steven v4 or pilkwang labels missing")
    train = pd.read_csv(train_path, dtype={ID_COLUMN: str})
    if "Report" in train.columns:
        train = train.drop(columns=["Report"])
    v4 = pd.read_csv(v4_path, dtype={ID_COLUMN: str})
    pw = pd.read_csv(pw_path, dtype={ID_COLUMN: str})
    if "Report" in v4.columns or "Report" in pw.columns:
        raise RuntimeError("label tables must not carry Report")
    targets, weights, hold_ids = graft_head_frames(v4, pw, train)
    work.mkdir(parents=True, exist_ok=True)
    targets.to_csv(work / "graft_fit_targets.csv", index=False)
    weights.to_csv(work / "graft_fit_weights.csv", index=False)
    submission = work / "submission.csv"
    if submission.exists():
        submission.unlink()
    print(
        f"[graft-head] fit_n={len(targets)} holdout_n={len(hold_ids)} "
        "gold_in_train=0 DO NOT SUBMIT",
        flush=True,
    )
    return len(targets), len(hold_ids)


def main() -> None:
    export_directory(Path("/kaggle/input"), Path("/kaggle/working"))
