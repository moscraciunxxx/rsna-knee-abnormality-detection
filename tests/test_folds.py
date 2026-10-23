"""Study folds cover every train study and flag the 58 expert rows."""

from pathlib import Path

import pandas as pd

from rsna_knee.folds import N_FOLDS, assign_study_folds
from rsna_knee.schema import ID_COLUMN
from rsna_knee.v4_split import expert_gold_ids

ROOT = Path(__file__).resolve().parents[1]


def test_study_folds_match_the_written_file_and_exclude_report():
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    folds = assign_study_folds(train)
    written = pd.read_csv(ROOT / "data" / "folds.csv", dtype={ID_COLUMN: str})
    assert list(folds.columns) == [ID_COLUMN, "fold", "is_gold"]
    assert "Report" not in folds.columns
    assert list(written.columns) == list(folds.columns)
    pd.testing.assert_frame_equal(written.reset_index(drop=True), folds.reset_index(drop=True))
    assert set(folds["fold"]) == set(range(N_FOLDS))
    counts = folds["fold"].value_counts()
    assert int(counts.max() - counts.min()) <= 1
    gold = set(expert_gold_ids(train))
    flagged = set(folds.loc[folds["is_gold"] == 1, ID_COLUMN])
    assert flagged == gold
    assert len(flagged) == 58
    assert folds[ID_COLUMN].is_unique
    assert len(folds) == train[ID_COLUMN].nunique()
