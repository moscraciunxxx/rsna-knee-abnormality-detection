"""Drive shipped steven-v4 split: gold 58 val-only, never in the train loss."""

from pathlib import Path

import pandas as pd
import pytest

from rsna_knee.schema import FINDINGS, ID_COLUMN
from rsna_knee.v4_split import (
    expert_gold_ids,
    make_steven_v4_split,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "train.csv"
GOLD_CSV = ROOT / "data" / "gold58_ids.csv"
V4_CSV = ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv"


@pytest.fixture(scope="module")
def split():
    if not V4_CSV.is_file():
        pytest.skip(f"steven v4 CSV missing: {V4_CSV}")
    train = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    v4 = pd.read_csv(V4_CSV, dtype={ID_COLUMN: str})
    return make_steven_v4_split(train, v4)


def test_val_ids_are_exactly_the_58_expert_gold(split):
    gold = pd.read_csv(GOLD_CSV, dtype={ID_COLUMN: str})
    gold_ids = set(gold[ID_COLUMN].astype(str))
    train = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    from_train = set(expert_gold_ids(train))
    assert from_train == gold_ids
    assert set(split.val_ids) == gold_ids
    assert split.n_val == 58
    assert split.y_val.shape == (58, len(FINDINGS))


def test_train_ids_contain_none_of_the_gold_58(split):
    gold = set(split.val_ids)
    assert set(split.train_ids).isdisjoint(gold)
    assert split.n_train == 4349
    assert split.y_train.shape == (split.n_train, len(FINDINGS))
    assert split.y_train.shape[1] == 12


def test_llm_soft_holdout_is_disjoint_and_excludes_gold(split):
    from rsna_knee.v4_split import llm_soft_holdout_indices

    fit, holdout = llm_soft_holdout_indices(split.n_train, frac=0.2, seed=2026)
    assert len(fit) + len(holdout) == split.n_train
    assert set(fit.tolist()).isdisjoint(set(holdout.tolist()))
    assert len(holdout) >= 1
    ho_ids = {split.train_ids[i] for i in holdout.tolist()}
    assert ho_ids.isdisjoint(set(split.val_ids))


def test_gold_uid_without_pilkwang_row_stays_val_only(split):
    missing_pilkwang = "1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220"
    assert missing_pilkwang in set(split.val_ids)
    assert missing_pilkwang not in set(split.train_ids)
    # steven v4 has all 4407 UIDs; this UID is gold so it is still excluded from train
    v4 = pd.read_csv(V4_CSV, dtype={ID_COLUMN: str})
    assert missing_pilkwang in set(v4[ID_COLUMN].astype(str))
