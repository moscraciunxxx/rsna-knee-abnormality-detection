"""Drive mill-holdout join: fail-closed without mill preds; gold 58 never in holdout."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.fusion import FusionError, per_finding_rank_blend
from rsna_knee.guards import (
    kernel_discovers_competitions_mount,
    kernel_has_t4_gpu_probe,
    kernel_keeps_bakers_apostrophe,
    kernel_reads_test_report_column,
    kernel_source_text,
)
from rsna_knee.mill_holdout import (
    HOLD_SEED,
    MISSING_MILL_BANNER,
    MillHoldoutMissing,
    holdout_ids_and_y,
    join_mill_holdout,
    licensed_overlay_weights,
    mill_holdout_table,
)
from rsna_knee.schema import FINDINGS, ID_COLUMN
from rsna_knee.v4_backbone import kernel_loads_r336_as_scored_head

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "train.csv"
GOLD_CSV = ROOT / "data" / "gold58_ids.csv"
V4_CSV = ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv"
KERNEL_MILL = ROOT / "kaggle_kernel_mill_holdout" / "main.py"
KERNEL_DENSE = ROOT / "kaggle_kernel_dense12" / "main.py"
KERNEL_B = ROOT / "kaggle_kernel_pfoa_syn" / "main.py"
KERNEL_C = ROOT / "kaggle_kernel_attn25d" / "main.py"


@pytest.fixture(scope="module")
def holdout():
    if not V4_CSV.is_file():
        pytest.skip(f"steven v4 CSV missing: {V4_CSV}")
    train = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    v4 = pd.read_csv(V4_CSV, dtype={ID_COLUMN: str})
    return holdout_ids_and_y(train, v4)


def test_holdout_excludes_gold_58_and_uses_seed_2026(holdout):
    ids, y = holdout
    gold = set(pd.read_csv(GOLD_CSV, dtype={ID_COLUMN: str})[ID_COLUMN].astype(str))
    assert HOLD_SEED == 2026
    assert set(ids).isdisjoint(gold)
    assert len(ids) >= 800
    assert y.shape == (len(ids), len(FINDINGS))
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_join_fail_closes_when_mill_preds_missing(holdout):
    ids, y = holdout
    with pytest.raises(MillHoldoutMissing, match="MISSING"):
        join_mill_holdout(ids, None, y)
    empty = pd.DataFrame(columns=[ID_COLUMN, *FINDINGS])
    with pytest.raises(MillHoldoutMissing, match="MISSING"):
        join_mill_holdout(ids, empty, y)
    partial = mill_holdout_table(ids[:3], np.zeros((3, 12)))
    with pytest.raises(MillHoldoutMissing, match="MISSING"):
        join_mill_holdout(ids, partial, y)
    assert "MILL V10 PREDS: MISSING" in MISSING_MILL_BANNER


def test_join_aligns_when_mill_preds_cover_holdout(holdout):
    ids, y = holdout
    rng = np.random.default_rng(0)
    mill_s = rng.random((len(ids), 12))
    # shuffle mill row order — join must realign
    perm = rng.permutation(len(ids))
    frame = mill_holdout_table(tuple(ids[i] for i in perm), mill_s[perm])
    joined = join_mill_holdout(ids, frame, y)
    assert joined.ids == ids
    np.testing.assert_allclose(joined.mill, mill_s)
    assert joined.per_auc.shape == (12,)
    assert joined.macro == joined.macro  # not nan if labels vary


def test_fusion_still_refuses_uniform_050_all_twelve():
    rng = np.random.default_rng(1)
    n = 6
    a = pd.DataFrame(rng.random((n, 12)), columns=list(FINDINGS))
    a.insert(0, ID_COLUMN, [f"s{i}" for i in range(n)])
    b = pd.DataFrame(rng.random((n, 12)), columns=list(FINDINGS))
    b.insert(0, ID_COLUMN, [f"s{i}" for i in range(n)])
    weights = {name: 0.50 for name in FINDINGS}
    with pytest.raises(FusionError, match="CNN-tri 0.917"):
        per_finding_rank_blend(a, b, weights)


def test_licensed_overlay_refuses_r336ov_pair_and_all_zeros_when_no_delta():
    mill = np.full(12, 0.80)
    same = np.full(12, 0.80)
    w = licensed_overlay_weights(mill, same)
    assert all(v == 0.0 for v in w.values())
    # PF OA + Synovitis is the r336ov finding set even at 0.15
    new = mill.copy()
    new[FINDINGS.index("PF OA")] = 0.95
    new[FINDINGS.index("Synovitis")] = 0.95
    with pytest.raises(FusionError, match="r336ov"):
        licensed_overlay_weights(mill, new, max_findings=2, weight=0.15)
    # a single non-ruled-out finding is legal
    one = mill.copy()
    one[FINDINGS.index("Effusion")] = 0.95
    w2 = licensed_overlay_weights(mill, one, max_findings=2, weight=0.15)
    assert w2["Effusion"] == 0.15
    assert sum(1 for v in w2.values() if v > 0) == 1


def test_real_mill_holdout870_csv_joins_n_and_excludes_gold():
    mill_path = ROOT / "data" / "mill_v10_holdout870.csv"
    if not mill_path.is_file():
        pytest.skip("mill_v10_holdout870.csv not downloaded yet")
    if not V4_CSV.is_file():
        pytest.skip(f"steven v4 CSV missing: {V4_CSV}")
    train = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    v4 = pd.read_csv(V4_CSV, dtype={ID_COLUMN: str})
    ids, y = holdout_ids_and_y(train, v4)
    mill = pd.read_csv(mill_path, dtype={ID_COLUMN: str})
    joined = join_mill_holdout(ids, mill, y)
    assert len(joined.ids) == len(ids)
    assert len(joined.ids) >= 800
    gold = set(pd.read_csv(GOLD_CSV, dtype={ID_COLUMN: str})[ID_COLUMN].astype(str))
    assert set(joined.ids).isdisjoint(gold)
    assert joined.mill.shape == (len(ids), 12)
    assert joined.macro == joined.macro


def test_mill_holdout_kernel_encodes_870_not_gold58_and_does_not_read_report():
    src = kernel_source_text(KERNEL_MILL)
    assert kernel_has_t4_gpu_probe(src)
    assert kernel_discovers_competitions_mount(src)
    assert kernel_keeps_bakers_apostrophe(src)
    assert not kernel_reads_test_report_column(src)
    assert "HOLDOUT_ONLY" in src
    assert "HOLD_SEED = 2026" in src
    assert "llm-soft holdout" in src.lower() or "LLM-soft holdout" in src
    assert "mill_v10_holdout" in src or "mill_v10_holdout870" in src
    assert "DO NOT SUBMIT" in src or "do not submit" in src.lower()
    # not a gold-58-only folklore job as the scored set
    assert "_holdout_id_tuple" in src or "llm_soft_holdout_indices" in src


def test_dense12_kernel_is_harder_than_bc_v1_not_a_clone():
    src = kernel_source_text(KERNEL_DENSE)
    b = kernel_source_text(KERNEL_B)
    c = kernel_source_text(KERNEL_C)
    assert kernel_has_t4_gpu_probe(src)
    assert kernel_discovers_competitions_mount(src)
    assert kernel_keeps_bakers_apostrophe(src)
    assert not kernel_reads_test_report_column(src)
    assert not kernel_loads_r336_as_scored_head(src)
    assert "UNFREEZE_LAST = 2" in src
    assert "EPOCHS = 5" in src
    assert "SLICES_PER_PLANE = 12" in src
    assert "GatedAttentionPool" in src
    assert "RESEARCH_METHOD" in src
    # not B v1 (axial-only 24 / 2 targets / 3 ep / unfreeze 1)
    assert "B_N_SLICES = 24" not in src
    assert "B_TARGETS" not in src
    # not C v1 (GROUP=3 / 8 windows / 3 ep / unfreeze 1)
    assert "C_WINDOWS_PER_PLANE = 8" not in src
    assert "C_GROUP = 3" not in src
    assert src != b
    assert src != c
    assert "MILL_SLOT_IMG = 336" in src
    assert "refusing mill" in src.lower() or "not mill 6-slot" in src.lower()
