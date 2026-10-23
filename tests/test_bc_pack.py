"""Drive B/C packing: not mill-slot, not GAP-linear, gold-58 never train."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.bc_pack import (
    B_BACKBONE,
    B_CROP_MM,
    B_HEAD,
    B_IMG,
    B_N_SLICES,
    B_PLANES,
    B_RESEARCH_METHOD,
    B_TARGETS,
    B_UNFREEZE_LAST,
    C_BACKBONE,
    C_CROP_MM,
    C_GROUP,
    C_HEAD,
    C_IMG,
    C_PLANES,
    C_RESEARCH_METHOD,
    C_UNFREEZE_LAST,
    C_WINDOWS_PER_PLANE,
    GatedAttentionPool,
    assert_b_pack_config,
    assert_c_pack_config,
    axial_dense_indices,
    is_gap_linear_head,
    kernel_is_frozen_gap_linear,
    kernel_uses_mill_family_backbone,
    pick_plane_series,
    specialist_fill_matrix,
    windows_2p5d_indices,
)
from rsna_knee.guards import (
    kernel_discovers_competitions_mount,
    kernel_has_t4_gpu_probe,
    kernel_keeps_bakers_apostrophe,
    kernel_reads_test_report_column,
    kernel_source_text,
)
from rsna_knee.schema import FINDINGS
from rsna_knee.v4_backbone import (
    MILL_N_SLOT,
    MILL_SLOT_CROP_MM,
    MILL_SLOT_IMG,
    is_mill_slot_recipe,
    kernel_loads_r336_as_scored_head,
)
from rsna_knee.v4_split import (
    GOLD_AUC_ABORT_THRESHOLD,
    abort_if_gold_overfit,
    should_abort_gold_overfit,
)

ROOT = Path(__file__).resolve().parents[1]
KERNEL_B = ROOT / "kaggle_kernel_pfoa_syn" / "main.py"
KERNEL_C = ROOT / "kaggle_kernel_attn25d" / "main.py"
TRAIN_CSV = ROOT / "data" / "train.csv"
GOLD_CSV = ROOT / "data" / "gold58_ids.csv"
V4_CSV = ROOT / ".cache" / "llm_steven" / "llm_labels_v4_blend.csv"


def test_b_pack_is_axial_dense_not_mill_slot_not_gap_linear():
    assert B_PLANES == ("Axial",)
    assert B_N_SLICES == 24
    assert B_IMG == 224
    assert B_CROP_MM == 0.0
    assert B_UNFREEZE_LAST == 1
    assert B_HEAD == "gated_attention_mil"
    assert B_TARGETS == ("PF OA", "Synovitis")
    assert B_BACKBONE == "resnet18"
    assert "Ilse" in B_RESEARCH_METHOD
    assert not is_mill_slot_recipe(img=B_IMG, crop_mm=B_CROP_MM, n_slot=B_N_SLICES)
    assert not is_mill_slot_recipe(
        img=B_IMG, crop_mm=B_CROP_MM, n_slot=MILL_N_SLOT
    )
    assert is_mill_slot_recipe(
        img=MILL_SLOT_IMG, crop_mm=MILL_SLOT_CROP_MM, n_slot=MILL_N_SLOT
    )
    assert not is_gap_linear_head(B_HEAD, B_UNFREEZE_LAST)
    assert is_gap_linear_head("linear", 0)
    assert_b_pack_config()


def test_c_pack_is_2p5d_attention_not_mill_backbone():
    assert C_PLANES == ("Axial", "Coronal", "Sagittal")
    assert C_WINDOWS_PER_PLANE == 8
    assert C_GROUP == 3
    assert C_IMG == 224
    assert C_CROP_MM == 0.0
    assert C_UNFREEZE_LAST == 1
    assert C_HEAD == "per_finding_gated_attention"
    assert C_BACKBONE == "resnet18"
    assert "resnet18" in C_RESEARCH_METHOD.lower()
    assert "CoAtNet" in C_RESEARCH_METHOD or "coatnet" in C_RESEARCH_METHOD.lower()
    assert not is_mill_slot_recipe(img=C_IMG, crop_mm=C_CROP_MM, n_slot=MILL_N_SLOT)
    assert not is_gap_linear_head(C_HEAD, C_UNFREEZE_LAST)
    assert_c_pack_config()


def test_axial_dense_indices_span_full_stack():
    idx = axial_dense_indices(100, 24)
    assert idx.dtype == np.int64
    assert idx[0] == 0
    assert idx[-1] == 99
    assert 16 <= len(idx) <= 24
    assert len(np.unique(idx)) == len(idx)


def test_windows_2p5d_are_consecutive_group3():
    win = windows_2p5d_indices(40, n_windows=8, group=3)
    assert win.shape == (8, 3)
    assert win[0, 0] == 0 or win[0, 1] == 0
    assert win[-1].max() == 39
    for row in win:
        assert row[1] >= row[0]
        assert row[2] >= row[1]
        assert (row[2] - row[0]) <= 2


def test_pick_plane_prefers_fat_suppressed_axial():
    rows = [
        {"Anatomical_Plane": "Sagittal", "Fat_Suppression": 1, "Fluid_Sensitive": 1},
        {"Anatomical_Plane": "Axial", "Fat_Suppression": 0, "Fluid_Sensitive": 0, "SeriesInstanceUID": "ax-plain"},
        {"Anatomical_Plane": "Axial", "Fat_Suppression": 1, "Fluid_Sensitive": 1, "SeriesInstanceUID": "ax-fs"},
    ]
    picked = pick_plane_series(rows, "Axial")
    assert picked is not None
    assert picked["SeriesInstanceUID"] == "ax-fs"


def test_specialist_fill_only_pf_oa_and_synovitis():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    mat = specialist_fill_matrix(2, probs)
    assert mat.shape == (2, 12)
    pf = FINDINGS.index("PF OA")
    syn = FINDINGS.index("Synovitis")
    np.testing.assert_allclose(mat[:, pf], [0.9, 0.2])
    np.testing.assert_allclose(mat[:, syn], [0.1, 0.8])
    other = [i for i in range(12) if i not in (pf, syn)]
    np.testing.assert_allclose(mat[:, other], 0.5)


def test_gold_abort_threshold_still_point_nine_four():
    assert GOLD_AUC_ABORT_THRESHOLD == 0.94
    assert should_abort_gold_overfit(0.941) is True
    assert should_abort_gold_overfit(0.94) is False
    abort_if_gold_overfit(0.6252)


def test_gated_attention_pool_is_not_mean_gap():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    pool = GatedAttentionPool(dim=8, n_queries=2, hidden=4)
    H = torch.randn(3, 5, 8)
    logits = pool(H)
    assert tuple(logits.shape) == (3, 2)
    w = pool.attention_weights(H)
    assert tuple(w.shape) == (3, 5, 2)
    s = w.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    n_attn = sum(p.numel() for p in pool.parameters())
    assert n_attn > 8 * 2  # more than a GAP-linear 8→2 map


def test_kernel_b_source_is_specialist_not_gap_linear():
    src = kernel_source_text(KERNEL_B)
    assert kernel_has_t4_gpu_probe(src)
    assert kernel_discovers_competitions_mount(src)
    assert kernel_keeps_bakers_apostrophe(src)
    assert not kernel_reads_test_report_column(src)
    assert not kernel_loads_r336_as_scored_head(src)
    assert not kernel_is_frozen_gap_linear(src)
    assert not kernel_uses_mill_family_backbone(src)
    assert "GatedAttentionPool" in src
    assert "UNFREEZE_LAST = 1" in src
    assert "B_N_SLICES = 24" in src
    assert "B_IMG = 224" in src
    assert "B_CROP_MM = 0.0" in src
    assert "PF OA" in src and "Synovitis" in src
    assert "llm_holdout" in src.lower() or "llm-soft holdout" in src.lower()
    assert "GOLD_AUC_ABORT_THRESHOLD" in src
    assert "SLOTS_RECOVERED" not in src
    assert "RESEARCH_METHOD" in src
    assert "Ilse" in src or "gated" in src.lower()


def test_kernel_c_source_is_2p5d_attention_not_mill_backbone():
    src = kernel_source_text(KERNEL_C)
    assert kernel_has_t4_gpu_probe(src)
    assert kernel_discovers_competitions_mount(src)
    assert kernel_keeps_bakers_apostrophe(src)
    assert not kernel_reads_test_report_column(src)
    assert not kernel_loads_r336_as_scored_head(src)
    assert not kernel_is_frozen_gap_linear(src)
    assert not kernel_uses_mill_family_backbone(src)
    assert "GatedAttentionPool" in src
    assert "UNFREEZE_LAST = 1" in src
    assert "C_GROUP = 3" in src
    assert "C_IMG = 224" in src
    assert "C_CROP_MM = 0.0" in src
    assert 'C_BACKBONE = "resnet18"' in src or 'BACKBONE_NAME = "resnet18"' in src
    assert "WINDOWS_PER_PLANE" in src
    assert "coatnet" not in src.lower() or "FORBIDDEN" in src
    assert "AutoModel" not in src
    assert "conv_seg" not in src
    assert "llm_holdout" in src.lower() or "llm-soft holdout" in src.lower()
    assert "RESEARCH_METHOD" in src


def test_train_ids_contain_none_of_the_gold_58_for_bc_split():
    if not V4_CSV.is_file():
        pytest.skip(f"steven v4 CSV missing: {V4_CSV}")
    from rsna_knee.schema import ID_COLUMN
    from rsna_knee.v4_split import llm_soft_holdout_indices, make_steven_v4_split

    train = pd.read_csv(TRAIN_CSV, dtype={ID_COLUMN: str})
    v4 = pd.read_csv(V4_CSV, dtype={ID_COLUMN: str})
    split = make_steven_v4_split(train, v4)
    gold = set(pd.read_csv(GOLD_CSV, dtype={ID_COLUMN: str})[ID_COLUMN].astype(str))
    assert set(split.val_ids) == gold
    assert set(split.train_ids).isdisjoint(gold)
    fit, holdout = llm_soft_holdout_indices(split.n_train, frac=0.2, seed=2026)
    ho_ids = {split.train_ids[i] for i in holdout.tolist()}
    assert ho_ids.isdisjoint(gold)
    assert "GatedAttentionPool" in kernel_source_text(KERNEL_B)
    assert "GatedAttentionPool" in kernel_source_text(KERNEL_C)
