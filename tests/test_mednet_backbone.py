"""Drive shipped MedicalNet encoder-only config: not mill slots, not a fake 12-head."""

from pathlib import Path

from rsna_knee.guards import kernel_reads_test_report_column, kernel_source_text
from rsna_knee.mednet_backbone import (
    BACKBONE_CROP_MM,
    BACKBONE_DEPTH,
    BACKBONE_HW,
    BACKBONE_NAME,
    FEATURE_DIM,
    FORBIDDEN_SCORED_CHECKPOINTS,
    MILL_N_SLOT,
    MILL_SLOT_CROP_MM,
    MILL_SLOT_IMG,
    UNFREEZE_LAST,
    WEIGHT_FILE,
    assert_mednet_config,
    encoder_only_weight_file,
    is_mill_slot_recipe,
)
from rsna_knee.v4_backbone import kernel_loads_r336_as_scored_head

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_mednet" / "main.py"


def test_mednet_config_is_encoder_only_not_mill_slot():
    assert UNFREEZE_LAST == 0
    assert BACKBONE_NAME == "medicalnet_resnet50_3d"
    assert BACKBONE_HW == 112
    assert BACKBONE_DEPTH == 32
    assert BACKBONE_CROP_MM == 0.0
    assert FEATURE_DIM == 2048
    assert encoder_only_weight_file(WEIGHT_FILE)
    assert not encoder_only_weight_file("r336_best.pt")
    assert is_mill_slot_recipe(img=MILL_SLOT_IMG, crop_mm=MILL_SLOT_CROP_MM, n_slot=MILL_N_SLOT)
    assert not is_mill_slot_recipe(img=BACKBONE_HW, crop_mm=BACKBONE_CROP_MM, n_slot=1)
    assert_mednet_config()
    assert "r336_best.pt" in FORBIDDEN_SCORED_CHECKPOINTS


def test_mednet_kernel_source_forbids_mill_recipe_fake_head_and_report():
    src = kernel_source_text(KERNEL)
    assert "UNFREEZE_LAST = 0" in src
    assert "resnet_50_23dataset.pth" in src
    assert "conv_seg" in src
    assert "layer4" in src
    assert "adaptive_avg_pool3d" in src.lower()
    assert "nn.Linear" in src
    assert "FORBIDDEN_SCORED_CHECKPOINTS" in src
    assert not kernel_loads_r336_as_scored_head(src)
    assert "SLOTS_RECOVERED" not in src
    assert "CROP_MM = 130" not in src or "MILL_SLOT_CROP_MM" in src
    assert not kernel_reads_test_report_column(src)
    assert "Baker's" in src
    assert "gold=58" in src or "gold 58" in src.lower() or "val-only" in src
    assert "llm-soft holdout" in src or "llm_holdout" in src
    assert "/kaggle/input/competitions/rsna-knee-abnormality-detection" in src
    assert "nvidia-smi" in src
    # MedicalNet conv_seg is not the scored 12-finding head
    assert "num_seg_classes" in src or "conv_seg" in src
    assert "FINDINGS" in src
    linear_12 = "nn.Linear(FEATURE_DIM, len(FINDINGS))" in src or "nn.Linear(2048, len(FINDINGS))" in src or "nn.Linear(dim, len(FINDINGS))" in src
    assert linear_12
