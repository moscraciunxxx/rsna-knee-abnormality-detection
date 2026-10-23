"""Drive shipped v4 backbone config: not mill 6-slot 336/130 mm, not r336 scored head."""

from pathlib import Path

from rsna_knee.guards import kernel_reads_test_report_column, kernel_source_text
from rsna_knee.v4_backbone import (
    BACKBONE_CROP_MM,
    BACKBONE_IMG,
    BACKBONE_NAME,
    FORBIDDEN_SCORED_CHECKPOINTS,
    MILL_N_SLOT,
    MILL_SLOT_CROP_MM,
    MILL_SLOT_IMG,
    PLANES,
    R336_SCORED_HEAD,
    SLICES_PER_PLANE,
    UNFREEZE_LAST,
    assert_v4_backbone_config,
    expand_conv1_rgb_to_n,
    is_mill_slot_recipe,
    kernel_loads_r336_as_scored_head,
)

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle_kernel_v4head" / "main.py"


def test_expand_conv1_rgb_to_24_keeps_out_channels():
    import numpy as np

    rng = np.random.default_rng(0)
    w = rng.normal(size=(64, 3, 7, 7)).astype(np.float32)
    import torch

    t = torch.from_numpy(w)
    out = expand_conv1_rgb_to_n(t, 24)
    assert tuple(out.shape) == (64, 24, 7, 7)
    np.testing.assert_allclose(
        out[:, :3].numpy() * 8.0,
        w,
        atol=1e-5,
    )


def test_v4_config_is_not_mill_six_slot_336_130():
    assert UNFREEZE_LAST == 0
    assert BACKBONE_NAME == "resnet18"
    assert BACKBONE_IMG == 224
    assert BACKBONE_CROP_MM == 0.0
    assert SLICES_PER_PLANE == 8
    assert PLANES == ("Axial", "Coronal", "Sagittal")
    assert is_mill_slot_recipe(img=MILL_SLOT_IMG, crop_mm=MILL_SLOT_CROP_MM, n_slot=MILL_N_SLOT)
    assert not is_mill_slot_recipe(
        img=BACKBONE_IMG, crop_mm=BACKBONE_CROP_MM, n_slot=len(PLANES)
    )
    assert_v4_backbone_config()
    assert R336_SCORED_HEAD in FORBIDDEN_SCORED_CHECKPOINTS


def test_v4_kernel_source_forbids_mill_recipe_and_report():
    src = kernel_source_text(KERNEL)
    assert "BACKBONE_IMG = 224" in src
    assert "UNFREEZE_LAST = 0" in src
    assert "r336_best.pt" in src
    assert "FORBIDDEN_SCORED_CHECKPOINTS" in src
    assert not kernel_loads_r336_as_scored_head(src)
    assert "CROP_MM = 130" not in src or "MILL_SLOT_CROP_MM" in src
    assert "SLOTS_RECOVERED" not in src
    assert not kernel_reads_test_report_column(src)
    assert "steven" in src.lower() or "v4" in src.lower()
    assert "Baker's" in src
    assert "unfreeze_last>0" not in src.replace(" ", "")
    assert "UNFREEZE_LAST = 0" in src
    assert "loaded 3-ch" in src
    assert src.index("model.load_state_dict") < src.index("nn.Conv2d(24, 64")
    assert "llm-soft holdout" in src
    assert "llm_holdout_macro" in src
