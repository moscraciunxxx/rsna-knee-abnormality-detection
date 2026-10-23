"""Frozen MedicalNet 3D ResNet-50 encoder — not a 12-head, not mill 6-slot.

The public `.pth` is Med3D/MedicalNet (23-dataset) with a segmentation `conv_seg`.
We freeze conv1–layer4 only and train a NEW linear head. Gold 58 stays val-only.
"""

from __future__ import annotations

BACKBONE_NAME = "medicalnet_resnet50_3d"
BACKBONE_DEPTH = 32
BACKBONE_HW = 112
BACKBONE_CROP_MM = 0.0
UNFREEZE_LAST = 0
FEATURE_DIM = 2048  # ResNet-50 Bottleneck expansion 4 × 512
SCORED_HEAD_WEIGHTS = "mednet_linear_head.pt"
WEIGHT_FILE = "resnet_50_23dataset.pth"
WEIGHT_DATASET = "royceplayz/rsna-medicalnet-resnet50-weights"

MILL_SLOT_IMG = 336
MILL_SLOT_CROP_MM = 130.0
MILL_N_SLOT = 6
FORBIDDEN_SCORED_CHECKPOINTS = (
    "r336_best.pt",
    "r224_best.pt",
    "raptor_ft_coatnet_v4_full.pt",
    "raptor_ft_coatnet_v5_full_swa.pt",
    "focal_fold",
)


def is_mill_slot_recipe(*, img: int, crop_mm: float, n_slot: int) -> bool:
    return (
        int(img) == MILL_SLOT_IMG
        and abs(float(crop_mm) - MILL_SLOT_CROP_MM) < 1e-6
        and int(n_slot) == MILL_N_SLOT
    )


def encoder_only_weight_file(name: str) -> bool:
    """True for MedicalNet .pth names that are not RSNA 12-logit heads."""
    n = name.lower()
    if "r336_best" in n or "12head" in n or "slot" in n:
        return False
    return n.endswith(".pth") and "resnet_50" in n


def assert_mednet_config() -> None:
    if UNFREEZE_LAST != 0:
        raise RuntimeError("unfreeze_last must stay 0")
    if is_mill_slot_recipe(img=BACKBONE_HW, crop_mm=BACKBONE_CROP_MM, n_slot=1):
        raise RuntimeError("mednet packing collides with mill 6-slot 336/130 mm")
    if not encoder_only_weight_file(WEIGHT_FILE):
        raise RuntimeError("refusing to treat a 12-head checkpoint as MedicalNet encoder")
    if FEATURE_DIM != 2048:
        raise RuntimeError("MedicalNet ResNet-50 encoder GAP dim must stay 2048")
