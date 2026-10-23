"""Frozen backbone for the steven-v4 head — not mill’s 6-slot 336/130 mm family.

r336 (`r336_best.pt`) is mill-slot ViT-B and already scored 0.935 (55856648).
The new head uses a 3-plane 224 px stack with no millimetre crop and a frozen
ImageNet/MedicalNet encoder. `unfreeze_last` stays 0.
"""

from __future__ import annotations

BACKBONE_NAME = "resnet18"
BACKBONE_IMG = 224
BACKBONE_CROP_MM = 0.0  # pixel centre crop; never mill 130 mm
PLANES = ("Axial", "Coronal", "Sagittal")
SLICES_PER_PLANE = 8
UNFREEZE_LAST = 0
SCORED_HEAD_WEIGHTS = "v4_linear_head.pt"

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

R336_SCORED_HEAD = "r336_best.pt"


def is_mill_slot_recipe(
    *,
    img: int,
    crop_mm: float,
    n_slot: int,
) -> bool:
    return (
        int(img) == MILL_SLOT_IMG
        and abs(float(crop_mm) - MILL_SLOT_CROP_MM) < 1e-6
        and int(n_slot) == MILL_N_SLOT
    )


def kernel_loads_r336_as_scored_head(source: str) -> bool:
    """True when source torch-loads r336_best.pt outside a forbidden-list."""
    if "r336_best.pt" not in source:
        return False
    if "FORBIDDEN_SCORED_CHECKPOINTS" in source:
        return False
    return "torch.load" in source and "r336_best.pt" in source


def expand_conv1_rgb_to_n(weight, n_in: int = 24):
    """Repeat ImageNet conv1 (64, 3, 7, 7) to n_in input channels, scale by 3/n_in."""
    if tuple(weight.shape[1:]) != (3, 7, 7) and weight.shape[1] != 3:
        raise ValueError(f"expected RGB conv1, got {tuple(weight.shape)}")
    reps = (int(n_in) + 2) // 3
    expanded = weight.repeat(1, reps, 1, 1)[:, : int(n_in)]
    return expanded * (3.0 / float(n_in))


def assert_v4_backbone_config() -> None:
    if UNFREEZE_LAST != 0:
        raise RuntimeError("unfreeze_last must stay 0")
    if is_mill_slot_recipe(
        img=BACKBONE_IMG, crop_mm=BACKBONE_CROP_MM, n_slot=len(PLANES)
    ):
        raise RuntimeError("v4 backbone must not be mill 6-slot 336/130 mm")
    if SCORED_HEAD_WEIGHTS in FORBIDDEN_SCORED_CHECKPOINTS:
        raise RuntimeError("scored head collides with a forbidden mill checkpoint")
    if BACKBONE_IMG == MILL_SLOT_IMG and abs(BACKBONE_CROP_MM - MILL_SLOT_CROP_MM) < 1e-6:
        raise RuntimeError("v4 packing collides with mill 336/130 mm")
