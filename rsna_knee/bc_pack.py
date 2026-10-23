"""B/C packing and heads: not mill 6-slot 336/130 mm, not frozen GAP-linear.

B: axial-dense 24-slice recess-inclusive FOV + gated-attention MIL (Ilse 2018)
   on PF OA + Synovitis. Last ResNet block unfrozen.
C: mill-*recipe* 2.5D GROUP=3 windows + per-finding gated attention on ImageNet
   ResNet18. Not CoAtNet, not mill DINOv2, not mill RadImageNet, not r336.
"""

from __future__ import annotations

import numpy as np

from rsna_knee.schema import FINDINGS
from rsna_knee.v4_backbone import (
    FORBIDDEN_SCORED_CHECKPOINTS,
    MILL_N_SLOT,
    MILL_SLOT_CROP_MM,
    MILL_SLOT_IMG,
    R336_SCORED_HEAD,
    is_mill_slot_recipe,
)

# --- B: PF OA / Synovitis specialist (axial-dense, not mill crop) ---
B_PLANES: tuple[str, ...] = ("Axial",)
B_N_SLICES = 24
B_IMG = 224
B_CROP_MM = 0.0
B_UNFREEZE_LAST = 1
B_HEAD = "gated_attention_mil"
B_TARGETS: tuple[str, ...] = ("PF OA", "Synovitis")
B_BACKBONE = "resnet18"
B_RESEARCH_METHOD = (
    "ABMIL gated attention (Ilse ICML 2018) + axial-dense 24-slice "
    "recess-inclusive FOV crop_mm=0 (MOAKS effusion-synovitis on axial; "
    "patellar cartilage en-face on axial). Prefer FS/fluid-sensitive axial. "
    "Last-block unfreeze. Not mill 6-slot 336/130, not frozen GAP-linear."
)

# --- C: mill recipe, non-mill backbone ---
C_PLANES: tuple[str, ...] = ("Axial", "Coronal", "Sagittal")
C_WINDOWS_PER_PLANE = 8
C_GROUP = 3
C_IMG = 224
C_CROP_MM = 0.0
C_UNFREEZE_LAST = 1
C_HEAD = "per_finding_gated_attention"
C_BACKBONE = "resnet18"
C_N_WINDOWS = len(C_PLANES) * C_WINDOWS_PER_PLANE
C_RESEARCH_METHOD = (
    "2.5D GROUP=3 consecutive-slice windows + per-finding ABMIL on ImageNet "
    "ResNet18 @224 crop_mm=0, last-block unfreeze. Not CoAtNet, not mill "
    "DINOv2, not mill RadImageNet ResNet-50, not r336_best.pt as scored head."
)

MILL_FAMILY_BACKBONE_NEEDLES: tuple[str, ...] = (
    "coatnet",
    "dinov2",
    "radimagenet",
    "r336_best.pt",
    "raptor_ft_coatnet",
    "convnextv2_base.fcmae",
)


def axial_dense_indices(n_files: int, n_slices: int = B_N_SLICES) -> np.ndarray:
    """Uniform axial sample spanning the full stack (recesses + trochlea)."""
    n_files = int(n_files)
    n_slices = int(n_slices)
    if n_files <= 0 or n_slices <= 0:
        return np.zeros((0,), dtype=np.int64)
    n = min(n_slices, n_files)
    return np.unique(np.linspace(0, n_files - 1, n).astype(np.int64))


def windows_2p5d_indices(
    n_files: int, n_windows: int = C_WINDOWS_PER_PLANE, group: int = C_GROUP
) -> np.ndarray:
    """(n_windows, group) consecutive-slice indices. Mill *recipe*, not mill crop."""
    n_files = int(n_files)
    n_windows = int(n_windows)
    group = int(group)
    if n_files <= 0 or n_windows <= 0 or group <= 0:
        return np.zeros((0, max(group, 0)), dtype=np.int64)
    half = group // 2
    centers = np.linspace(0, n_files - 1, n_windows)
    out = np.zeros((n_windows, group), dtype=np.int64)
    last = n_files - 1
    for i, c in enumerate(centers):
        ci = int(round(float(c)))
        for j, d in enumerate(range(-half, -half + group)):
            out[i, j] = int(np.clip(ci + d, 0, last))
    return out


def plane_series_score(row: dict) -> tuple[int, int, int]:
    """Prefer fat-suppressed / fluid-sensitive series (MOAKS PDFS surrogate)."""
    fs = int(row.get("Fat_Suppression") or 0)
    fl = int(row.get("Fluid_Sensitive") or 0)
    return (fs + fl, fl, fs)


def pick_plane_series(series_rows: list[dict], plane: str) -> dict | None:
    cands = [r for r in series_rows if str(r.get("Anatomical_Plane", "")) == plane]
    if not cands:
        return None
    cands = sorted(cands, key=plane_series_score, reverse=True)
    return cands[0]


def is_gap_linear_head(head_name: str, unfreeze_last: int) -> bool:
    frozen = int(unfreeze_last) == 0
    gap = str(head_name).lower() in {"linear", "gap_linear", "gap", "fc"}
    return frozen and gap


def kernel_is_frozen_gap_linear(source: str) -> bool:
    frozen = "UNFREEZE_LAST = 0" in source
    has_attn = "GatedAttentionPool" in source or "gated_attention" in source.lower()
    return frozen and not has_attn


def kernel_uses_mill_family_backbone(source: str) -> bool:
    """True when a mill-family backbone is constructed, not merely forbidden."""
    low = source.lower()
    if "timm.create_model" in low and "coatnet" in low:
        return True
    if "automodel.from_pretrained" in low.replace(" ", ""):
        return True
    if "r336_best.pt" in low and "FORBIDDEN_SCORED_CHECKPOINTS" not in source:
        return True
    if "radimagenet" in low and "FORBIDDEN" not in source:
        return True
    return False


def specialist_fill_matrix(
    n: int,
    specialist_probs: np.ndarray,
    *,
    findings: tuple[str, ...] = FINDINGS,
    targets: tuple[str, ...] = B_TARGETS,
) -> np.ndarray:
    """12-col matrix: specialist columns filled, others 0.5 (not a 12-head replace)."""
    n = int(n)
    out = np.full((n, len(findings)), 0.5, dtype=np.float64)
    if specialist_probs.size == 0:
        return out
    if specialist_probs.shape[0] != n:
        raise ValueError("specialist_probs rows must match n")
    if specialist_probs.shape[1] != len(targets):
        raise ValueError("specialist_probs cols must match targets")
    idx = {f: i for i, f in enumerate(findings)}
    for j, t in enumerate(targets):
        out[:, idx[t]] = np.asarray(specialist_probs[:, j], dtype=np.float64)
    return out


def assert_b_pack_config() -> None:
    if is_mill_slot_recipe(img=B_IMG, crop_mm=B_CROP_MM, n_slot=B_N_SLICES):
        raise RuntimeError("B packing collides with mill 6-slot 336/130 mm")
    if B_IMG == MILL_SLOT_IMG and abs(B_CROP_MM - MILL_SLOT_CROP_MM) < 1e-6:
        raise RuntimeError("B packing collides with mill 336/130 mm")
    if is_gap_linear_head(B_HEAD, B_UNFREEZE_LAST):
        raise RuntimeError("B must not be frozen GAP-linear")
    if B_UNFREEZE_LAST < 1:
        raise RuntimeError("B must unfreeze last block")
    if B_HEAD != "gated_attention_mil":
        raise RuntimeError("B head must be gated attention MIL")
    if B_PLANES != ("Axial",):
        raise RuntimeError("B must be axial-dense")
    if set(B_TARGETS) != {"PF OA", "Synovitis"}:
        raise RuntimeError("B targets must be PF OA and Synovitis")
    if B_CROP_MM != 0.0:
        raise RuntimeError("B FOV must be recess-inclusive (crop_mm=0)")
    if B_N_SLICES < 16:
        raise RuntimeError("B axial sampling is not dense")
    if R336_SCORED_HEAD not in FORBIDDEN_SCORED_CHECKPOINTS:
        raise RuntimeError("r336 must stay a forbidden scored head")


def assert_c_pack_config() -> None:
    if is_mill_slot_recipe(img=C_IMG, crop_mm=C_CROP_MM, n_slot=MILL_N_SLOT):
        raise RuntimeError("C packing collides with mill 6-slot 336/130 mm")
    if C_IMG == MILL_SLOT_IMG and abs(C_CROP_MM - MILL_SLOT_CROP_MM) < 1e-6:
        raise RuntimeError("C packing collides with mill 336/130 mm")
    if is_gap_linear_head(C_HEAD, C_UNFREEZE_LAST):
        raise RuntimeError("C must not be frozen GAP-linear")
    if C_UNFREEZE_LAST < 1:
        raise RuntimeError("C must unfreeze last block")
    if C_GROUP != 3:
        raise RuntimeError("C must use mill-style 2.5D GROUP=3")
    if C_BACKBONE != "resnet18":
        raise RuntimeError("C backbone must be non-mill ImageNet resnet18")
    if C_HEAD != "per_finding_gated_attention":
        raise RuntimeError("C head must be per-finding attention")
    if C_CROP_MM != 0.0:
        raise RuntimeError("C must not use mill 130 mm crop")


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if nn is not None:

    class GatedAttentionPool(nn.Module):
        """Ilse et al. ICML 2018 gated-attention MIL pool. Not GAP-linear.

        H: (B, T, D) slice/window embeddings → logits (B, n_queries).
        """

        def __init__(self, dim: int, n_queries: int, hidden: int = 128):
            super().__init__()
            self.V = nn.Linear(dim, hidden)
            self.U = nn.Linear(dim, hidden)
            self.w = nn.Linear(hidden, n_queries)
            self.cls = nn.Linear(dim, 1)

        def attention_weights(self, H):
            a = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))
            return torch.softmax(self.w(a), dim=1)

        def forward(self, H):
            w = self.attention_weights(H)
            z = torch.einsum("btq,btd->bqd", w, H)
            return self.cls(z).squeeze(-1)

else:  # pragma: no cover

    class GatedAttentionPool:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for GatedAttentionPool")
