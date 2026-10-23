"""Raptor MIL head and 3-slice windows for the public CoAtNet recipe.

The head matches RaptorClassifier in the public 0.949 inference notebook.
Sixteen train windows match the ``t16`` token in that checkpoint's filename.
Eval centres match ``_eval_centers`` in the same notebook.

This module does not read OAI images and does not load an OAI-trained checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SLOT_SIZES: tuple[int, ...] = (26, 22, 18, 12, 18)
N_SLICES = sum(SLOT_SIZES)
IMG = 384
K_TRAIN = 16
K_EVAL = N_SLICES - 2  # 94, one window on every valid centre of a full volume
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RaptorClassifier(nn.Module):
    """Per-finding attention over K three-slice windows."""

    def __init__(self, backbone: nn.Module, f_dim: int, n: int = 12, drop: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(
            nn.Linear(f_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, n),
        )
        self.clsW = nn.Parameter(torch.zeros(n, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch, _k = x.shape[:2]
        flat = x.flatten(0, 1)
        parts = [self.backbone(chunk) for chunk in flat.split(4)]
        feats = torch.cat(parts, dim=0)
        return feats.view(batch, _k, -1)

    def head(self, feats: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(feats)
        attn = torch.softmax(self.att(hidden), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attn, hidden)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def valid_centers(mask: np.ndarray, depth: int) -> list[int]:
    """Centres whose three-slice window stays inside the valid span."""
    valid = np.where(np.asarray(mask) > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    centers = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not centers:
        centers = [max(1, min((lo + hi) // 2, depth - 2))]
    return centers


def eval_centers(mask: np.ndarray, depth: int, k: int) -> list[int]:
    """Evenly spaced centres. Matches the public notebook's ``_eval_centers``."""
    centers = valid_centers(mask, depth)
    idx = np.linspace(0, len(centers) - 1, k).round().astype(int)
    return [centers[int(i)] for i in idx]


def sample_train_centers(mask: np.ndarray, depth: int, k: int, rng: np.random.Generator) -> list[int]:
    """k centres drawn from the valid span. Repeats only when the span is shorter than k."""
    centers = valid_centers(mask, depth)
    pick = rng.choice(len(centers), size=k, replace=len(centers) < k)
    return [centers[int(i)] for i in pick]


def windows_from_volume(
    vol: np.ndarray,
    centers: list[int],
    res: int,
) -> np.ndarray:
    """Stack 3-slice windows. ``vol`` is uint8 (D, IMG, IMG). Values are in [0, 1].

    When ``res`` is smaller than the volume, take the centre crop. That is the
    public inference path (384 train, 320 score). Training uses ``res == IMG``.
    """
    depth = int(vol.shape[0])
    if vol.shape[-2:] != (IMG, IMG):
        raise ValueError(f"volume spatial size {vol.shape[-2:]} is not {IMG}")
    if res > IMG:
        raise ValueError("window res cannot exceed the volume")
    lo = (IMG - res) // 2
    out = np.empty((len(centers), 3, res, res), np.float32)
    for j, center in enumerate(centers):
        c = max(1, min(int(center), depth - 2))
        tri = vol[c - 1 : c + 2].astype(np.float32) / 255.0
        out[j] = tri[:, lo : lo + res, lo : lo + res]
    return out


def imagenet_norm(windows: torch.Tensor) -> torch.Tensor:
    """``windows`` is (K, 3, H, W) or (B, K, 3, H, W) in [0, 1]."""
    mean = windows.new_tensor(IMAGENET_MEAN).view(*([1] * (windows.ndim - 3)), 3, 1, 1)
    std = windows.new_tensor(IMAGENET_STD).view(*([1] * (windows.ndim - 3)), 3, 1, 1)
    return (windows - mean) / std


def epoch_seconds(sample_seconds: float, n_seen: int, n_train: int) -> float:
    """Project a full epoch from a timed prefix. ``n_seen`` must be positive."""
    if n_seen < 1:
        raise ValueError("need at least one timed study")
    if n_train < 1:
        raise ValueError("empty train split")
    return float(sample_seconds) * (float(n_train) / float(n_seen))
