"""Window contract and Raptor head shapes. No DICOM and no OAI weights."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

from rsna_knee.raptor_model import (
    IMG,
    K_EVAL,
    K_TRAIN,
    N_SLICES,
    RaptorClassifier,
    epoch_seconds,
    eval_centers,
    imagenet_norm,
    sample_train_centers,
    windows_from_volume,
)


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(3, 8)
        self.num_features = 8

    def forward(self, x):
        return self.proj(self.pool(x).flatten(1))


def _volume():
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 256, size=(N_SLICES, IMG, IMG), dtype=np.uint8)
    mask = np.ones(N_SLICES, np.uint8)
    mask[:2] = 0
    mask[-2:] = 0
    return vol, mask


def test_eval_centers_cover_a_full_volume_and_train_draws_sixteen():
    _vol, mask = _volume()
    centers = eval_centers(mask, N_SLICES, K_EVAL)
    assert len(centers) == K_EVAL
    assert min(centers) >= 1
    assert max(centers) <= N_SLICES - 2
    rng = np.random.default_rng(2026)
    train = sample_train_centers(mask, N_SLICES, K_TRAIN, rng)
    assert len(train) == K_TRAIN


def test_windows_keep_384_and_centre_crop_320():
    vol, mask = _volume()
    centers = eval_centers(mask, N_SLICES, 4)
    full = windows_from_volume(vol, centers, IMG)
    assert full.shape == (4, 3, IMG, IMG)
    assert full.min() >= 0.0 and full.max() <= 1.0
    crop = windows_from_volume(vol, centers, 320)
    assert crop.shape == (4, 3, 320, 320)
    lo = (IMG - 320) // 2
    np.testing.assert_allclose(crop[0, 0], full[0, 0, lo : lo + 320, lo : lo + 320])


def test_raptor_head_returns_twelve_logits():
    model = RaptorClassifier(_Stub(), f_dim=8, n=12)
    x = torch.rand(2, K_TRAIN, 3, 8, 8)
    logits = model(imagenet_norm(x))
    assert logits.shape == (2, 12)


def test_epoch_projection_scales_with_the_train_count():
    assert epoch_seconds(8.0, 4, 400) == 800.0
