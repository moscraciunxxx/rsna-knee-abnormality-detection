"""Drive the shipped 2.5D slice-as-channel encoder (ResNet34 contract)."""

import numpy as np

from rsna_knee.encoder import N_SLICE_CHANNELS, encode_study, resample_slices_as_channels
from rsna_knee.schema import FINDINGS


def test_resample_uses_twenty_four_depth_channels_like_public_resnet34():
    volume = np.linspace(0, 1, 40 * 8 * 8, dtype=np.float32).reshape(40, 8, 8)
    stacked = resample_slices_as_channels(volume)
    assert stacked.shape[0] == N_SLICE_CHANNELS == 24
    assert stacked.shape[1:] == (8, 8)


def test_encode_study_returns_twelve_probabilities_in_unit_interval():
    volumes = [
        np.random.default_rng(0).random((12, 16, 16)).astype(np.float32),
        np.random.default_rng(1).random((9, 16, 16)).astype(np.float32),
    ]
    probs = encode_study(volumes)
    assert probs.shape == (len(FINDINGS),)
    assert np.isfinite(probs).all()
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
