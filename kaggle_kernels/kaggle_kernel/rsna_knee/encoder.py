"""2.5D slice-as-channel encoder (adapted from the public ResNet34 baseline).

Public method
-------------
Kaggle notebook `[Baseline] RSNA Knee MRI — 2.5D ResNet34`
(https://www.kaggle.com/code/barun2104/baseline-rsna-knee-mri-2-5d-resnet34):

- Concatenate a study's series along depth.
- Resample to a fixed **24** slices.
- Treat those 24 slices as channels of a 2D CNN (ResNet-34 `conv1` 3→24,
  ImageNet weights repeated and scaled by sqrt(3/24)).
- Linear head with **12** logits, sigmoid at export, scored with macro ROC AUC.

MRNet (Bien et al., PLOS Medicine 2018) is the prior art: slice-wise CNN plus
pooling across a series, then plane fusion. Astuto et al. (Radiology: AI 2021)
motivates keeping per-finding heads (ACL / meniscus / OA are not one bit).

This module ships the **same 24-channel / 12-head contract** so a GPU ResNet-34
can drop in. The default path is a numpy 2.5D encoder (per-channel moments +
fixed 12-way affine + sigmoid) so smoke inference runs without GPU, torch, or
the full DICOM dump. Optional DINOv3 slice embeddings
(`tonylica/rsna-knee-bend-dinov3-0917-repro-assets`) and LLM report soft labels
(`pilkwang/rsna-knee-llm-labels`) are train-time only and are not read here.
"""

from __future__ import annotations

import numpy as np

from rsna_knee.schema import N_FINDINGS

# Public 2.5D ResNet34 baseline: 24 spatial-depth channels.
N_SLICE_CHANNELS = 24
_HEAD_SEED = 20260825


def _as_volume(array: np.ndarray) -> np.ndarray:
    vol = np.asarray(array, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim != 3:
        raise ValueError(f"expected (S,H,W) or (H,W), got {vol.shape}")
    return vol


def resample_slices_as_channels(
    volume: np.ndarray, n_channels: int = N_SLICE_CHANNELS
) -> np.ndarray:
    """Uniformly sample `n_channels` slices — the 2.5D depth-as-channel stack."""
    vol = _as_volume(volume)
    n_slices = vol.shape[0]
    if n_slices < 1:
        raise ValueError("volume has no slices")
    index = np.linspace(0, n_slices - 1, n_channels).astype(int)
    stacked = vol[index]
    peak = float(np.max(np.abs(stacked)))
    if peak > 1.0:
        stacked = stacked / peak
    return stacked.astype(np.float32, copy=False)


def _channel_features(stacked: np.ndarray) -> np.ndarray:
    flat = stacked.reshape(stacked.shape[0], -1)
    return np.concatenate([flat.mean(axis=1), flat.std(axis=1)], axis=0)


def _heads(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_HEAD_SEED)
    weight = rng.normal(0.0, 0.35, size=(N_FINDINGS, n_features))
    bias = rng.normal(0.0, 0.05, size=(N_FINDINGS,))
    return weight, bias


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def encode_volume(volume: np.ndarray) -> np.ndarray:
    stacked = resample_slices_as_channels(volume)
    feats = _channel_features(stacked)
    weight, bias = _heads(feats.shape[0])
    return _sigmoid(weight @ feats + bias).astype(np.float64)


def encode_study(volumes) -> np.ndarray:
    """Study-level 2.5D encode: concat series on depth, then 24-channel heads."""
    series = [_as_volume(v) for v in list(volumes) if np.asarray(v).size]
    if not series:
        series = [np.zeros((1, 8, 8), dtype=np.float32)]
    combined = np.concatenate(series, axis=0)
    return encode_volume(combined)
