"""Load a Keras RadImageNet DenseNet121-notop file into torchvision.

The public file is ``sjudal/radimagenet-densenet121``
``RadImageNet-DenseNet121_notop.h5``. It is a 3-channel medical conv, not mill
DINO, CoAtNet, or RadImageNet-ResNet-50. The classifier is a new 12-way head
and is not read from the file.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torchvision.models import densenet121

BLOCK_LAYERS = (6, 12, 24, 16)


def _dataset(group: h5py.Group, kind: str) -> np.ndarray:
    hits: list[np.ndarray] = []

    def walk(node: h5py.Group) -> None:
        for key, value in node.items():
            if isinstance(value, h5py.Dataset) and key.startswith(kind):
                hits.append(np.asarray(value))
            elif isinstance(value, h5py.Group):
                walk(value)

    walk(group)
    if len(hits) != 1:
        raise ValueError(f"{group.name} has {len(hits)} datasets for {kind}")
    return hits[0]


def _conv(kernel_hwio: np.ndarray) -> torch.Tensor:
    # Keras HWIO -> torch OIHW.
    array = np.transpose(kernel_hwio, (3, 2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(array))


def _bn(group: h5py.Group) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.from_numpy(np.ascontiguousarray(_dataset(group, "gamma"))),
        "bias": torch.from_numpy(np.ascontiguousarray(_dataset(group, "beta"))),
        "running_mean": torch.from_numpy(np.ascontiguousarray(_dataset(group, "moving_mean"))),
        "running_var": torch.from_numpy(np.ascontiguousarray(_dataset(group, "moving_variance"))),
    }


def keras_feature_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Map the notop Keras tensors onto torchvision ``features.*`` keys."""
    weights = h5py.File(path, "r")["model_weights"]
    state: dict[str, torch.Tensor] = {}
    state["features.conv0.weight"] = _conv(_dataset(weights["conv1"]["conv"], "kernel"))
    for name, tensor in _bn(weights["conv1"]["bn"]).items():
        state[f"features.norm0.{name}"] = tensor
    for name, tensor in _bn(weights["bn"]).items():
        state[f"features.norm5.{name}"] = tensor

    for block_i, n_layers in enumerate(BLOCK_LAYERS, start=1):
        keras_conv = block_i + 1
        for layer_j in range(1, n_layers + 1):
            prefix = f"features.denseblock{block_i}.denselayer{layer_j}"
            base = f"conv{keras_conv}_block{layer_j}"
            for name, tensor in _bn(weights[f"{base}_0_bn"]).items():
                state[f"{prefix}.norm1.{name}"] = tensor
            state[f"{prefix}.conv1.weight"] = _conv(_dataset(weights[f"{base}_1_conv"], "kernel"))
            for name, tensor in _bn(weights[f"{base}_1_bn"]).items():
                state[f"{prefix}.norm2.{name}"] = tensor
            state[f"{prefix}.conv2.weight"] = _conv(_dataset(weights[f"{base}_2_conv"], "kernel"))
        if block_i < 4:
            trans = f"features.transition{block_i}"
            pool = f"pool{block_i + 1}"
            for name, tensor in _bn(weights[f"{pool}_bn"]).items():
                state[f"{trans}.norm.{name}"] = tensor
            state[f"{trans}.conv.weight"] = _conv(_dataset(weights[f"{pool}_conv"], "kernel"))
    return state


def load_radimagenet_densenet121(path: str | Path) -> tuple[nn.Module, dict[str, object]]:
    """Frozen features plus a new 12-logit head. Gold labels are not touched."""
    model = densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 12)
    incoming = keras_feature_state(path)
    current = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    shape_errors: list[str] = []
    for key, tensor in incoming.items():
        if key not in current:
            continue
        if tuple(current[key].shape) != tuple(tensor.shape):
            shape_errors.append(f"{key} keras {tuple(tensor.shape)} torch {tuple(current[key].shape)}")
            continue
        matched[key] = tensor
    if shape_errors:
        raise ValueError("shape mismatch: " + "; ".join(shape_errors[:8]))
    missing, unexpected = model.load_state_dict(matched, strict=False)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    report = {
        "n_loaded": len(matched),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "classifier_out": int(model.classifier.out_features),
        "features_frozen": all(not p.requires_grad for p in model.features.parameters()),
        "head_trainable": any(p.requires_grad for p in model.classifier.parameters()),
    }
    return model, report
