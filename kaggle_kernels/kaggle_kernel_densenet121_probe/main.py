"""8-study RadImageNet-DenseNet121 timing probe. Do not Submit.

Gold-58 stay out of the sample. Abort when the projected 4407-study encode
is above 2.8 h. A new 12-way head is created and not trained here.
"""
from __future__ import annotations

ID_COLUMN = "StudyInstanceUID"
FINDINGS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)


def expert_gold_ids(train):
    frame = train.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    mask = frame[list(FINDINGS)].notna().all(axis=1)
    ids = tuple(frame.loc[mask, ID_COLUMN].tolist())
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate gold StudyInstanceUID")
    return ids


"""Load a Keras RadImageNet DenseNet121-notop file into torchvision.

The public file is ``sjudal/radimagenet-densenet121``
``RadImageNet-DenseNet121_notop.h5``. It is a 3-channel medical conv, not mill
DINO, CoAtNet, or RadImageNet-ResNet-50. The classifier is a new 12-way head
and is not read from the file.
"""


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

"""8-study timing math for a frozen RadImageNet DenseNet121 encode.

The full 4349-study train is not started here. Gold expert rows are excluded
from the probe sample so they cannot enter a loss.
"""


import numpy as np
import pandas as pd


PROBE_STUDIES = 8
IMAGES_PER_STUDY = 18
N_ENCODE_STUDIES = 4407
ENCODE_ABORT_S = 2.8 * 3600.0


def probe_study_ids(
    train: pd.DataFrame,
    *,
    n: int = PROBE_STUDIES,
    seed: int = 2026,
) -> tuple[str, ...]:
    """Non-gold study IDs. The returned set is disjoint from the expert 58."""
    gold = set(expert_gold_ids(train))
    frame = train.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    pool = [uid for uid in frame[ID_COLUMN].tolist() if uid not in gold]
    if len(pool) < int(n):
        raise ValueError(f"need {n} non-gold studies, have {len(pool)}")
    rng = np.random.default_rng(int(seed))
    chosen_at = np.sort(rng.choice(len(pool), size=int(n), replace=False))
    chosen = tuple(pool[int(i)] for i in chosen_at.tolist())
    if set(chosen) & gold:
        raise ValueError("gold UID leaked into the DenseNet probe sample")
    return chosen


def projected_encode_seconds(
    elapsed_s: float,
    n_images: int,
    *,
    n_studies: int = N_ENCODE_STUDIES,
    images_per_study: int = IMAGES_PER_STUDY,
) -> float:
    """Scale a measured image batch onto the full encode set."""
    elapsed = float(elapsed_s)
    images = int(n_images)
    if elapsed <= 0.0 or images <= 0:
        raise ValueError("elapsed time and image count must be positive")
    return elapsed * (int(n_studies) * int(images_per_study)) / images


def encode_probe_aborts(
    elapsed_s: float,
    n_images: int,
    *,
    limit_s: float = ENCODE_ABORT_S,
) -> bool:
    """True when the projected full encode is strictly above 2.8 h."""
    return projected_encode_seconds(elapsed_s, n_images) > float(limit_s)


# --- 8-study timing probe. Do not competitions-submit. Keep 55779066. ---
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
print("nvidia-smi:", subprocess.getoutput("nvidia-smi -L"), flush=True)
print(
    "[dn121-probe] 8-study frozen RadImageNet-DenseNet121 timing probe. "
    "New 12-way head. Gold-58 out of the loss. DO NOT SUBMIT. keep 55779066.",
    flush=True,
)

HARD_LIMIT_S = 1500


def _watchdog() -> None:
    time.sleep(HARD_LIMIT_S)
    print(f"[dn121-probe] WATCHDOG {HARD_LIMIT_S}s", flush=True)
    os._exit(99)


threading.Thread(target=_watchdog, daemon=True, name="dn121-watchdog").start()

FINDINGS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)


def _comp_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for path in candidates:
        if (path / "train.csv").is_file() and (path / "train_series").is_dir():
            return path
    base = Path("/kaggle/input")
    for path in base.rglob("train.csv"):
        if (path.parent / "train_series").is_dir():
            return path.parent
    raise FileNotFoundError("competition train_series not mounted")


def _find_h5() -> Path:
    base = Path("/kaggle/input")
    skip = {"train_series", "test_series", "train_images", "test_images"}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [name for name in dirnames if name not in skip and not name.startswith("1.2.")]
        if "RadImageNet-DenseNet121_notop.h5" in filenames:
            return Path(dirpath) / "RadImageNet-DenseNet121_notop.h5"
    raise FileNotFoundError("RadImageNet-DenseNet121_notop.h5")


def _study_batch(root: Path, uid: str, k: int = IMAGES_PER_STUDY) -> torch.Tensor | None:
    folder = root / "train_series" / uid
    files = sorted(path for path in folder.rglob("*") if path.suffix.lower() == ".dcm")
    if not files:
        return None
    if len(files) > k:
        pick = np.linspace(0, len(files) - 1, k).astype(int)
        files = [files[int(i)] for i in pick]
    images = []
    for path in files:
        array = np.asarray(pydicom.dcmread(str(path), force=True).pixel_array, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        low, high = np.percentile(array, [1, 99])
        scale = max(float(high - low), 1e-6)
        array = np.clip((array - float(low)) / scale, 0.0, 1.0).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        images.append(tensor.repeat(1, 3, 1, 1))
    if not images:
        return None
    return torch.cat(images, dim=0)


def main() -> None:
    comp = _comp_root()
    train = pd.read_csv(comp / "train.csv", usecols=["StudyInstanceUID", *FINDINGS])
    ids = probe_study_ids(train, n=PROBE_STUDIES, seed=2026)
    gold = set(expert_gold_ids(train))
    print(f"[dn121-probe] n={len(ids)} gold_in_probe={len(set(ids) & gold)}", flush=True)
    if set(ids) & gold:
        raise SystemExit("[dn121-probe] gold leaked into the probe sample")
    model, report = load_radimagenet_densenet121(_find_h5())
    print(
        f"[dn121-probe] loaded n={report['n_loaded']} unexpected={len(report['unexpected'])} "
        f"frozen={report['features_frozen']} head={report['classifier_out']}",
        flush=True,
    )
    if report["unexpected"] or not report["features_frozen"] or report["classifier_out"] != 12:
        raise SystemExit("[dn121-probe] weight load failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    started = time.perf_counter()
    n_images = 0
    for uid in ids:
        batch = _study_batch(comp, uid)
        if batch is None:
            print(f"[dn121-probe] no slices {uid}", flush=True)
            continue
        with torch.no_grad():
            model.features(batch.to(device).float())
        n_images += int(batch.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if n_images <= 0:
        raise SystemExit("[dn121-probe] encoded 0 images")
    projected = projected_encode_seconds(elapsed, n_images)
    print(
        f"[dn121-probe] images={n_images} elapsed_s={elapsed:.3f} "
        f"projected_s={projected:.1f} abort_limit_s={ENCODE_ABORT_S:.1f}",
        flush=True,
    )
    sub = Path("/kaggle/working/submission.csv")
    if sub.is_file():
        sub.unlink()
    if encode_probe_aborts(elapsed, n_images):
        print("[dn121-probe] ABORT projected encode > 2.8h. No full train. DO NOT SUBMIT.", flush=True)
        raise SystemExit(2)
    print(
        "[dn121-probe] PROBE_OK projected encode <= 2.8h. "
        "gold_macro=not_fit gold_in_loss=0. Full train not started. DO NOT SUBMIT. keep 55779066.",
        flush=True,
    )


if __name__ == "__main__":
    main()
