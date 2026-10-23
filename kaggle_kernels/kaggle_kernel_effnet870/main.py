#!/usr/bin/env python3
"""Private 870 encode of dreaddevelopment/rsna-knee-model EfficientNet-B0.

Not mill. Not a Submit. Geometry is reconstructed from the checkpoint fields
(nser=4, nslc=6, sz=224, 1-channel stem). No author inference script is
attached to that dataset. A loss on the 870 does not kill the checkpoint.
A gain above 0.01 on at most two findings can be joined locally.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

print(
    "[effnet870] PRIVATE holdout. DO NOT SUBMIT. "
    "Reconstructed 4x6 grayscale geometry. Loss is inconclusive.",
    flush=True,
)

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
HEADER = ["StudyInstanceUID", *TARGETS]
HOLD_SEED = 2026
HOLD_FRAC = 0.2
NSER = 4
NSLC = 6
IMG = 224


def _find_file(name: str):
    root = "/kaggle/input"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in {"train_series", "test_series", "train_images", "test_images"}
            and not d.startswith("1.2.")
        ]
        if name in filenames:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(name)


def _comp_root():
    cands = [
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
        "/kaggle/input/rsna-knee-abnormality-detection",
    ]
    for path in cands:
        if os.path.isfile(os.path.join(path, "train.csv")):
            return path
    raise FileNotFoundError("competition train.csv not mounted")


def holdout_ids(train: pd.DataFrame, v4: pd.DataFrame) -> list[str]:
    gold_mask = train[TARGETS].notna().all(axis=1)
    gold = set(train.loc[gold_mask, "StudyInstanceUID"].astype(str))
    if len(gold) != 58:
        raise RuntimeError(f"gold count {len(gold)} != 58")
    labels = v4.drop_duplicates("StudyInstanceUID").copy()
    labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)
    train_ids = [
        uid for uid in labels["StudyInstanceUID"].tolist() if uid not in gold
    ]
    if set(train_ids) & gold:
        raise RuntimeError("gold leaked into effnet train ids")
    rng = np.random.default_rng(HOLD_SEED)
    perm = rng.permutation(len(train_ids))
    n_ho = max(1, min(len(train_ids) - 1, int(round(len(train_ids) * HOLD_FRAC))))
    chosen = np.sort(perm[:n_ho])
    ids = [train_ids[int(i)] for i in chosen.tolist()]
    if set(ids) & gold:
        raise RuntimeError("gold leaked into effnet holdout")
    print(
        f"[effnet870] holdout n={len(ids)} gold=58 excluded leaked=0 seed={HOLD_SEED}",
        flush=True,
    )
    return ids


def _to_224(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., 0]
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip(img, lo, hi)
    img = (img - img.min()) / (float(img.max() - img.min()) + 1e-6)
    side = min(img.shape)
    y0 = (img.shape[0] - side) // 2
    x0 = (img.shape[1] - side) // 2
    img = img[y0:y0 + side, x0:x0 + side]
    ten = torch.from_numpy(img)[None, None]
    ten = torch.nn.functional.interpolate(ten, size=(IMG, IMG), mode="bilinear", align_corners=False)
    return ten[0, 0].numpy()


def _read_slice(path: str) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return _to_224(img * slope + intercept)


def _series_slices(comp: str, study: str, series: str) -> list[str]:
    folder = os.path.join(comp, "train_series", study, series)
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".dcm")
    ]
    files.sort()
    if len(files) <= NSLC:
        return files
    idx = np.linspace(0, len(files) - 1, NSLC).astype(int)
    return [files[int(i)] for i in idx]


def _pick_series(rows: pd.DataFrame) -> list[str]:
    order = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Axial", 1)]
    picked: list[str] = []
    used: set[str] = set()
    for plane, fluid in order:
        hit = rows[(rows["Anatomical_Plane"] == plane) & (rows["Fluid_Sensitive"].astype(int) == fluid)]
        for sid in hit["SeriesInstanceUID"].astype(str):
            if sid not in used:
                picked.append(sid)
                used.add(sid)
                break
        if len(picked) >= NSER:
            return picked
    for sid in rows["SeriesInstanceUID"].astype(str):
        if sid not in used:
            picked.append(sid)
            used.add(sid)
        if len(picked) >= NSER:
            break
    return picked


def build_net():
    import timm
    bb = timm.create_model(
        "tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        num_classes=0,
        in_chans=1,
        global_pool="avg",
    )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = bb
            self.att = nn.Sequential(nn.Linear(1280, 256), nn.Tanh(), nn.Linear(256, 1))
            self.head = nn.Sequential(nn.Identity(), nn.Linear(1280, 12))

        def forward(self, x):
            b, k = x.shape[:2]
            feat = self.enc(x.flatten(0, 1)).view(b, k, -1)
            attn = torch.softmax(self.att(feat), dim=1)
            pooled = (attn * feat).sum(1)
            return self.head(pooled)

    net = Net()
    ck = torch.load(_find_file("model.pt"), map_location="cpu", weights_only=False)
    if ck.get("backbone") != "tf_efficientnet_b0.ns_jft_in1k":
        raise RuntimeError(f"unexpected backbone {ck.get('backbone')}")
    if list(ck.get("lab")) != TARGETS:
        raise RuntimeError("checkpoint labels are not the official 12")
    missing, unexpected = net.load_state_dict(ck["sd"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict missing={len(missing)} unexpected={len(unexpected)}")
    print(
        f"[effnet870] loaded b0 sz={ck.get('sz')} nser={ck.get('nser')} nslc={ck.get('nslc')} "
        "strict_ok=1. DO NOT SUBMIT.",
        flush=True,
    )
    return net


def main() -> None:
    comp = _comp_root()
    train = pd.read_csv(os.path.join(comp, "train.csv"), dtype={"StudyInstanceUID": str})
    v4 = pd.read_csv(_find_file("llm_labels_v4_blend.csv"), dtype={"StudyInstanceUID": str})
    ids = holdout_ids(train, v4)
    series = pd.read_csv(os.path.join(comp, "train_series.csv"), dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_net().to(device).eval()
    rows = []
    missing_series = 0
    with torch.inference_mode():
        for n_done, uid in enumerate(ids, start=1):
            sub = series[series["StudyInstanceUID"] == uid]
            chosen = _pick_series(sub) if len(sub) else []
            if not chosen:
                missing_series += 1
            stack = np.zeros((NSER * NSLC, IMG, IMG), dtype=np.float32)
            slot = 0
            for sid in chosen:
                for path in _series_slices(comp, uid, sid):
                    if slot >= NSER * NSLC:
                        break
                    try:
                        stack[slot] = _read_slice(path)
                    except Exception:
                        stack[slot] = 0.0
                    slot += 1
            ten = torch.from_numpy(stack)[None, :, None].to(device)
            logit = net(ten)[0].float().cpu().numpy()
            prob = 1.0 / (1.0 + np.exp(-logit))
            rows.append([uid, *prob.tolist()])
            if n_done % 50 == 0:
                print(f"[effnet870] {n_done}/{len(ids)} missing_series={missing_series}", flush=True)
    out = pd.DataFrame(rows, columns=HEADER)
    dest = "/kaggle/working/effnet_holdout870.csv"
    os.makedirs("/kaggle/working", exist_ok=True)
    out.to_csv(dest, index=False)
    note = (
        "DO NOT SUBMIT vitaliecervinschi/rsna-knee-effnet870.\n"
        "Train holdout IDs. Reconstructed geometry. Loss is inconclusive.\n"
        "Keep 55779066 unless a local mill-870 join licenses a subset.\n"
    )
    with open("/kaggle/working/DO_NOT_SUBMIT_HOLDOUT.txt", "w", encoding="utf-8") as fh:
        fh.write(note)
    print(
        f"[effnet870] wrote {dest} shape={out.shape} missing_series={missing_series}. DO NOT SUBMIT.",
        flush=True,
    )


if __name__ == "__main__":
    main()
