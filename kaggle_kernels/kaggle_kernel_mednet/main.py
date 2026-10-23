#!/usr/bin/env python3
"""Frozen MedicalNet 3D ResNet-50 encoder + NEW linear head on steven v4.

Gold 58 is val-only never train. Not mill 6-slot 336/130 mm. Not r336_best.pt.
MedicalNet conv_seg is NOT the scored 12-head. unfreeze_last=0.
Abort if gold macro AUC > 0.94. LLM-soft holdout is not gold. Do not read Report at test.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _gpu_is_t4_or_newer():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        print("nvidia-smi:", out.strip(), flush=True)
        bad = ("P100" in out) or ("6.0" in out)
        good = any(x in out for x in ("T4", "V100", "A100", "A10", "L4", "7.5", "8.0", "8.6", "8.9", "9.0"))
        return (not bad) or good
    except Exception as exc:
        print("nvidia-smi probe failed:", exc, flush=True)
        return True


if not _gpu_is_t4_or_newer():
    print("Hiding pre-Volta GPU before importing torch", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import glob
import time

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ID_COLUMN = "StudyInstanceUID"
FINDINGS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
BACKBONE_NAME = "medicalnet_resnet50_3d"
BACKBONE_DEPTH = 32
BACKBONE_HW = 112
BACKBONE_CROP_MM = 0.0
UNFREEZE_LAST = 0
FEATURE_DIM = 2048
GOLD_AUC_ABORT_THRESHOLD = 0.94
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
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def should_abort_gold_overfit(gold_macro, threshold=GOLD_AUC_ABORT_THRESHOLD):
    return float(gold_macro) > float(threshold)


def find_comp():
    for p in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (p / "test.csv").is_file():
            return p
    raise FileNotFoundError("competition test.csv not mounted")


def find_file(name):
    skip = {"train_series", "test_series", "test_images"}
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    stack = [base]
    depth = 0
    while stack and depth < 80:
        cur = stack.pop(0)
        depth += 1
        try:
            kids = list(cur.iterdir())
        except OSError:
            continue
        names = {k.name for k in kids}
        if name in names:
            return cur / name
        for k in kids:
            if k.is_dir() and k.name not in skip and not k.name.startswith("1.2."):
                stack.append(k)
    return None


def expert_gold_ids(train):
    mask = train[list(FINDINGS)].notna().all(axis=1)
    return tuple(train.loc[mask, ID_COLUMN].astype(str).tolist())


def make_steven_v4_split(train, v4):
    gold = expert_gold_ids(train)
    gold_set = set(gold)
    gold_frame = train[train[ID_COLUMN].astype(str).isin(gold_set)].drop_duplicates(ID_COLUMN)
    gold_frame = gold_frame.set_index(ID_COLUMN).loc[list(gold)]
    y_val = gold_frame[list(FINDINGS)].to_numpy(dtype=np.float64)
    labels = v4.copy()
    labels[ID_COLUMN] = labels[ID_COLUMN].astype(str)
    labels = labels.drop_duplicates(ID_COLUMN)
    train_frame = labels[~labels[ID_COLUMN].isin(gold_set)].copy()
    if set(train_frame[ID_COLUMN]) & gold_set:
        raise ValueError("gold 58 leaked into train")
    return (
        tuple(train_frame[ID_COLUMN].tolist()),
        gold,
        train_frame[list(FINDINGS)].to_numpy(dtype=np.float64),
        y_val,
    )


def llm_soft_holdout_indices(n, frac=0.2, seed=2026):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_ho = max(1, min(int(n) - 1, int(round(int(n) * float(frac)))))
    return np.sort(perm[n_ho:]), np.sort(perm[:n_ho])


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedNetEncoder(nn.Module):
    """MedicalNet ResNet-50 through layer4 + GAP. conv_seg is NOT used as a 12-head."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=1, dilation=2)
        self.layer4 = self._make_layer(512, 3, stride=1, dilation=4)
        # Unused segmentation tower kept only so leftover ckpt keys are obvious, never scored.
        self.conv_seg = nn.Identity()

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * 4),
            )
        layers = [_Bottleneck(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


def build_encoder():
    model = MedNetEncoder()
    path = find_file("resnet_50_23dataset.pth") or find_file("resnet_50.pth")
    if path is None:
        raise FileNotFoundError("MedicalNet resnet_50_23dataset.pth not mounted")
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    sd = {}
    for k, v in blob.items():
        nk = k.replace("module.", "")
        if nk.startswith("conv_seg"):
            continue
        sd[nk] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"loaded encoder-only {path.name} missing={len(missing)} unexpected={len(unexpected)}")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def _dcm_plane(path, hw):
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    t = torch.from_numpy(arr)[None, None]
    t = F.interpolate(t, size=(hw, hw), mode="bilinear", align_corners=False)
    return t[0, 0]


def study_volume(study, series_root, series_rows):
    """1 x D x H x W from sagittal (else any) series. Pixel resize, not 130 mm mill slots."""
    vol = torch.zeros(1, BACKBONE_DEPTH, BACKBONE_HW, BACKBONE_HW)
    files = []
    sag = [r for r in series_rows if str(r.get("Anatomical_Plane", "")) == "Sagittal"]
    pool = sag or list(series_rows)
    for rec in pool:
        d = series_root / study / rec["SeriesInstanceUID"]
        if d.is_dir():
            cand = sorted(d.glob("*.dcm"))
            if len(cand) > len(files):
                files = cand
    n = len(files)
    if n == 0:
        return vol
    idx = np.unique(np.linspace(0, n - 1, BACKBONE_DEPTH).astype(int))
    for j, k in enumerate(idx[:BACKBONE_DEPTH]):
        try:
            vol[0, j] = _dcm_plane(files[int(k)], BACKBONE_HW)
        except Exception:
            continue
    return vol


@torch.no_grad()
def encode_ids(ids, series_root, by_study, model, device):
    feats = []
    model = model.to(device)
    for i, sid in enumerate(ids):
        vol = study_volume(sid, series_root, by_study.get(sid, [])).unsqueeze(0).to(device)
        feats.append(model(vol).cpu())
        if (i + 1) % 50 == 0:
            log(f"encode {i + 1}/{len(ids)}")
    return torch.cat(feats, 0)


def macro_auc(y, s):
    per = []
    for j in range(y.shape[1]):
        yt = y[:, j]
        if len(np.unique(yt)) < 2:
            per.append(float("nan"))
            continue
        per.append(float(roc_auc_score(yt, s[:, j])))
    vals = [v for v in per if v == v]
    return per, (float(np.mean(vals)) if vals else float("nan"))


def main():
    if UNFREEZE_LAST != 0:
        raise RuntimeError("unfreeze_last must stay 0")
    if BACKBONE_HW == MILL_SLOT_IMG and abs(BACKBONE_CROP_MM - MILL_SLOT_CROP_MM) < 1e-9:
        raise RuntimeError("refusing mill 6-slot 336/130 mm packing")
    log(f"forbidden scored checkpoint listed: r336_best.pt")
    comp = find_comp()
    v4_csv = find_file("llm_labels_v4_blend.csv")
    if v4_csv is None:
        raise FileNotFoundError("steven v4 llm_labels_v4_blend.csv not mounted")
    train = pd.read_csv(comp / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(v4_csv, dtype={ID_COLUMN: str})
    test = pd.read_csv(comp / "test.csv", dtype={ID_COLUMN: str})
    tser = pd.read_csv(comp / "train_series.csv", dtype={ID_COLUMN: str, "SeriesInstanceUID": str})
    teser = pd.read_csv(comp / "test_series.csv", dtype={ID_COLUMN: str, "SeriesInstanceUID": str})
    train_ids, val_ids, y_train, y_val = make_steven_v4_split(train, v4)
    log(f"steven v4 train n={len(train_ids)} gold=58 val-only n={len(val_ids)} leaked={len(set(train_ids)&set(val_ids))}")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("gold 58 in train loss")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device} backbone={BACKBONE_NAME} D={BACKBONE_DEPTH} HW={BACKBONE_HW} unfreeze_last={UNFREEZE_LAST}")
    encoder = build_encoder()
    by_tr = {k: v.to_dict("records") for k, v in tser.groupby(ID_COLUMN)}
    by_te = {k: v.to_dict("records") for k, v in teser.groupby(ID_COLUMN)}
    x_train = encode_ids(train_ids, comp / "train_series", by_tr, encoder, device)
    x_val = encode_ids(val_ids, comp / "train_series", by_tr, encoder, device)
    x_test = encode_ids(tuple(test[ID_COLUMN].astype(str)), comp / "test_series", by_te, encoder, device)
    dim = int(x_train.shape[1])
    if dim != FEATURE_DIM:
        log(f"feature dim {dim} (expected {FEATURE_DIM})")
    fit_idx, ho_idx = llm_soft_holdout_indices(len(train_ids), frac=0.2, seed=2026)
    if {train_ids[int(i)] for i in ho_idx.tolist()} & set(val_ids):
        raise RuntimeError("LLM-soft holdout overlapped gold 58")
    log(f"llm-soft holdout n={len(ho_idx)} fit n={len(fit_idx)} (not gold)")
    head = nn.Linear(dim, len(FINDINGS)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    ytr = torch.from_numpy(y_train).float()
    y_ho_bin = (y_train[ho_idx] >= 0.5).astype(np.float64)
    fit_t = torch.from_numpy(fit_idx).long()
    bs = 64
    for epoch in range(8):
        head.train()
        perm = torch.from_numpy(np.random.default_rng(2026 + epoch).permutation(len(fit_t)))
        total = 0.0
        n = 0
        for s in range(0, len(perm), bs):
            idx = fit_t[perm[s : s + bs]]
            logits = head(x_train[idx].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, ytr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
            n += len(idx)
        head.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(head(x_val.to(device))).cpu().numpy()
            ho_prob = torch.sigmoid(head(x_train[torch.from_numpy(ho_idx)].to(device))).cpu().numpy()
        per, macro = macro_auc(y_val, val_prob)
        _, ho_macro = macro_auc(y_ho_bin, ho_prob)
        log(f"epoch {epoch} loss={total / max(n, 1):.4f} gold_macro={macro:.4f} llm_holdout_macro={ho_macro:.4f}")
        if should_abort_gold_overfit(macro):
            print(
                f"[mednet] ABORT gold macro {macro:.4f} > {GOLD_AUC_ABORT_THRESHOLD:.2f} "
                "(leak/overfit alarm; gold 58 val-only; do not overlay / do not submit)",
                flush=True,
            )
            break
    else:
        print(
            f"[mednet] gold_macro={macro:.4f} llm_holdout_macro={ho_macro:.4f} "
            "under abort threshold; still not auto-submit",
            flush=True,
        )
    for name, auc in zip(FINDINGS, per):
        print(f"[mednet] gold {name:22s} {auc:.4f}", flush=True)
    print(f"[mednet] llm-soft holdout macro={ho_macro:.4f} n={len(ho_idx)} (not gold)", flush=True)
    work = Path("/kaggle/working")
    work.mkdir(exist_ok=True)
    torch.save({"head": head.state_dict(), "dim": dim, "macro": macro}, work / "mednet_linear_head.pt")
    pd.DataFrame({"finding": list(FINDINGS), "gold_auc": per}).to_csv(work / "mednet_gold_aucs.csv", index=False)
    head.eval()
    with torch.no_grad():
        te = torch.sigmoid(head(x_test.to(device))).cpu().numpy()
    sub = pd.DataFrame(te, columns=list(FINDINGS))
    sub.insert(0, ID_COLUMN, test[ID_COLUMN].astype(str).tolist())
    sub.to_csv(work / "submission.csv", index=False)
    print(
        f"[mednet] backbone={BACKBONE_NAME} D={BACKBONE_DEPTH} HW={BACKBONE_HW} crop_mm={BACKBONE_CROP_MM} "
        f"unfreeze_last={UNFREEZE_LAST} gold=58 val-only steven v4 train encoder-only MedicalNet "
        f"not r336_best.pt not mill 6-slot 336/130 | shape={sub.shape} | Baker's ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
