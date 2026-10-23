#!/usr/bin/env python3
"""Steven v4 linear head on frozen ResNet18 224, 3-plane 8-slice stack.

Gold 58 is val-only never train. Not mill 6-slot 336/130 mm. Not r336_best.pt.
unfreeze_last stays 0. Abort if gold macro AUC > 0.94. Do not read Report at test.
"""
from __future__ import annotations

import os
import subprocess
import sys
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
import math
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
BACKBONE_NAME = "resnet18"
BACKBONE_IMG = 224
BACKBONE_CROP_MM = 0.0
PLANES = ("Axial", "Coronal", "Sagittal")
SLICES_PER_PLANE = 8
UNFREEZE_LAST = 0
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


class GoldOverfitAbort(RuntimeError):
    pass


def should_abort_gold_overfit(gold_macro, threshold=GOLD_AUC_ABORT_THRESHOLD):
    return float(gold_macro) > float(threshold)


def find_comp():
    cands = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for p in cands:
        if (p / "test.csv").is_file():
            return p
    raise FileNotFoundError("competition test.csv not mounted")


def find_v4_csv():
    skip = {"train_series", "test_series", "test_images"}
    base = Path("/kaggle/input")
    hits = []
    if base.is_dir():
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
            if "llm_labels_v4_blend.csv" in names:
                hits.append(cur / "llm_labels_v4_blend.csv")
            for k in kids:
                if k.is_dir() and k.name not in skip and not k.name.startswith("1.2."):
                    stack.append(k)
    if not hits:
        raise FileNotFoundError("steven v4 llm_labels_v4_blend.csv not mounted")
    return hits[0]


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
    holdout = np.sort(perm[:n_ho])
    fit = np.sort(perm[n_ho:])
    return fit, holdout


def _dcm_to_224(path):
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    t = torch.from_numpy(arr)[None, None]
    t = F.interpolate(t, size=(BACKBONE_IMG, BACKBONE_IMG), mode="bilinear", align_corners=False)
    return t[0, 0]


def study_stack(study, series_root, series_rows):
    """24 x 224 x 224 from 8 slices x 3 planes. Pixel resize, not 130 mm."""
    out = torch.zeros(len(PLANES) * SLICES_PER_PLANE, BACKBONE_IMG, BACKBONE_IMG)
    by_plane = {p: [] for p in PLANES}
    for rec in series_rows:
        plane = str(rec.get("Anatomical_Plane", ""))
        sid = rec["SeriesInstanceUID"]
        d = series_root / study / sid
        if plane in by_plane and d.is_dir():
            files = sorted(d.glob("*.dcm"))
            if files:
                by_plane[plane].append(files)
    for i, plane in enumerate(PLANES):
        files = []
        if by_plane[plane]:
            files = by_plane[plane][0]
        n = len(files)
        if n == 0:
            continue
        idx = np.unique(np.linspace(0, n - 1, SLICES_PER_PLANE).astype(int))
        for j, k in enumerate(idx[:SLICES_PER_PLANE]):
            try:
                out[i * SLICES_PER_PLANE + j] = _dcm_to_224(files[int(k)])
            except Exception:
                continue
    return out


def build_resnet18():
    from torchvision.models import resnet18

    try:
        model = resnet18(weights=None)
    except TypeError:
        model = resnet18(pretrained=False)
    weight_path = None
    prefer = []
    here = Path(__file__).resolve().parent
    for cand in (
        here / "resnet18_imagenet.pt",
        Path("/kaggle/working/resnet18_imagenet.pt"),
    ):
        if cand.is_file():
            prefer.append(cand)
    hits = [
        Path(h)
        for h in glob.glob("/kaggle/input/**/resnet18*.pth", recursive=True)
        + glob.glob("/kaggle/input/**/resnet18*.pt", recursive=True)
        if "train_series" not in h and "multi_task" not in h
    ]
    hits.sort(key=lambda p: (0 if "f37072fd" in p.name or "5c106cde" in p.name else 1, p.name))
    prefer.extend(hits)
    if prefer:
        weight_path = prefer[0]
        sd = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"loaded 3-ch {weight_path.name} missing={len(missing)} unexpected={len(unexpected)}")
    else:
        log("no resnet18 imagenet weights; random-init frozen backbone (still not mill-slot)")
    w = model.conv1.weight.data
    conv = nn.Conv2d(24, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv.weight.data = w.repeat(1, 8, 1, 1)[:, :24] * (3.0 / 24.0)
    model.conv1 = conv
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@torch.no_grad()
def encode_ids(ids, series_root, by_study, model, device):
    feats = []
    model = model.to(device)
    for i, sid in enumerate(ids):
        stack = study_stack(sid, series_root, by_study.get(sid, []))
        x = stack.unsqueeze(0).to(device)
        if x.max() > 1.5:
            x = x / 255.0
        x = x.clamp(0, 1)
        mean = torch.tensor([0.485]).view(1, 1, 1, 1).to(device)
        std = torch.tensor([0.229]).view(1, 1, 1, 1).to(device)
        x = (x - mean) / std
        f = model(x).reshape(1, -1)
        feats.append(f.cpu())
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
    if BACKBONE_IMG == MILL_SLOT_IMG and abs(BACKBONE_CROP_MM - MILL_SLOT_CROP_MM) < 1e-9:
        raise RuntimeError("refusing mill 6-slot 336/130 mm packing")
    for name in FORBIDDEN_SCORED_CHECKPOINTS:
        if name == "r336_best.pt":
            log(f"forbidden scored checkpoint listed: {name}")
    comp = find_comp()
    log(f"comp {comp}")
    train = pd.read_csv(comp / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(find_v4_csv(), dtype={ID_COLUMN: str})
    test = pd.read_csv(comp / "test.csv", dtype={ID_COLUMN: str})
    tser = pd.read_csv(comp / "train_series.csv", dtype={ID_COLUMN: str, "SeriesInstanceUID": str})
    teser = pd.read_csv(comp / "test_series.csv", dtype={ID_COLUMN: str, "SeriesInstanceUID": str})
    train_ids, val_ids, y_train, y_val = make_steven_v4_split(train, v4)
    log(f"steven v4 train n={len(train_ids)} gold=58 val-only n={len(val_ids)} leaked={len(set(train_ids)&set(val_ids))}")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("gold 58 in train loss")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device} backbone={BACKBONE_NAME} img={BACKBONE_IMG} unfreeze_last={UNFREEZE_LAST}")
    model = build_resnet18()
    by_tr = {k: v.to_dict("records") for k, v in tser.groupby(ID_COLUMN)}
    by_te = {k: v.to_dict("records") for k, v in teser.groupby(ID_COLUMN)}
    tr_root = comp / "train_series"
    te_root = comp / "test_series"
    x_train = encode_ids(train_ids, tr_root, by_tr, model, device)
    x_val = encode_ids(val_ids, tr_root, by_tr, model, device)
    x_test = encode_ids(tuple(test[ID_COLUMN].astype(str)), te_root, by_te, model, device)
    dim = int(x_train.shape[1])
    fit_idx, ho_idx = llm_soft_holdout_indices(len(train_ids), frac=0.2, seed=2026)
    ho_ids = {train_ids[int(i)] for i in ho_idx.tolist()}
    if ho_ids & set(val_ids):
        raise RuntimeError("LLM-soft holdout overlapped gold 58")
    log(f"llm-soft holdout n={len(ho_idx)} fit n={len(fit_idx)} (not gold)")
    head = nn.Linear(dim, len(FINDINGS)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    ytr = torch.from_numpy(y_train).float()
    y_ho_bin = (y_train[ho_idx] >= 0.5).astype(np.float64)
    bs = 64
    for epoch in range(8):
        head.train()
        perm = torch.from_numpy(np.random.default_rng(2026 + epoch).permutation(len(fit_idx)))
        total = 0.0
        n = 0
        for s in range(0, len(perm), bs):
            local = perm[s : s + bs]
            idx = torch.from_numpy(fit_idx[local.numpy()]).long()
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
        log(
            f"epoch {epoch} loss={total / max(n, 1):.4f} gold_macro={macro:.4f} "
            f"llm_holdout_macro={ho_macro:.4f}"
        )
        if should_abort_gold_overfit(macro):
            print(
                f"[v4head] ABORT gold macro {macro:.4f} > {GOLD_AUC_ABORT_THRESHOLD:.2f} "
                "(leak/overfit alarm; gold 58 val-only; do not overlay / do not submit)",
                flush=True,
            )
            break
    else:
        print(
            f"[v4head] gold_macro={macro:.4f} llm_holdout_macro={ho_macro:.4f} "
            "under abort threshold; still not auto-submit",
            flush=True,
        )
    for name, auc in zip(FINDINGS, per):
        print(f"[v4head] gold {name:22s} {auc:.4f}", flush=True)
    print(f"[v4head] llm-soft holdout macro={ho_macro:.4f} n={len(ho_idx)} (not gold)", flush=True)
    work = Path("/kaggle/working")
    work.mkdir(exist_ok=True)
    torch.save({"head": head.state_dict(), "dim": dim, "macro": macro}, work / "v4_linear_head.pt")
    pd.DataFrame({"finding": list(FINDINGS), "gold_auc": per}).to_csv(work / "v4_gold_aucs.csv", index=False)
    head.eval()
    with torch.no_grad():
        te = torch.sigmoid(head(x_test.to(device))).cpu().numpy()
    sub = pd.DataFrame(te, columns=list(FINDINGS))
    sub.insert(0, ID_COLUMN, test[ID_COLUMN].astype(str).tolist())
    sub.to_csv(work / "submission.csv", index=False)
    print(
        f"[v4head] backbone={BACKBONE_NAME} img={BACKBONE_IMG} crop_mm={BACKBONE_CROP_MM} "
        f"unfreeze_last={UNFREEZE_LAST} gold=58 val-only steven v4 train "
        f"not r336_best.pt not mill 6-slot 336/130 | shape={sub.shape} | Baker's ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
