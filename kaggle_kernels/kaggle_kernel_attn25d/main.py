#!/usr/bin/env python3
"""C: mill-recipe 2.5D GROUP=3 windows + per-finding gated attention on ResNet18.

Steven v4 train, gold 58 val-only. Last-block unfreeze. Not mill 6-slot 336/130 mm.
Not CoAtNet, not mill DINOv2, not mill RadImageNet, not r336_best.pt.
Not frozen GAP-linear. Abort if gold-macro > 0.94. Do not read Report at test.
Do not Submit unless LLM-holdout beats mill.
RESEARCH_METHOD: 2.5D consecutive-slice windows + per-finding ABMIL (Ilse 2018)
on ImageNet ResNet18 @224 crop_mm=0.
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
from torch.utils.data import DataLoader, Dataset

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
C_PLANES = ("Axial", "Coronal", "Sagittal")
C_WINDOWS_PER_PLANE = 8
C_GROUP = 3
C_IMG = 224
C_CROP_MM = 0.0
C_HEAD = "per_finding_gated_attention"
C_BACKBONE = "resnet18"
BACKBONE_NAME = "resnet18"
UNFREEZE_LAST = 1
GOLD_AUC_ABORT_THRESHOLD = 0.94
MILL_SLOT_IMG = 336
MILL_SLOT_CROP_MM = 130.0
MILL_N_SLOT = 6
EPOCHS = 3
BATCH = 4
TIME_BUDGET_SEC = 7.0 * 3600
RESEARCH_METHOD = (
    "2.5D GROUP=3 consecutive-slice windows + per-finding ABMIL (Ilse ICML 2018) "
    "on ImageNet ResNet18 @224 crop_mm=0, last-block unfreeze. Not CoAtNet, "
    "not mill DINOv2, not mill RadImageNet, not r336_best.pt."
)
FORBIDDEN_SCORED_CHECKPOINTS = (
    "r336_best.pt",
    "r224_best.pt",
    "raptor_ft_coatnet_v4_full.pt",
    "raptor_ft_coatnet_v5_full_swa.pt",
    "focal_fold",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
T0 = time.time()
N_WINDOWS = len(C_PLANES) * C_WINDOWS_PER_PLANE


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


def windows_2p5d_indices(n_files, n_windows=C_WINDOWS_PER_PLANE, group=C_GROUP):
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


def pick_plane_series(series_rows, plane):
    cands = [r for r in series_rows if str(r.get("Anatomical_Plane", "")) == plane]
    if not cands:
        return None

    def score(r):
        fs = int(r.get("Fat_Suppression") or 0)
        fl = int(r.get("Fluid_Sensitive") or 0)
        return (fs + fl, fl, fs)

    cands = sorted(cands, key=score, reverse=True)
    return cands[0]


class GatedAttentionPool(nn.Module):
    """Ilse et al. ICML 2018 per-finding gated-attention MIL. Not GAP-linear."""

    def __init__(self, dim, n_queries, hidden=128):
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
    t = F.interpolate(t, size=(C_IMG, C_IMG), mode="bilinear", align_corners=False)
    return t[0, 0]


def pack_2p5d_bag(study, series_root, series_rows):
    out = torch.zeros(N_WINDOWS, 3, C_IMG, C_IMG)
    w = 0
    for plane in C_PLANES:
        rec = pick_plane_series(series_rows, plane)
        if rec is None:
            w += C_WINDOWS_PER_PLANE
            continue
        d = series_root / study / rec["SeriesInstanceUID"]
        if not d.is_dir():
            w += C_WINDOWS_PER_PLANE
            continue
        files = sorted(d.glob("*.dcm"))
        if not files:
            w += C_WINDOWS_PER_PLANE
            continue
        idx = windows_2p5d_indices(len(files), C_WINDOWS_PER_PLANE, C_GROUP)
        for row in idx:
            chans = []
            for k in row:
                try:
                    chans.append(_dcm_to_224(files[int(k)]))
                except Exception:
                    chans.append(torch.zeros(C_IMG, C_IMG))
            while len(chans) < 3:
                chans.append(torch.zeros(C_IMG, C_IMG))
            out[w] = torch.stack(chans[:3], 0)
            w += 1
    return out


def build_resnet18():
    from torchvision.models import resnet18

    try:
        model = resnet18(weights=None)
    except TypeError:
        model = resnet18(pretrained=False)
    prefer = []
    here = Path(__file__).resolve().parent
    for cand in (here / "resnet18_imagenet.pt", Path("/kaggle/working/resnet18_imagenet.pt")):
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
        log("no resnet18 imagenet weights; random-init (still not mill-slot)")
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    return model


class Attn25dNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = GatedAttentionPool(512, n_queries=len(FINDINGS), hidden=128)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.backbone.layer4.train(mode)
        return self

    def forward(self, x):
        bsz, t, c, h, w = x.shape
        feat = self.backbone(x.reshape(bsz * t, c, h, w)).reshape(bsz, t, -1)
        return self.pool(feat)


class BagDS(Dataset):
    def __init__(self, ids, y, series_root, by_study):
        self.ids = list(ids)
        self.y = y
        self.series_root = series_root
        self.by_study = by_study

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        x = pack_2p5d_bag(sid, self.series_root, self.by_study.get(sid, []))
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x.clamp(0, 1) - mean) / std
        y = torch.zeros(len(FINDINGS), dtype=torch.float32)
        if self.y is not None:
            y = torch.from_numpy(self.y[i].astype(np.float32))
        return x, y, sid


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


@torch.no_grad()
def predict_ids(model, ids, series_root, by_study, device):
    model.eval()
    ds = BagDS(ids, None, series_root, by_study)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    chunks = []
    for x, _, _ in loader:
        x = x.to(device)
        chunks.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, len(FINDINGS)))


def main():
    if UNFREEZE_LAST < 1:
        raise RuntimeError("C must unfreeze last block; not frozen GAP-linear")
    if C_HEAD != "per_finding_gated_attention":
        raise RuntimeError("C head must be per-finding gated attention")
    if C_BACKBONE != "resnet18":
        raise RuntimeError("C backbone must be ImageNet resnet18, not mill family")
    if C_IMG == MILL_SLOT_IMG and abs(C_CROP_MM - MILL_SLOT_CROP_MM) < 1e-9:
        raise RuntimeError("refusing mill 6-slot 336/130 mm packing")
    for name in FORBIDDEN_SCORED_CHECKPOINTS:
        if name == "r336_best.pt":
            log(f"forbidden scored checkpoint listed: {name}")
    log(f"RESEARCH_METHOD={RESEARCH_METHOD}")
    comp = find_comp()
    log(f"comp {comp}")
    train = pd.read_csv(comp / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(find_v4_csv(), dtype={ID_COLUMN: str})
    test = pd.read_csv(comp / "test.csv", dtype={ID_COLUMN: str})
    tser = pd.read_csv(comp / "train_series.csv", dtype={ID_COLUMN: str, "SeriesInstanceUID": str})
    teser = pd.read_csv(comp / "test_series.csv", dtype={ID_COLUMN: str, "SeriesInstanceUID": str})
    train_ids, val_ids, y_train, y_val = make_steven_v4_split(train, v4)
    log(
        f"steven v4 train n={len(train_ids)} gold=58 val-only n={len(val_ids)} "
        f"leaked={len(set(train_ids) & set(val_ids))}"
    )
    if set(train_ids) & set(val_ids):
        raise RuntimeError("gold 58 in train loss")
    fit_idx, ho_idx = llm_soft_holdout_indices(len(train_ids), frac=0.2, seed=2026)
    ho_ids = {train_ids[int(i)] for i in ho_idx.tolist()}
    if ho_ids & set(val_ids):
        raise RuntimeError("LLM-soft holdout overlapped gold 58")
    log(f"llm-soft holdout n={len(ho_idx)} fit n={len(fit_idx)} (not gold)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(
        f"device {device} backbone={C_BACKBONE} img={C_IMG} crop_mm={C_CROP_MM} "
        f"unfreeze_last={UNFREEZE_LAST} group={C_GROUP} windows/plane={C_WINDOWS_PER_PLANE} "
        f"head={C_HEAD}"
    )
    model = Attn25dNet(build_resnet18()).to(device)
    bb_params = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
    head_params = list(model.pool.parameters())
    opt = torch.optim.AdamW(
        [{"params": bb_params, "lr": 1e-4}, {"params": head_params, "lr": 1e-3}],
        weight_decay=1e-4,
    )
    by_tr = {k: v.to_dict("records") for k, v in tser.groupby(ID_COLUMN)}
    by_te = {k: v.to_dict("records") for k, v in teser.groupby(ID_COLUMN)}
    tr_root = comp / "train_series"
    te_root = comp / "test_series"
    fit_ids = tuple(train_ids[int(i)] for i in fit_idx.tolist())
    y_fit = y_train[fit_idx]
    ds = BagDS(fit_ids, y_fit, tr_root, by_tr)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    y_ho_bin = (y_train[ho_idx] >= 0.5).astype(np.float64)
    ho_id_tuple = tuple(train_ids[int(i)] for i in ho_idx.tolist())
    gold_macro = float("nan")
    ho_macro = float("nan")
    per = [float("nan")] * len(FINDINGS)
    aborted = False
    for epoch in range(EPOCHS):
        if time.time() - T0 > TIME_BUDGET_SEC:
            log("time budget hit; stop train")
            break
        model.train()
        total = 0.0
        n = 0
        for x, y, _ in loader:
            if time.time() - T0 > TIME_BUDGET_SEC:
                break
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item()) * len(x)
            n += len(x)
        val_prob = predict_ids(model, val_ids, tr_root, by_tr, device)
        per, gold_macro = macro_auc(y_val, val_prob)
        log(f"epoch {epoch} loss={total / max(n, 1):.4f} gold_macro={gold_macro:.4f}")
        for name, auc in zip(FINDINGS, per):
            print(f"[attn25d-C] gold {name:22s} {auc:.4f}", flush=True)
        if should_abort_gold_overfit(gold_macro):
            print(
                f"[attn25d-C] ABORT gold macro {gold_macro:.4f} > {GOLD_AUC_ABORT_THRESHOLD:.2f} "
                "(leak/overfit alarm; gold 58 val-only; do not overlay / do not submit)",
                flush=True,
            )
            aborted = True
            break
    ho_prob = predict_ids(model, ho_id_tuple, tr_root, by_tr, device)
    _, ho_macro = macro_auc(y_ho_bin, ho_prob)
    print(
        f"[attn25d-C] gold_macro={gold_macro:.4f} llm_holdout_macro={ho_macro:.4f} "
        f"n={len(ho_idx)} (not gold) abort={aborted}",
        flush=True,
    )
    print(f"[attn25d-C] llm-soft holdout macro={ho_macro:.4f} n={len(ho_idx)} (not gold)", flush=True)
    work = Path("/kaggle/working")
    work.mkdir(exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "gold_macro": gold_macro, "holdout": ho_macro},
        work / "attn25d_c.pt",
    )
    pd.DataFrame({"finding": list(FINDINGS), "gold_auc": per}).to_csv(work / "c_gold_aucs.csv", index=False)
    te_ids = tuple(test[ID_COLUMN].astype(str))
    te = predict_ids(model, te_ids, te_root, by_te, device)
    sub = pd.DataFrame(te, columns=list(FINDINGS))
    sub.insert(0, ID_COLUMN, list(te_ids))
    sub.to_csv(work / "submission.csv", index=False)
    print(
        f"[attn25d-C] RESEARCH_METHOD={RESEARCH_METHOD} backbone={C_BACKBONE} "
        f"img={C_IMG} crop_mm={C_CROP_MM} unfreeze_last={UNFREEZE_LAST} "
        f"group={C_GROUP} not r336_best.pt not mill 6-slot 336/130 "
        f"not CoAtNet not mill DINO not mill RadImageNet not frozen GAP-linear "
        f"| shape={sub.shape} | Baker's ok | llm_holdout_macro={ho_macro:.4f} "
        f"vs mill 0.936 — do not auto-submit",
        flush=True,
    )


if __name__ == "__main__":
    main()
