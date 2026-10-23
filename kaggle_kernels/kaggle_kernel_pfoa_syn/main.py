#!/usr/bin/env python3
"""B: PF OA + Synovitis specialist. Axial-dense 24-slice FOV + gated-attention MIL.

Steven v4 train, gold 58 val-only. Last-block unfreeze. Not mill 6-slot 336/130 mm.
Not r336_best.pt. Not frozen GAP-linear. Abort if gold-macro > 0.94.
Do not read Report at test. Do not Submit unless LLM-holdout beats mill.
RESEARCH_METHOD: ABMIL gated attention (Ilse ICML 2018) + axial-dense
recess-inclusive FOV (MOAKS effusion-synovitis axial; PF cartilage axial).
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
B_PLANES = ("Axial",)
B_N_SLICES = 24
B_IMG = 224
B_CROP_MM = 0.0
B_TARGETS = ("PF OA", "Synovitis")
B_HEAD = "gated_attention_mil"
UNFREEZE_LAST = 1
BACKBONE_NAME = "resnet18"
GOLD_AUC_ABORT_THRESHOLD = 0.94
MILL_SLOT_IMG = 336
MILL_SLOT_CROP_MM = 130.0
MILL_N_SLOT = 6
EPOCHS = 3
BATCH = 4
TIME_BUDGET_SEC = 7.0 * 3600
RESEARCH_METHOD = (
    "ABMIL gated attention (Ilse ICML 2018) + axial-dense 24-slice "
    "recess-inclusive FOV crop_mm=0 (MOAKS effusion-synovitis axial; "
    "patellar cartilage en-face axial). Prefer FS axial. Last-block unfreeze."
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


def axial_dense_indices(n_files, n_slices=B_N_SLICES):
    n_files = int(n_files)
    if n_files <= 0 or n_slices <= 0:
        return np.zeros((0,), dtype=np.int64)
    n = min(int(n_slices), n_files)
    return np.unique(np.linspace(0, n_files - 1, n).astype(np.int64))


def pick_plane_series(series_rows, plane="Axial"):
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
    """Ilse et al. ICML 2018 gated-attention MIL. Not GAP-linear."""

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
    t = F.interpolate(t, size=(B_IMG, B_IMG), mode="bilinear", align_corners=False)
    return t[0, 0]


def pack_axial_bag(study, series_root, series_rows):
    out = torch.zeros(B_N_SLICES, 3, B_IMG, B_IMG)
    rec = pick_plane_series(series_rows, "Axial")
    if rec is None:
        return out
    d = series_root / study / rec["SeriesInstanceUID"]
    if not d.is_dir():
        return out
    files = sorted(d.glob("*.dcm"))
    if not files:
        return out
    idx = axial_dense_indices(len(files), B_N_SLICES)
    for j, k in enumerate(idx[:B_N_SLICES]):
        try:
            gray = _dcm_to_224(files[int(k)])
            out[j] = gray.unsqueeze(0).repeat(3, 1, 1)
        except Exception:
            continue
    return out


def specialist_fill_matrix(n, specialist_probs):
    out = np.full((n, len(FINDINGS)), 0.5, dtype=np.float64)
    idx = {f: i for i, f in enumerate(FINDINGS)}
    for j, t in enumerate(B_TARGETS):
        out[:, idx[t]] = specialist_probs[:, j]
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


class SpecialistNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = GatedAttentionPool(512, n_queries=len(B_TARGETS), hidden=128)

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
        x = pack_axial_bag(sid, self.series_root, self.by_study.get(sid, []))
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x.clamp(0, 1) - mean) / std
        y = torch.zeros(len(B_TARGETS), dtype=torch.float32)
        if self.y is not None:
            y = torch.from_numpy(self.y[i].astype(np.float32))
        return x, y, sid


def target_cols():
    return [FINDINGS.index(t) for t in B_TARGETS]


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
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, len(B_TARGETS)))


def main():
    if UNFREEZE_LAST < 1:
        raise RuntimeError("B must unfreeze last block; not frozen GAP-linear")
    if B_HEAD != "gated_attention_mil":
        raise RuntimeError("B head must be gated attention MIL")
    if B_IMG == MILL_SLOT_IMG and abs(B_CROP_MM - MILL_SLOT_CROP_MM) < 1e-9:
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
    train_ids, val_ids, y_train_all, y_val_all = make_steven_v4_split(train, v4)
    cols = target_cols()
    y_train = y_train_all[:, cols]
    y_val = y_val_all[:, cols]
    log(
        f"steven v4 train n={len(train_ids)} gold=58 val-only n={len(val_ids)} "
        f"leaked={len(set(train_ids) & set(val_ids))} targets={B_TARGETS}"
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
        f"device {device} backbone={BACKBONE_NAME} img={B_IMG} crop_mm={B_CROP_MM} "
        f"unfreeze_last={UNFREEZE_LAST} slices={B_N_SLICES} head={B_HEAD}"
    )
    model = SpecialistNet(build_resnet18()).to(device)
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
    per = [float("nan")] * len(B_TARGETS)
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
        for name, auc in zip(B_TARGETS, per):
            print(f"[pfoa-syn-B] gold {name:22s} {auc:.4f}", flush=True)
        if should_abort_gold_overfit(gold_macro):
            print(
                f"[pfoa-syn-B] ABORT gold macro {gold_macro:.4f} > {GOLD_AUC_ABORT_THRESHOLD:.2f} "
                "(leak/overfit alarm; gold 58 val-only; do not overlay / do not submit)",
                flush=True,
            )
            aborted = True
            break
    ho_prob = predict_ids(model, ho_id_tuple, tr_root, by_tr, device)
    _, ho_macro = macro_auc(y_ho_bin, ho_prob)
    print(
        f"[pfoa-syn-B] gold_macro={gold_macro:.4f} llm_holdout_macro={ho_macro:.4f} "
        f"n={len(ho_idx)} (not gold) abort={aborted}",
        flush=True,
    )
    print(f"[pfoa-syn-B] llm-soft holdout macro={ho_macro:.4f} n={len(ho_idx)} (not gold)", flush=True)
    work = Path("/kaggle/working")
    work.mkdir(exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "gold_macro": gold_macro, "holdout": ho_macro},
        work / "pfoa_syn_b.pt",
    )
    te_ids = tuple(test[ID_COLUMN].astype(str))
    te = predict_ids(model, te_ids, te_root, by_te, device)
    mat = specialist_fill_matrix(len(te_ids), te)
    sub = pd.DataFrame(mat, columns=list(FINDINGS))
    sub.insert(0, ID_COLUMN, list(te_ids))
    sub.to_csv(work / "submission.csv", index=False)
    print(
        f"[pfoa-syn-B] RESEARCH_METHOD={RESEARCH_METHOD} backbone={BACKBONE_NAME} "
        f"img={B_IMG} crop_mm={B_CROP_MM} unfreeze_last={UNFREEZE_LAST} "
        f"slices={B_N_SLICES} not r336_best.pt not mill 6-slot 336/130 "
        f"not frozen GAP-linear | shape={sub.shape} | Baker's ok | "
        f"llm_holdout_macro={ho_macro:.4f} vs mill 0.936 — do not auto-submit",
        flush=True,
    )


if __name__ == "__main__":
    main()
