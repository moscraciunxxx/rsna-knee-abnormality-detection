"""Holdout870 variants. No submit, no push.

Reader flip TTA uses the author's infer settings (k=12, three folds, fp16)
on the local reader cache. Widedense uses its own 64-slice slot layout and
42 windows, built from that same cache: 140 mm centre crop of the 153.6 mm
field, then 336 to 384. Rank blends use fixed weights only.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

ROOT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts" / "provenance" / "reader_weights"))

from knee import LABELS, N_SLOTS, KneeNet, build_study_table, window_centres
from rsna_knee.metrics import macro_roc_auc, per_finding_roc_auc
from rsna_knee.mill_holdout import holdout_ids_and_y
from rsna_knee.schema import FINDINGS, ID_COLUMN

CACHE = Path(r"D:\rsna_cache")
WEIGHTS = ROOT / "artifacts" / "provenance" / "reader_weights"
WIDE_PT = ROOT / "artifacts" / "provenance" / "widedense" / "raptor_ft_coatnet_v4_full.pt"
READER_CSV = ROOT / "artifacts" / "convnext_mil870" / "convnext_holdout870.csv"
OUT = ROOT / "artifacts" / "clean_blend"
LOG = CACHE / "holdout_variants.log"

SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = 64
CROP = 350  # 140 mm at the cache's 0.4 mm pixels
IMG336 = 336
K_WIDE = 42
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def cache_index() -> dict[str, Path]:
    found = {}
    for fold in range(5):
        for path in (CACHE / f"f{fold}").glob("*.npz"):
            found[path.stem] = path
    return found


def load_study(index: dict[str, Path], study: str) -> dict[str, np.ndarray]:
    with np.load(index[study]) as blob:
        return {name: blob[name] for name in blob.files}


def reader_models(device):
    models = []
    for fold in range(3):
        ck = torch.load(WEIGHTS / f"cnxt_v0_fold{fold}.pt", map_location="cpu", weights_only=False)
        model = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
        model.load_state_dict(ck["model"])
        models.append((model.to(device).eval(), int(ck["args"]["res"])))
    return models


def reader_item(vols: dict[str, np.ndarray], slots, k: int, flip: bool):
    x = np.zeros((N_SLOTS, k, 3, 384, 384), np.uint8)
    mask = np.zeros(N_SLOTS, bool)
    pos = np.zeros((N_SLOTS, k), np.float32)
    for s, cands in enumerate(slots):
        if not cands or cands[0] not in vols:
            continue
        vol = vols[cands[0]]
        if flip:
            vol = np.ascontiguousarray(vol[:, :, ::-1])
        n = len(vol)
        if n < 3:
            continue
        centres = window_centres(n, k, False)
        x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in centres])
        mask[s] = True
        pos[s] = centres / max(n - 1, 1)
    return x, mask, pos


@torch.no_grad()
def reader_probs(models, x, mask, pos, device) -> np.ndarray:
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    mt = torch.from_numpy(mask).unsqueeze(0).to(device)
    pt = torch.from_numpy(pos).unsqueeze(0).to(device)
    preds = []
    with torch.autocast("cuda", dtype=torch.float16):
        for model, res in models:
            preds.append(torch.sigmoid(model(xt, mt, pt, res=res).float()))
    return torch.stack(preds).mean(0)[0].cpu().numpy()


def crop336(vol: np.ndarray) -> np.ndarray:
    y0 = (vol.shape[1] - CROP) // 2
    x0 = (vol.shape[2] - CROP) // 2
    crop = vol[:, y0:y0 + CROP, x0:x0 + CROP]
    out = np.empty((len(crop), IMG336, IMG336), np.uint8)
    for i, sl in enumerate(crop):
        out[i] = cv2.resize(sl, (IMG336, IMG336), interpolation=cv2.INTER_AREA)
    return out


def pick_slot(rows, plane, fluid, used):
    cands = [row for row in rows if row["Anatomical_Plane"] == plane and row["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [row for row in cands if int(row.get("Fluid_Sensitive") or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def wide_volume(vols: dict[str, np.ndarray], rows) -> tuple[np.ndarray, np.ndarray]:
    vol = np.zeros((MAXS, IMG336, IMG336), np.uint8)
    idx = 0
    used = set()
    for plane, fluid, k in SLOTS:
        row = pick_slot(rows, plane, fluid, used)
        if row is None or row["SeriesInstanceUID"] not in vols:
            idx += k
            continue
        used.add(row["SeriesInstanceUID"])
        series = vols[row["SeriesInstanceUID"]]
        n = len(series)
        lo, hi = int(n * 0.06), max(int(n * 0.94) - 1, int(n * 0.06))
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, int)
        chosen = series[np.clip(picks, 0, n - 1)]
        resized = crop336(chosen)
        take = min(k, MAXS - idx)
        vol[idx:idx + take] = resized[:take]
        idx += k
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def wide_windows(vol, mask, res=384) -> torch.Tensor:
    depth = vol.shape[0]
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    centres = [c for c in range(lo + 1, hi) if lo <= c - 1 and c + 1 <= hi]
    if not centres:
        centres = [max(1, min((lo + hi) // 2, depth - 2))]
    idx = np.linspace(0, len(centres) - 1, K_WIDE).round().astype(int)
    chosen = [centres[int(i)] for i in idx]
    wins = np.empty((len(chosen), 3, res, res), np.float32)
    for j, c in enumerate(chosen):
        c = max(1, min(c, depth - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(tri)
        if tensor.shape[-1] != res:
            tensor = F.interpolate(tensor[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        wins[j] = tensor.numpy()
    x = torch.from_numpy(wins)
    return (x - MEAN) / STD


class WideNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        dim = backbone.num_features
        self.backbone = backbone
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, dim))
        self.clsb = nn.Parameter(torch.zeros(12))

    def forward(self, x):
        batch, k = x.shape[:2]
        flat = x.flatten(0, 1)
        parts = [self.backbone(chunk) for chunk in flat.split(2)]
        feats = torch.cat(parts, 0).view(batch, k, -1)
        hidden = self.norm(feats)
        attn = torch.softmax(self.att(hidden), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attn, hidden)
        return (pooled * self.clsW).sum(-1) + self.clsb


def load_wide(device):
    import timm
    ck = torch.load(WIDE_PT, map_location="cpu", weights_only=False)
    backbone = timm.create_model(ck["arch"], pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    model = WideNet(backbone)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), list(ck["lab"])


@torch.no_grad()
def wide_probs(model, windows, device) -> np.ndarray:
    x = windows.unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        return torch.sigmoid(model(x).float())[0].cpu().numpy()


def ranks(scores: np.ndarray) -> np.ndarray:
    out = np.empty_like(scores, dtype=np.float64)
    for j in range(scores.shape[1]):
        out[:, j] = rankdata(scores[:, j], method="average")
    return out


def row(name: str, y, scores) -> dict:
    per = per_finding_roc_auc(y, scores)
    macro = macro_roc_auc(y, scores)
    out = {"model": name, "macro": macro}
    out.update({FINDINGS[i]: float(per[i]) for i in range(12)})
    log(f"{name} macro {macro:.4f}")
    return out


def main() -> None:
    log("holdout variants start")
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={ID_COLUMN: str})
    v4 = pd.read_csv(ROOT / "data" / "external_labels" / "llm_labels_v4_blend.csv", dtype={ID_COLUMN: str})
    ids, y = holdout_ids_and_y(train, v4)
    log(f"holdout {len(ids)}")
    reader_file = pd.read_csv(READER_CSV, dtype={ID_COLUMN: str})
    reader_file[ID_COLUMN] = reader_file[ID_COLUMN].astype(str)
    reader_file = reader_file.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN).loc[list(ids), LABELS]
    reader_base = reader_file.to_numpy(np.float64)
    log(f"reader file macro {macro_roc_auc(y, reader_base):.4f}")

    series = pd.read_csv(ROOT / "data" / "train_series.csv", dtype=str)
    series = series[series.StudyInstanceUID.isin(ids)]
    by_study = {sid: frame.to_dict("records") for sid, frame in series.groupby("StudyInstanceUID", sort=False)}
    index = cache_index()
    missing = [sid for sid in ids if sid not in index]
    if missing:
        raise SystemExit(f"cache missing {len(missing)}")

    # Slice counts for the reader table, from the holdout npz headers only.
    counts = {}
    for i, sid in enumerate(ids, start=1):
        vols = load_study(index, sid)
        for name, vol in vols.items():
            counts[name] = len(vol)
        if i % 200 == 0:
            log(f"counted {i}/{len(ids)}")
    table = build_study_table(series, counts)
    device = torch.device("cuda")
    log(torch.cuda.get_device_name(0))

    models = reader_models(device)
    off, on = [], []
    t0 = time.time()
    for i, sid in enumerate(ids, start=1):
        slots = table.get(sid, [[] for _ in range(N_SLOTS)])
        vols = load_study(index, sid)
        x, mask, pos = reader_item(vols, slots, 12, False)
        off.append(reader_probs(models, x, mask, pos, device))
        x, mask, pos = reader_item(vols, slots, 12, True)
        on.append(reader_probs(models, x, mask, pos, device))
        if i % 50 == 0:
            log(f"reader {i}/{len(ids)} s/study {(time.time() - t0) / i:.3f}")
    reader_off = np.stack(off)
    reader_on = np.stack(on)
    reader_tta = 0.5 * (reader_off + reader_on)
    log(f"reader cache vs file max_abs {np.max(np.abs(reader_off - reader_base)):.5f}")
    log(f"reader s/study both passes {(time.time() - t0) / len(ids):.3f}")

    wide, _lab = load_wide(device)
    wide_out = []
    t1 = time.time()
    for i, sid in enumerate(ids, start=1):
        vol, mask = wide_volume(load_study(index, sid), by_study.get(sid, []))
        wide_out.append(wide_probs(wide, wide_windows(vol, mask), device))
        if i % 25 == 0:
            log(f"widedense {i}/{len(ids)} s/study {(time.time() - t1) / i:.3f}")
    wide_scores = np.stack(wide_out)
    log(f"widedense s/study {(time.time() - t1) / len(ids):.3f}")

    rows = [
        row("reader_file", y, reader_base),
        row("reader_cache", y, reader_off),
        row("reader_flip", y, reader_on),
        row("reader_flip_tta", y, reader_tta),
        row("widedense", y, wide_scores),
    ]
    rr, ww = ranks(reader_base), ranks(wide_scores)
    for wr, wwgt in ((1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5)):
        rows.append(row(f"rank_{wr:.1f}_{wwgt:.1f}", y, wr * rr + wwgt * ww))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(wide_scores, index=pd.Index(list(ids), name=ID_COLUMN), columns=LABELS).to_csv(OUT / "widedense_holdout870.csv")
    pd.DataFrame(reader_tta, index=pd.Index(list(ids), name=ID_COLUMN), columns=LABELS).to_csv(OUT / "reader_flip_tta_holdout870.csv")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "holdout870_variants.csv", index=False)
    log(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = max(rows, key=lambda item: item["macro"])
    log(f"BEST {best['model']} {best['macro']:.4f} delta {best['macro'] - 0.9439456543904313:+.4f}")


if __name__ == "__main__":
    main()
