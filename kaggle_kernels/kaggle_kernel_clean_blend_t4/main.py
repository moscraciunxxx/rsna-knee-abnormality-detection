"""Bare goodpjw2008 ConvNeXt reader, three author folds, T4.

Loads only cnxt_v0_fold{0,1,2}.pt. Same forward as the author.
No E11/E9 heads, no gold58-fit weights, no OAI. Internet off.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"
FORBIDDEN = ("tonylica", "dinosaur", "jev-hybrid", "oai", "widedense", "maxspan", "e11", "e9")


def find_test_root() -> Path:
    for base in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (base / "test.csv").is_file():
            return base
    root = Path("/kaggle/input")
    if root.is_dir():
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in {"train_series", "test_series"}]
            if "test.csv" in files:
                return Path(dirpath)
    raise FileNotFoundError("test.csv is not mounted")


def find_one(name: str) -> Path:
    root = Path("/kaggle/input")
    hits: list[Path] = []
    for dirpath, dirnames, files in os.walk(root):
        low = dirpath.lower()
        if any(token in low for token in FORBIDDEN):
            raise RuntimeError(f"refusing forbidden mount {dirpath}")
        dirnames[:] = [
            d for d in dirnames
            if d not in {"train_series", "test_series", "train_images", "test_images"}
        ]
        if name in files:
            hits.append(Path(dirpath) / name)
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: found {len(hits)}")
    return hits[0]


def main() -> None:
    print("clean_option reader_alone", flush=True)
    comp = find_test_root()
    ckpt0 = find_one("cnxt_v0_fold0.pt")
    assets = ckpt0.parent
    ckpts = sorted(glob.glob(str(assets / "cnxt_v0_fold*.pt")))
    if [Path(p).name for p in ckpts] != ["cnxt_v0_fold0.pt", "cnxt_v0_fold1.pt", "cnxt_v0_fold2.pt"]:
        raise RuntimeError(f"unexpected checkpoints {ckpts}")
    sys.path.insert(0, str(assets))
    from knee import LABELS as KNEE_LABELS
    from knee import N_SLOTS, KneeNet, build_study_table, window_centres
    from preprocess import MAX_SLICES, SIZE, load_series

    if list(KNEE_LABELS) != LABELS:
        raise RuntimeError(f"label mismatch {KNEE_LABELS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    models = []
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        backbone = ck["args"]["backbone"].split(".")[0]
        if backbone != "convnext_tiny":
            raise RuntimeError(f"{path} backbone {backbone}")
        model = KneeNet(backbone, pretrained=False)
        model.load_state_dict(ck["model"])
        models.append((model.to(device).eval(), int(ck["args"]["res"])))
        print("loaded", Path(path).name, "backbone", backbone, "res", ck["args"]["res"], flush=True)

    split = "test"
    studies = pd.read_csv(comp / f"{split}.csv", dtype={ID_COL: str})[ID_COL].tolist()
    se = pd.read_csv(comp / f"{split}_series.csv")
    se = se[se.StudyInstanceUID.isin(set(studies))]
    root = comp / f"{split}_series"
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        folder = root / str(st) / str(sr)
        n_files[sr] = min(len(list(folder.iterdir())), MAX_SLICES) if folder.is_dir() else 0
    table = build_study_table(se, n_files)
    k = 12
    rows = []
    t0 = time.time()
    for n, st in enumerate(studies, start=1):
        slots = table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, k, 3, SIZE, SIZE), np.uint8)
        mask = np.zeros(N_SLOTS, bool)
        pos = np.zeros((N_SLOTS, k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            try:
                vol, _ = load_series(str(root / str(st) / str(cands[0])))
            except Exception:
                continue
            nvol = len(vol)
            if nvol < 3:
                continue
            centres = window_centres(nvol, k, False)
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, nvol - 1)] for j in centres])
            mask[s] = True
            pos[s] = centres / max(nvol - 1, 1)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        mt = torch.from_numpy(mask).unsqueeze(0).to(device)
        pt = torch.from_numpy(pos).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = [torch.sigmoid(model(xt, mt, pt, res=res).float()) for model, res in models]
            prob = torch.stack(preds).mean(0)[0].cpu().numpy()
        rows.append({ID_COL: st, **{name: float(prob[i]) for i, name in enumerate(LABELS)}})
        if n % 25 == 0 or n == len(studies):
            print(f"studies {n}/{len(studies)}", flush=True)
    sec = (time.time() - t0) / max(len(studies), 1)
    print(f"seconds/study {sec:.3f} n {len(studies)}", flush=True)
    frame = pd.DataFrame(rows)[[ID_COL, *LABELS]]
    if frame[LABELS].isna().any().any() or len(frame) != len(studies):
        raise RuntimeError("submission has missing scores")
    out = Path("/kaggle/working/submission.csv")
    frame.to_csv(out, index=False)
    print("wrote", out, "rows", len(frame), "cols", list(frame.columns), flush=True)


if __name__ == "__main__":
    main()
