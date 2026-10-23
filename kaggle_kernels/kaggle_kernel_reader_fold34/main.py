"""Train reader folds 3 and 4. Draft. Do not push until a separate OK.

Same recipe as goodpjw2008 train.py: ConvNeXt-Tiny, res 256, 14 epochs,
batch 8, k=8, AdamW 1e-4 / 5e-4, EMA 0.998. Labels are v0, the unweighted
mean of steven v4, pilkwang v2, lixin sol56, and laymond qwen. Gold58 is
fold -1 and is not in the loss. The checkpoint is the last EMA, not the
best gold epoch. Internet is on only so timm can fetch the ImageNet init.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
FOLDS = (3, 4)
EPOCHS = 14
BATCH = 8
LR = 1e-4
LR_HEAD = 5e-4
WD = 0.02
EMA = 0.998
K_TRAIN = 8
RES = 256
WALL_H = 8.5
SEED = 0


def find_one(name: str) -> Path:
    skip = {"train_series", "test_series", "train_images", "test_images"}
    hits = []
    root = Path("/kaggle/input")
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith("1.2.")]
        if name in files:
            hits.append(Path(dirpath) / name)
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: found {len(hits)}")
    return hits[0]


def competition_root() -> Path:
    for base in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (base / "train.csv").is_file():
            return base
    return find_one("train.csv").parent


def report_group(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def build_v0(train: pd.DataFrame) -> pd.DataFrame:
    ids = train["StudyInstanceUID"].astype(str)
    gold = train.set_index("StudyInstanceUID")[LABELS]
    gold.index = gold.index.astype(str)
    is_gold = gold.notna().all(axis=1)
    steven = pd.read_csv(find_one("llm_labels_v4_blend.csv"), dtype={"StudyInstanceUID": str})
    pilk = pd.read_csv(find_one("report_labels_v2.csv"), dtype={"StudyInstanceUID": str})
    lixin = pd.read_csv(find_one("labels_llm_gpt56sol.csv"), dtype={"StudyInstanceUID": str})
    qwen = pd.read_csv(find_one("qwen_knee_weak_labels.csv"), dtype={"StudyInstanceUID": str})
    qwen = qwen[["StudyInstanceUID", *[f"{c}__label" for c in LABELS]]]
    qwen.columns = ["StudyInstanceUID", *LABELS]
    sources = []
    for frame in (steven, pilk, lixin, qwen):
        frame = frame.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        sources.append(frame[LABELS].astype(float).reindex(ids))
    num = sum(frame.fillna(0) for frame in sources)
    den = sum(frame.notna() for frame in sources)
    soft = (num / den).clip(0.01, 0.99)
    soft.index = ids.to_numpy()
    out = soft.copy()
    out.insert(0, "is_gold", is_gold.reindex(soft.index).astype(int).to_numpy())
    out["fold"] = -1
    weak = out["is_gold"].to_numpy() == 0
    groups = train["Report"].map(report_group).to_numpy()[weak]
    order = np.random.RandomState(SEED).permutation(int(weak.sum()))
    folds = np.zeros(int(weak.sum()), int)
    for k, (_, va) in enumerate(GroupKFold(5).split(order, groups=groups[order])):
        folds[order[va]] = k
    out.loc[weak, "fold"] = folds
    print("fold sizes", out["fold"].value_counts().sort_index().to_dict(), flush=True)
    return out


def main() -> None:
    t0 = time.time()
    comp = competition_root()
    train = pd.read_csv(comp / "train.csv", dtype={"StudyInstanceUID": str})
    labels = build_v0(train)
    code = find_one("knee.py").parent
    sys.path.insert(0, str(code))
    from knee import KneeNet, StudyDataset, build_study_table
    from preprocess import load_series

    series = pd.read_csv(comp / "train_series.csv", dtype=str)
    paths = {
        sr: str(comp / "train_series" / st / sr)
        for st, sr in zip(series.StudyInstanceUID, series.SeriesInstanceUID)
    }

    def loader(sr):
        vol, _ = load_series(paths[sr], 0)
        return vol

    n_slices = {}
    out = Path("/kaggle/working")
    out.mkdir(parents=True, exist_ok=True)
    # Slice counts come from the series folders. Gold rows stay out of every train loader.
    for sr, path in paths.items():
        folder = Path(path)
        n_slices[sr] = len(list(folder.glob("*.dcm"))) if folder.is_dir() else 0
    table = build_study_table(series, n_slices)
    device = torch.device("cuda")
    for fold in FOLDS:
        if (time.time() - t0) > WALL_H * 3600:
            print("wall reached before fold", fold, flush=True)
            return
        weak = labels[labels["is_gold"] == 0]
        tr = weak[weak["fold"] != fold]
        va = weak[weak["fold"] == fold]
        print(f"fold {fold} train {len(tr)} val {len(va)} gold_in_loss 0", flush=True)
        y = tr[LABELS].to_numpy(np.float32)
        train_dl = torch.utils.data.DataLoader(
            StudyDataset(tr.index, table, None, y, K_TRAIN, train=True, loader=loader),
            batch_size=BATCH, shuffle=True, drop_last=True, num_workers=2,
        )
        model = KneeNet("convnext_tiny", pretrained=True).to(device)
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        enc = list(model.enc.parameters())
        head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
        opt = torch.optim.AdamW(
            [{"params": enc, "lr": LR}, {"params": head, "lr": LR_HEAD}],
            weight_decay=WD,
        )
        steps = EPOCHS * max(len(train_dl), 1)
        warm = max(len(train_dl), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(steps, 1))))),
        )
        step = 0
        for ep in range(EPOCHS):
            if (time.time() - t0) > WALL_H * 3600:
                print("wall reached during fold", fold, "epoch", ep, flush=True)
                torch.save({"model": ema.state_dict(), "args": {"fold": fold, "res": RES}}, out / f"fold{fold}_partial.pt")
                return
            model.train()
            total, nstep = 0.0, 0
            t_ep = time.time()
            for x, mask, pos, yb in train_dl:
                x, mask, pos, yb = x.to(device), mask.to(device), pos.to(device), yb.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(x, mask, pos, res=RES, train=True)
                loss = F.binary_cross_entropy_with_logits(logit.float(), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                opt.step()
                sched.step()
                step += 1
                decay = min(EMA, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - decay)
                total += float(loss.detach())
                nstep += 1
            print(f"fold {fold} ep {ep} loss {total / max(nstep, 1):.4f} sec {time.time() - t_ep:.0f} val_rows {len(va)}", flush=True)
        torch.save({"model": ema.state_dict(), "args": {"fold": fold, "backbone": "convnext_tiny", "res": RES, "labels": "v0"}}, out / f"fold{fold}.pt")
        print("saved", out / f"fold{fold}.pt", flush=True)


if __name__ == "__main__":
    main()
