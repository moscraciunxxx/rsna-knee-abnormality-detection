"""Local v4-hyperparam learn check on real downloaded studies. No Kaggle, no push.

Uses data/train_subset (from collect_train_subset.py) + steven v4 labels.
Builds 96x384 volumes with the same study_lut builder, trains K=4 windows at
160 px on the RTX 3060 (the shape test_learns.py proved fits), v4 optimizer
(head 1e-3, backbone 1e-4, accum 4, clip 1.0, warmup+cosine).

Pass bar: loss falls faster than v3's 0.632 -> 0.577 over 3 epochs and the
probability std stays above 0.05. Prints one line per 25 updates.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kaggle_kernel_nartaa_fold0"))

import numpy as np
import pandas as pd
import torch

from rsna_knee.raptor_model import ARCH, imagenet_norm, sample_train_centers
from rsna_knee.raptor_train import Stepper, make_optimizer, probability_std, weighted_bce
from rsna_knee.schema import FINDINGS, ID_COLUMN

import _fastread

IMG = 384
LOCAL_RES = 160
K = 4
UPDATES = 150
WARMUP = 10
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]

V3_START, V3_END = 0.632, 0.577


def main() -> None:
    import timm

    t0 = time.time()
    subset = os.path.join(ROOT, "data", "train_subset", "train_series")
    v4 = pd.read_csv(
        os.path.join(ROOT, "data", "external_labels", "llm_labels_v4_blend.csv"),
        dtype={ID_COLUMN: str},
    ).set_index(ID_COLUMN)
    ids = pd.read_csv(os.path.join(ROOT, "data", "subset_ids.csv"), dtype=str)[ID_COLUMN].tolist()
    series = pd.read_csv(os.path.join(ROOT, "data", "train_series.csv"), dtype=str)
    ser = {}
    for row in series.to_dict("records"):
        if row["StudyInstanceUID"] in set(ids):
            ser.setdefault(row["StudyInstanceUID"], []).append(row)
    ready = [s for s in ids if os.path.isdir(os.path.join(subset, s))]
    print(f"studies_ready {len(ready)}/{len(ids)}", flush=True)
    if len(ready) < 8:
        raise SystemExit("not enough downloaded studies yet")

    builder = _fastread.make_builder("study_lut", IMG, SLOTS, hdr_threads=8, px_threads=4)
    rng = np.random.default_rng(2026)
    backbone = timm.create_model(
        ARCH, pretrained=True, num_classes=0, in_chans=3, global_pool="avg", img_size=LOCAL_RES
    )
    from rsna_knee.raptor_model import RaptorClassifier

    model = RaptorClassifier(backbone, f_dim=backbone.num_features, n=12).cuda().train()
    stepper = Stepper(make_optimizer(model), torch.amp.GradScaler("cuda"), accum=4)
    stepper.opt.zero_grad(set_to_none=True)
    losses, probs = [], []
    i = 0
    while stepper.updates < UPDATES:
        sid = ready[i % len(ready)]
        i += 1
        vol, mask = builder(sid, ser, subset)
        centers = sample_train_centers(mask, 96, K, rng)
        wins = []
        for c in centers:
            tri = vol[c - 1 : c + 2].astype(np.float32) / 255.0
            wins.append(tri)
        x = torch.from_numpy(np.stack(wins)).unsqueeze(0)
        x = torch.nn.functional.interpolate(
            x.flatten(0, 1), size=(LOCAL_RES, LOCAL_RES), mode="bilinear", align_corners=False
        ).view(1, K, 3, LOCAL_RES, LOCAL_RES)
        x = imagenet_norm(x).cuda()
        y = torch.tensor(v4.loc[sid, FINDINGS].to_numpy(np.float32), device="cuda").view(1, 12)
        w = torch.ones(1, 12, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = weighted_bce(logits, y, w)
        stepper.backward(loss)
        if stepper.ready():
            losses.append(float(loss.detach()))
            stepper.update(WARMUP, UPDATES)
            probs.append(torch.sigmoid(logits.detach()).float().cpu())
            if stepper.updates % 25 == 0:
                print(
                    f"update {stepper.updates} loss {losses[-1]:.4f} "
                    f"std {probability_std(torch.cat(probs[-8:], 0)):.4f}",
                    flush=True,
                )
    first, last = losses[0], losses[-1]
    std = probability_std(torch.cat(probs, 0))
    print(f"first {first:.4f} last {last:.4f} std {std:.4f} minutes {(time.time()-t0)/60:.1f}", flush=True)
    print(f"v3 reference {V3_START} -> {V3_END} over 3 epochs", flush=True)
    assert last <= first * 0.70, "loss did not fall 30%"
    assert std > 0.05, "predictions flat"


if __name__ == "__main__":
    main()
