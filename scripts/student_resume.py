"""Resume the epoch-4 ConvNeXt-nano for 10 epochs, then a 288/32 run if epochs stay under 1.5 h.

No submit, no push. Gold58 stays out of the loss. Collapse alarm stays on.
"""

from __future__ import annotations

import json
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F

from efficiency_student import (
    ACCUM,
    CACHE,
    LABELS,
    OUT,
    SERIES,
    Student,
    build_table,
    cache_index,
    ids_and_labels,
    infer_ids,
    load_study,
    log,
    study_batch,
)
import pandas as pd

CKPT = OUT / "student.pt"
MORE = 10
PEAK_ENC = 3e-5
PEAK_HEAD = 1e-4
EPOCH_CAP_H = 1.5


def fit_chunk(model: Student, device, res: int) -> int:
    model.train()
    for chunk in (8, 4, 2):
        torch.cuda.empty_cache()
        try:
            x = torch.zeros(chunk, 3, res, res, device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model.encode(x).sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            log(f"chunk {chunk} fits at {res}")
            return chunk
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise torch.cuda.OutOfMemoryError(f"no chunk fits at {res}")


def run_epochs(model, tag, train_ids, teacher, soft, hold_ids, y_hold, index, table, device, k, res, epochs):
    model.chunk = fit_chunk(model, device, res)
    enc = list(model.enc.parameters())
    head = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": PEAK_ENC}, {"params": head, "lr": PEAK_HEAD}],
        weight_decay=0.05,
    )
    steps = max(len(train_ids) // ACCUM, 1)
    total = epochs * steps
    warm = max(steps // 5, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(total, 1)))),
    )
    scaler = torch.amp.GradScaler("cuda")
    order = np.array(train_ids)
    history = []
    for epoch in range(epochs):
        rng = np.random.default_rng(100 + epoch)
        rng.shuffle(order)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        pending = 0
        for sid in order:
            if sid not in teacher.index or sid not in soft.index:
                continue
            batch = study_batch(load_study(index, sid), table.get(sid, []), k, res)
            if batch is None:
                continue
            x = torch.from_numpy(batch).to(device)
            report = torch.tensor(soft.loc[sid, LABELS].to_numpy(np.float32), device=device)
            teach = torch.tensor(teacher.loc[sid, LABELS].to_numpy(np.float32), device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x)
                loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(logit, report)
                    + F.binary_cross_entropy_with_logits(logit, teach)
                ) / ACCUM
            scaler.scale(loss).backward()
            pending += 1
            if pending == ACCUM:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                pending = 0
        t_inf = time.time()
        probs = infer_ids(model, hold_ids, index, table, device, k, res)
        s_study = (time.time() - t_inf) / len(hold_ids)
        std = float(probs.std())
        from rsna_knee.metrics import macro_roc_auc
        score = macro_roc_auc(y_hold, probs)
        epoch_h = (time.time() - t0) / 3600
        row = {"tag": tag, "epoch": epoch, "macro": score, "std": std, "s_study": s_study, "epoch_h": epoch_h}
        history.append(row)
        log(f"{tag} ep {epoch} holdout {score:.4f} std {std:.4f} s/study {s_study:.3f} epoch_h {epoch_h:.2f}")
        torch.save({"model": model.state_dict(), "name": "convnext_nano", "k": k, "res": res, "epoch": epoch, "tag": tag}, OUT / f"student_{tag}.pt")
        if std < 0.02:
            log(f"COLLAPSE {tag} std {std:.4f}")
            break
        if epoch_h >= EPOCH_CAP_H and tag != "resume224":
            log(f"STOP {tag} epoch {epoch_h:.2f} h >= {EPOCH_CAP_H}")
            break
    return history


def main() -> None:
    device = torch.device("cuda")
    log(f"resume start {torch.cuda.get_device_name(0)}")
    index = cache_index()
    table = build_table(index, pd.read_csv(SERIES, dtype=str))
    teacher = pd.read_csv(OUT / "teacher_reader.csv", dtype={"StudyInstanceUID": "str"}).set_index("StudyInstanceUID")
    hold_ids, y_hold, hold_set, gold, soft = ids_and_labels()
    train_ids = [sid for sid in index if sid not in hold_set and sid not in gold and sid in soft.index and sid in teacher.index]
    if set(train_ids) & gold:
        raise RuntimeError("gold leaked into train")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    log(f"loaded epoch {ck.get('epoch')} k {ck.get('k')}")
    model = Student("convnext_nano")
    model.enc = __import__("timm").create_model("convnext_nano", pretrained=False, num_classes=0)
    model.head = torch.nn.Linear(model.enc.num_features, 12)
    model.load_state_dict(ck["model"])
    model.to(device)
    hist = run_epochs(model, "resume224", train_ids, teacher, soft, hold_ids, y_hold, index, table, device, 24, 224, MORE)
    slow = any(row["epoch_h"] >= EPOCH_CAP_H for row in hist)
    second = []
    if hist and not slow:
        log("epoch under 1.5 h, starting 288 px / 32 slices")
        second = run_epochs(model, "run288", train_ids, teacher, soft, hold_ids, y_hold, index, table, device, 32, 288, MORE)
    else:
        log("skip 288 run")
    payload = {"resume224": hist, "run288": second}
    (OUT / "resume_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("resume done")


if __name__ == "__main__":
    main()
