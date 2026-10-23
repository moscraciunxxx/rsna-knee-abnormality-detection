"""288 px / 32-slice student. Starts only after the 224 resume has saved epoch 9.

DataLoader workers prefetch studies. The image microbatch is the largest chunk
that fits this GPU. The 3060 is 6 GB, so a 9-10 GB target cannot be met.
"""

from __future__ import annotations

import os
import subprocess
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from efficiency_student import (
    ACCUM,
    LABELS,
    OUT,
    SERIES,
    Student,
    build_table,
    cache_index,
    ids_and_labels,
    infer_ids,
    log,
    study_batch,
)

RES = 288
K = 32
EPOCHS = 10
PEAK_ENC = 3e-5
PEAK_HEAD = 1e-4


def unwrap_batch(batch):
    return batch[0]


class StudySet(Dataset):
    def __init__(self, ids, index, table, teacher, soft):
        self.ids = ids
        self.index = index
        self.table = table
        self.teacher = teacher
        self.soft = soft

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        sid = self.ids[item]
        from efficiency_student import load_study

        batch = study_batch(load_study(self.index, sid), self.table.get(sid, []), K, RES)
        if batch is None:
            batch = np.zeros((1, 3, RES, RES), np.uint8)
        report = self.soft.loc[sid, LABELS].to_numpy(np.float32)
        teach = self.teacher.loc[sid, LABELS].to_numpy(np.float32)
        return torch.from_numpy(batch), torch.from_numpy(report), torch.from_numpy(teach)


def gpu_now():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    util, used, total = [float(part) for part in text.split(",")]
    return util, used / 1024, total / 1024


def largest_chunk(model, device) -> int:
    """Grow the image microbatch until the forward/backward no longer fits."""
    model.train()
    best = 1
    best_gb = 0.0
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    cap = min(9.5, total_gb * 0.90)
    log(f"vram total {total_gb:.2f} GB target {cap:.2f} GB")
    for chunk in (4, 8, 12, 16, 24, 32, 48, 64, 96):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model.chunk = chunk
            x = torch.zeros(chunk, 3, RES, RES, device=device)
            y = torch.full((12,), 0.5, device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            used = torch.cuda.max_memory_allocated() / 2**30
            log(f"probe chunk {chunk} allocated {used:.2f} GB")
            if used > 5.3:
                log("chunk fills the 6 GB card, keeping the previous microbatch")
                break
            best, best_gb = chunk, used
            if used >= cap:
                break
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            log(f"probe chunk {chunk} oom")
            break
    log(f"microbatch {best} allocated {best_gb:.2f} GB")
    return best


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    log(f"288 start {torch.cuda.get_device_name(0)}")
    index = cache_index()
    table = build_table(index, pd.read_csv(SERIES, dtype=str))
    teacher = pd.read_csv(OUT / "teacher_reader.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    hold_ids, y_hold, hold_set, gold, soft = ids_and_labels()
    train_ids = [sid for sid in index if sid not in hold_set and sid not in gold and sid in soft.index and sid in teacher.index]
    if set(train_ids) & gold:
        raise RuntimeError("gold leaked into train")
    ck = torch.load(OUT / "student_resume224.pt", map_location="cpu", weights_only=False)
    log(f"loaded resume224 epoch {ck.get('epoch')}")
    import timm

    model = Student("convnext_nano")
    model.enc = timm.create_model("convnext_nano", pretrained=False, num_classes=0)
    model.head = torch.nn.Linear(model.enc.num_features, 12)
    model.load_state_dict(ck["model"])
    model.to(device)
    model.chunk = largest_chunk(model, device)
    ds = StudySet(train_ids, index, table, teacher, soft)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=unwrap_batch,
    )
    opt = torch.optim.AdamW(
        [{"params": model.enc.parameters(), "lr": PEAK_ENC}, {"params": model.head.parameters(), "lr": PEAK_HEAD}],
        weight_decay=0.05,
    )
    steps = max(len(train_ids) // ACCUM, 1)
    total = EPOCHS * steps
    warm = max(steps // 5, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(total, 1)))),
    )
    scaler = torch.amp.GradScaler("cuda")
    opt_step = 0
    pending = 0
    step_t = time.time()
    step_times = []
    utils = []
    model.train()
    for epoch in range(EPOCHS):
        t0 = time.time()
        for images, report, teach in loader:
            images = images.to(device, non_blocking=True)
            report = report.to(device, non_blocking=True)
            teach = teach.to(device, non_blocking=True)
            if pending == 0:
                step_t = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(images)
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
                opt_step += 1
                step_times.append(time.time() - step_t)
                util, used, _total = gpu_now()
                utils.append(util)
                if opt_step == 50:
                    log(
                        f"iter50 s/iter {np.mean(step_times):.3f} gpu_util {np.mean(utils):.0f}% "
                        f"vram_used {used:.2f} GB microbatch {model.chunk}"
                    )
        from rsna_knee.metrics import macro_roc_auc

        t_inf = time.time()
        probs = infer_ids(model, hold_ids, index, table, device, K, RES)
        s_study = (time.time() - t_inf) / len(hold_ids)
        std = float(probs.std())
        score = macro_roc_auc(y_hold, probs)
        log(f"run288 ep {epoch} holdout {score:.4f} std {std:.4f} s/study {s_study:.3f} epoch_h {(time.time() - t0) / 3600:.2f}")
        torch.save({"model": model.state_dict(), "k": K, "res": RES, "epoch": epoch, "chunk": model.chunk}, OUT / "student_run288.pt")
        if std < 0.02:
            log(f"COLLAPSE run288 std {std:.4f}")
            break
        if (time.time() - t0) / 3600 >= 1.5:
            log("STOP run288 epoch >= 1.5 h")
            break
    log("288 done")


if __name__ == "__main__":
    main()
