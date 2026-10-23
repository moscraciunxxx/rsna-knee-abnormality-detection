"""Local efficiency student. No submit, no push, no selection change.

Teacher is the 3-fold reader on the local cache. The student is ConvNeXt-nano
at 224 px with 24 slices from the same series. Gold58 is not in the loss.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from torch.utils.checkpoint import checkpoint

ROOT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "artifacts" / "provenance" / "reader_weights"))

from knee import LABELS, N_SLOTS, build_study_table
from rsna_knee.metrics import macro_roc_auc
from rsna_knee.schema import ID_COLUMN
from score_holdout_variants import cache_index, load_study, reader_item, reader_models, reader_probs

CACHE = Path(r"D:\rsna_cache")
OUT = ROOT / "artifacts" / "efficiency_student"
TEACHER_CSV = OUT / "teacher_reader.csv"
LOG = CACHE / "student.log"
HOLD = ROOT / "kaggle_kernel_raptor_holdout" / "holdout870_y.csv"
V4 = ROOT / "data" / "external_labels" / "llm_labels_v4_blend.csv"
GOLD = ROOT / "data" / "gold58_ids.csv"
SERIES = ROOT / "data" / "train_series.csv"
K_SLICES = 24
RES = 224
PLAN_EPOCHS = 4
ACCUM = 4
WALL_LIMIT_H = 20.0


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class Student(torch.nn.Module):
    def __init__(self, name: str, chunk: int = 8):
        super().__init__()
        self.enc = timm.create_model(name, pretrained=True, num_classes=0)
        self.head = torch.nn.Linear(self.enc.num_features, 12)
        self.chunk = chunk

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for chunk in x.split(self.chunk):
            if self.training:
                parts.append(checkpoint(self.enc, chunk, use_reentrant=False))
            else:
                parts.append(self.enc(chunk))
        return torch.cat(parts, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        return self.head(self.encode(x).mean(0))


def slices_of(vol: np.ndarray, k: int, res: int = RES) -> np.ndarray:
    n = len(vol)
    idx = np.linspace(0, n - 1, k).round().astype(int) if n > 1 else np.zeros(k, int)
    picked = vol[np.clip(idx, 0, n - 1)]
    out = np.empty((k, res, res), np.uint8)
    for i, image in enumerate(picked):
        out[i] = cv2.resize(image, (res, res), interpolation=cv2.INTER_AREA)
    return np.repeat(out[:, None, :, :], 3, axis=1)


def study_batch(vols: dict[str, np.ndarray], slots, k: int, res: int = RES) -> np.ndarray | None:
    parts = []
    for cands in slots:
        if not cands or cands[0] not in vols or len(vols[cands[0]]) < 1:
            continue
        parts.append(slices_of(vols[cands[0]], k, res))
    if not parts:
        return None
    return np.concatenate(parts, 0)


def ids_and_labels():
    hold = pd.read_csv(HOLD, dtype={ID_COLUMN: str})
    hold_ids = set(hold[ID_COLUMN].astype(str))
    y_hold = hold.set_index(ID_COLUMN).loc[list(hold[ID_COLUMN]), LABELS].to_numpy(np.float64)
    gold = set(pd.read_csv(GOLD, dtype={ID_COLUMN: str})[ID_COLUMN].astype(str))
    v4 = pd.read_csv(V4, dtype={ID_COLUMN: str}).drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    soft = v4[LABELS].astype(float).clip(0.01, 0.99)
    return hold[ID_COLUMN].astype(str).tolist(), y_hold, hold_ids, gold, soft


def build_table(index: dict[str, Path], series: pd.DataFrame):
    counts = {}
    ids = list(index)
    for i, sid in enumerate(ids, start=1):
        for name, vol in load_study(index, sid).items():
            counts[name] = len(vol)
        if i % 400 == 0:
            log(f"counted {i}/{len(ids)}")
    return build_study_table(series, counts)


def write_teacher(index, table, device) -> pd.DataFrame:
    if TEACHER_CSV.is_file():
        frame = pd.read_csv(TEACHER_CSV, dtype={ID_COLUMN: str})
        if len(frame) == len(index):
            log(f"teacher already saved {len(frame)}")
            return frame.set_index(ID_COLUMN)
    models = reader_models(device)
    rows = []
    ids = list(index)
    t0 = time.time()
    for i, sid in enumerate(ids, start=1):
        vols = load_study(index, sid)
        x, mask, pos = reader_item(vols, table.get(sid, [[] for _ in range(N_SLOTS)]), 12, False)
        rows.append(reader_probs(models, x, mask, pos, device))
        if i % 200 == 0:
            log(f"teacher {i}/{len(ids)} s/study {(time.time() - t0) / i:.3f}")
    elapsed = time.time() - t0
    log(f"teacher done n {len(ids)} seconds {elapsed:.0f} s/study {elapsed / len(ids):.3f}")
    frame = pd.DataFrame(np.stack(rows), index=pd.Index(ids, name=ID_COLUMN), columns=LABELS)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TEACHER_CSV)
    return frame


def fit_chunk(model: Student, device) -> int:
    dummy = torch.zeros(1, 3, RES, RES, device=device)
    for chunk in (16, 8, 4, 2):
        torch.cuda.empty_cache()
        try:
            model.train()
            x = dummy.repeat(chunk, 1, 1, 1)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model.encode(x).sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            log(f"chunk {chunk} fits on {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
            return chunk
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            log(f"chunk {chunk} oom")
    raise torch.cuda.OutOfMemoryError("student does not fit")


@torch.no_grad()
def infer_ids(model, ids, index, table, device, k: int, res: int = RES) -> np.ndarray:
    model.eval()
    out = []
    for sid in ids:
        batch = study_batch(load_study(index, sid), table.get(sid, []), k, res)
        if batch is None:
            out.append(np.full(12, 0.5, np.float32))
            continue
        x = torch.from_numpy(batch).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    return np.stack(out)


def time_infer(model, ids, index, table, device, k: int) -> float:
    sample = ids[:20]
    torch.cuda.synchronize()
    t0 = time.time()
    infer_ids(model, sample, index, table, device, k)
    torch.cuda.synchronize()
    return (time.time() - t0) / len(sample)


def train(model, name, train_ids, teacher, soft, hold_ids, y_hold, index, table, device, k: int):
    chunk = fit_chunk(model, device)
    model.chunk = chunk
    pace = time_infer(model, hold_ids, index, table, device, k)
    log(f"{name} infer_s/study {pace:.3f}")
    if name == "convnext_nano" and pace > 1.5:
        log("nano slower than 1.5 s/study, switching to convnext_atto")
        return None
    enc = list(model.enc.parameters())
    head = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": 1e-4}, {"params": head, "lr": 5e-4}],
        weight_decay=0.05,
    )
    steps_per_epoch = max(len(train_ids) // ACCUM, 1)
    total_steps = PLAN_EPOCHS * steps_per_epoch
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(total_steps, 1)))),
    )
    scaler = torch.amp.GradScaler("cuda")
    order = np.array(train_ids)
    epochs = PLAN_EPOCHS
    history = []
    opt_step = 0
    for epoch in range(epochs):
        rng = np.random.default_rng(epoch)
        rng.shuffle(order)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        step_times = []
        pending = 0
        seen = 0
        for sid in order:
            if sid not in teacher.index or sid not in soft.index:
                continue
            batch = study_batch(load_study(index, sid), table.get(sid, []), k)
            if batch is None:
                continue
            x = torch.from_numpy(batch).to(device)
            report = torch.tensor(soft.loc[sid, LABELS].to_numpy(np.float32), device=device)
            teach = torch.tensor(teacher.loc[sid, LABELS].to_numpy(np.float32), device=device)
            if pending == 0:
                step_t = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x)
                loss = 0.5 * F.binary_cross_entropy_with_logits(logit, report)
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(logit, teach)
                loss = loss / ACCUM
            scaler.scale(loss).backward()
            pending += 1
            seen += 1
            if pending == ACCUM:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                opt_step += 1
                pending = 0
                step_times.append(time.time() - step_t)
                if opt_step % 50 == 0:
                    log(f"ep {epoch} step {opt_step} s/iter {np.mean(step_times):.3f}")
        probs = infer_ids(model, hold_ids, index, table, device, k)
        std = float(probs.std())
        score = macro_roc_auc(y_hold, probs)
        s_iter = float(np.mean(step_times)) if step_times else float("nan")
        epoch_h = (time.time() - t0) / 3600
        log(f"ep {epoch} holdout {score:.4f} std {std:.4f} s/iter {s_iter:.3f} epoch_h {epoch_h:.2f}")
        history.append({"epoch": epoch, "macro": score, "std": std, "s_iter": s_iter, "epoch_h": epoch_h})
        torch.save({"model": model.state_dict(), "name": name, "k": k, "epoch": epoch}, OUT / "student.pt")
        if std < 0.02:
            log(f"COLLAPSE std {std:.4f}")
            break
        if epoch == 0:
            projected = epoch_h * epochs
            log(f"projected_total_hours {projected:.2f} for {epochs} epochs")
            if projected > WALL_LIMIT_H:
                fit = max(1, int(WALL_LIMIT_H // max(epoch_h, 1e-6)))
                log(f"SHRINK epochs {epochs} -> {fit} because projected {projected:.2f} h")
                epochs = fit
                if epochs == 1:
                    break
    torch.cuda.synchronize()
    t1 = time.time()
    infer_ids(model, hold_ids, index, table, device, k)
    torch.cuda.synchronize()
    s_study = (time.time() - t1) / len(hold_ids)
    return {"name": name, "k": k, "history": history, "s_study": s_study, "epochs_ran": len(history)}


def main() -> None:
    log(f"efficiency student start gpu {torch.cuda.get_device_name(0)} gb {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}")
    device = torch.device("cuda")
    index = cache_index()
    series = pd.read_csv(SERIES, dtype=str)
    table = build_table(index, series)
    teacher = write_teacher(index, table, device)
    hold_ids, y_hold, hold_set, gold, soft = ids_and_labels()
    train_ids = [sid for sid in index if sid not in hold_set and sid not in gold and sid in soft.index and sid in teacher.index]
    leaked = set(train_ids) & gold
    if leaked:
        raise RuntimeError(f"gold in train {len(leaked)}")
    log(f"train {len(train_ids)} holdout {len(hold_ids)} gold_excluded {len(gold & set(index))}")
    OUT.mkdir(parents=True, exist_ok=True)
    model = Student("convnext_nano").to(device)
    result = train(model, "convnext_nano", train_ids, teacher, soft, hold_ids, y_hold, index, table, device, K_SLICES)
    if result is None:
        del model
        torch.cuda.empty_cache()
        model = Student("convnext_atto").to(device)
        result = train(model, "convnext_atto", train_ids, teacher, soft, hold_ids, y_hold, index, table, device, K_SLICES)
    result["reader_holdout"] = 0.9439
    result["reader_t4_s"] = 16.2
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    last = result["history"][-1]
    log(
        f"DONE {result['name']} last_macro {last['macro']:.4f} s/study {result['s_study']:.3f} "
        f"vs reader 0.9439 and 16.2 s"
    )


if __name__ == "__main__":
    main()
