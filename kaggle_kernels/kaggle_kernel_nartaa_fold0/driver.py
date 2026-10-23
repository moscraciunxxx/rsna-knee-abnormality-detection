from __future__ import annotations

"""No-OAI fold-0 CoAtNet train, v5. Not a submission kernel.

Resumes the v3 checkpoint (epoch 3, OOF 0.72). Does not start from ImageNet.
Same 96-slice slots and 140 mm crop as the public 0.949 notebook. Gold studies
stay out of the loss. Internet is unused for weights: the run aborts if
fold0_last.pt does not load strict.
"""

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raptor_model import (  # noqa: E402
    ARCH,
    IMG,
    K_EVAL,
    N_SLICES,
    RaptorClassifier,
    epoch_seconds,
    eval_centers,
    imagenet_norm,
    sample_train_centers,
    windows_from_volume,
)
import _fastread  # noqa: E402

FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
FOLD = 0
K_TRAIN = 8
TRAIN_RES = 384
EPOCHS = 10
SWA_EPOCHS = {7, 8, 9, 10}
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]
HEAD_LR = 3e-4
BACKBONE_LR = 3e-5
WEIGHT_DECAY = 0.05
ACCUM = 4
WARMUP_STEPS = 500
COLLAPSE_STD = 1e-3
WALL_HOURS = 4.0
OOF_SUBSET_N = 200
OOF_KILL_RESUMED_EPOCH = 2
OOF_KILL_FLOOR = 0.78
V3_EPOCH = 3
V3_FIRST_MB = "eb0e7aa36def97fb"
SEED = 2026


def first_mb_sha16(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read(1024 * 1024)).hexdigest()[:16]


def require_v3_hash(path: Path) -> str:
    if not path.is_file():
        print(f"ABORT resume: missing {path}", flush=True)
        sys.exit(1)
    digest = first_mb_sha16(path)
    print(f"fold0_last.pt first_mb_sha256 {digest}", flush=True)
    if digest != V3_FIRST_MB:
        print(f"ABORT resume: first_mb_sha256 {digest} is not v3 {V3_FIRST_MB}", flush=True)
        sys.exit(1)
    return digest


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per * weight).sum() / weight.sum().clamp_min(1e-6)


def make_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    backbone = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    head = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    opt = torch.optim.AdamW(
        [
            {"params": backbone, "lr": BACKBONE_LR},
            {"params": head, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt


def lr_multiplier(step: int, warmup: int, total: int) -> float:
    """Linear warmup, then cosine down to 0. ``step`` is 1-based after an update."""
    if total < 1:
        return 1.0
    if step <= warmup:
        return float(step) / float(max(warmup, 1))
    progress = (step - warmup) / float(max(total - warmup, 1))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lrs(opt: torch.optim.Optimizer, step: int, warmup: int, total: int) -> None:
    scale = lr_multiplier(step, warmup, total)
    for group in opt.param_groups:
        group["lr"] = float(group["initial_lr"]) * scale


class _IdentityScaler:
    """CPU stand-in so the dry run still executes clip_grad_norm_."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, opt: torch.optim.Optimizer) -> None:
        return None

    def step(self, opt: torch.optim.Optimizer) -> None:
        opt.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


@dataclass
class Stepper:
    opt: torch.optim.Optimizer
    scaler: object
    accum: int = ACCUM
    micro: int = 0
    updates: int = 0

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss / self.accum).backward()
        self.micro += 1

    def ready(self) -> bool:
        return self.micro > 0 and self.micro % self.accum == 0

    def update(self, warmup: int, total: int) -> None:
        self.scaler.unscale_(self.opt)
        params = [p for group in self.opt.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.updates += 1
        apply_lrs(self.opt, self.updates, warmup, total)


def probability_std(probs: torch.Tensor) -> float:
    return float(probs.detach().float().std(unbiased=False))


def collapsed(probs: torch.Tensor, floor: float = COLLAPSE_STD) -> bool:
    if probs.numel() < 2:
        return False
    return probability_std(probs) < floor


def log_versions() -> None:
    import timm
    print("python", sys.version.replace("\n", " "), flush=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    print("timm", timm.__version__, flush=True)
    print(
        "arch", ARCH, "img", IMG, "train_res", TRAIN_RES, "k_train", K_TRAIN,
        "fold", FOLD, "head_lr", HEAD_LR, "backbone_lr", BACKBONE_LR,
        "warmup", WARMUP_STEPS, flush=True,
    )
    print("oai_images", False, "oai_checkpoint", False, flush=True)


SKIP_WALK = {"train_series", "test_series", "train_images", "test_images"}


def pruned_walk(root: Path):
    """Skip DICOM trees. Walking them looks for a CSV among millions of slices."""
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_WALK]
        yield dirpath, files


def find_one(names: list[str]) -> Path | None:
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    wanted = {name.lower() for name in names}
    for dirpath, files in pruned_walk(root):
        for name in files:
            if name.lower() in wanted:
                return Path(dirpath) / name
    return None


def competition_root() -> Path:
    for base in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (base / "train.csv").is_file():
            return base
    found = find_one(["train.csv"])
    if found is None:
        raise FileNotFoundError("train.csv is not mounted")
    return found.parent


def image_root(comp: Path) -> Path:
    cand = comp / "train_series"
    if cand.is_dir():
        return cand
    raise FileNotFoundError("competition train_series is not mounted")


def load_folds() -> pd.DataFrame:
    local = Path(__file__).resolve().parent / "folds.csv"
    if local.is_file():
        return pd.read_csv(local, dtype={"StudyInstanceUID": str})
    import base64
    import io
    return pd.read_csv(io.BytesIO(base64.b64decode(FOLDS_B64)), dtype={"StudyInstanceUID": str})


def load_targets(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v4_path = find_one(["llm_labels_v4_blend.csv"])
    if v4_path is None:
        raise FileNotFoundError("llm_labels_v4_blend.csv is not mounted")
    v4 = pd.read_csv(v4_path, dtype={"StudyInstanceUID": str})
    missing = [name for name in FINDINGS if name not in v4.columns]
    if missing:
        raise ValueError(f"v4 labels missing {missing}")
    targets = v4[["StudyInstanceUID", *FINDINGS]].drop_duplicates("StudyInstanceUID")
    for name in FINDINGS:
        targets[name] = pd.to_numeric(targets[name], errors="coerce")
    if targets[FINDINGS].isna().any().any():
        raise ValueError("v4 scores have missing values")
    weights = targets[["StudyInstanceUID"]].copy()
    for name in FINDINGS:
        weights[name] = 1.0
    pw = None
    root = Path("/kaggle/input")
    if root.is_dir():
        for dirpath, files in pruned_walk(root):
            for name in files:
                if not name.lower().endswith(".csv"):
                    continue
                path = Path(dirpath) / name
                if path == v4_path:
                    continue
                header = pd.read_csv(path, nrows=0)
                if "ACL__conf" in header.columns and "StudyInstanceUID" in header.columns:
                    pw = pd.read_csv(path, dtype={"StudyInstanceUID": str})
                    print("pilkwang", path, flush=True)
                    break
            if pw is not None:
                break
    if pw is not None:
        pw = pw.drop_duplicates("StudyInstanceUID")
        merged = targets[["StudyInstanceUID"]].merge(pw, on="StudyInstanceUID", how="left")
        for name in FINDINGS:
            conf = pd.to_numeric(merged[f"{name}__conf"], errors="coerce").clip(0, 1)
            verdict = merged[f"{name}__verdict"] if f"{name}__verdict" in merged.columns else None
            w = conf.to_numpy(dtype=np.float64)
            if verdict is not None:
                missing_verdict = verdict.isna().to_numpy()
                w = np.where(missing_verdict, 1.0, np.where(np.isfinite(w), w, 0.0))
            else:
                w = np.where(np.isfinite(w), w, 1.0)
            weights[name] = w
    else:
        print("pilkwang confidence table not mounted; v4 weights stay 1", flush=True)
    gold = set(folds.loc[folds["is_gold"] == 1, "StudyInstanceUID"].astype(str))
    weights.loc[weights["StudyInstanceUID"].isin(gold), FINDINGS] = 0.0
    print("v4", v4_path, "rows", len(targets), "gold_zeroed", len(gold), flush=True)
    return targets.set_index("StudyInstanceUID"), weights.set_index("StudyInstanceUID")


def series_map(comp: Path) -> dict:
    series = pd.read_csv(comp / "train_series.csv", dtype=str)
    out: dict = {}
    for row in series.to_dict("records"):
        out.setdefault(row["StudyInstanceUID"], []).append(row)
    return out


V3_MOUNT = Path("/kaggle/input/rsna-knee-f0-v3-ckpt/fold0_last.pt")


def mounted_v3_checkpoint() -> Path:
    path = Path(os.environ["FOLD0_CKPT"]) if os.environ.get("FOLD0_CKPT") else V3_MOUNT
    print(f"resume_path {path}", flush=True)
    return path


def load_v3_weights(model: torch.nn.Module, path: Path) -> int:
    require_v3_hash(path)
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or "model" not in saved:
            raise RuntimeError("checkpoint has no model state")
        if saved.get("oai"):
            raise RuntimeError("refusing an OAI checkpoint")
        epoch_n = int(saved.get("epoch", -1))
        if epoch_n != V3_EPOCH:
            raise RuntimeError(f"checkpoint epoch {saved.get('epoch')} is not v3 epoch {V3_EPOCH}")
        model.load_state_dict(saved["model"], strict=True)
    except Exception as exc:
        print(f"ABORT resume: {exc}", flush=True)
        sys.exit(1)
    print(f"RESUMED from v3 epoch {epoch_n}", flush=True)
    return epoch_n


def _checkpoint(path: Path, model, stepper: Stepper, epoch: int, swa) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "opt": stepper.opt.state_dict(),
            "scaler": stepper.scaler.state_dict(),
            "epoch": epoch,
            "updates": stepper.updates,
            "oai": False,
            "fold": FOLD,
            "k_train": K_TRAIN,
            "train_res": TRAIN_RES,
            "swa": None if swa is None else swa.state_dict(),
        },
        path,
    )


def _oof_rows(model, ids, builder, ser, images, targets, log_every: int = 0) -> list[dict]:
    rows = []
    model.eval()
    with torch.no_grad():
        for n, sid in enumerate(ids, start=1):
            if sid not in targets.index:
                continue
            vol, mask = builder(sid, ser, str(images))
            centers = eval_centers(mask, N_SLICES, K_EVAL)
            wins = windows_from_volume(vol, centers, TRAIN_RES)
            x = imagenet_norm(torch.from_numpy(wins)).unsqueeze(0).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prob = torch.sigmoid(model(x)).float().cpu().numpy()[0]
            rows.append({"StudyInstanceUID": sid, **{name: float(prob[i]) for i, name in enumerate(FINDINGS)}})
            if log_every and n % log_every == 0:
                print(f"oof {n}/{len(ids)}", flush=True)
    model.train()
    return rows


def _oof_macro(frame: pd.DataFrame, targets: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    aucs = []
    labels = targets.loc[frame["StudyInstanceUID"].astype(str), FINDINGS]
    for name in FINDINGS:
        raw = pd.to_numeric(labels[name], errors="coerce").to_numpy(dtype=np.float64)
        score = frame[name].to_numpy(dtype=np.float64)
        known = np.isfinite(raw) & (raw != 0.5)
        if int(known.sum()) < 2 or len(np.unique(raw[known] > 0.5)) < 2:
            continue
        aucs.append(float(roc_auc_score((raw[known] > 0.5).astype(np.float64), score[known])))
    return float(np.mean(aucs)) if aucs else 0.5


def _pred_std(frame: pd.DataFrame) -> float:
    return float(frame[FINDINGS].to_numpy(dtype=np.float64).std())


def _mean_text(frame: pd.DataFrame) -> str:
    return " ".join(f"{name}:{float(frame[name].mean()):.3f}" for name in FINDINGS)


def cpu_dry_run() -> None:
    """20 optimizer steps on CPU from the bundled resume path. K=1 to stay in RAM."""
    path = Path(os.environ.get("FOLD0_CKPT", ""))
    import timm
    torch.manual_seed(SEED)
    backbone = timm.create_model(
        ARCH, pretrained=False, num_classes=0, in_chans=3, global_pool="avg", img_size=TRAIN_RES,
    )
    model = RaptorClassifier(backbone, f_dim=backbone.num_features, n=12)
    load_v3_weights(model, path)
    stepper = Stepper(make_optimizer(model), _IdentityScaler(), accum=1)
    model.train()
    stepper.opt.zero_grad(set_to_none=True)
    gen = torch.Generator()
    gen.manual_seed(SEED)
    print("dry device cpu k 1 img", TRAIN_RES, "steps 20", flush=True)
    for _ in range(20):
        x = torch.randn(1, 1, 3, TRAIN_RES, TRAIN_RES, generator=gen)
        y = torch.randint(0, 2, (1, 12), generator=gen).float()
        w = torch.ones(1, 12)
        loss = weighted_bce(model(x), y, w)
        stepper.backward(loss)
        stepper.update(WARMUP_STEPS, 5000)
        groups = stepper.opt.param_groups
        print(
            f"step {stepper.updates} loss {float(loss.detach()):.4f} "
            f"backbone_lr {groups[0]['lr']:.8e} head_lr {groups[1]['lr']:.8e}",
            flush=True,
        )


def main() -> None:
    if os.environ.get("FOLD0_CPU_DRY") == "1":
        cpu_dry_run()
        return
    ckpt_in = mounted_v3_checkpoint()
    require_v3_hash(ckpt_in)
    log_versions()
    t0 = time.time()
    comp = competition_root()
    images = image_root(comp)
    folds = load_folds()
    targets, weights = load_targets(folds)
    train_ids = folds.loc[folds["fold"] != FOLD, "StudyInstanceUID"].astype(str).tolist()
    val_ids = folds.loc[folds["fold"] == FOLD, "StudyInstanceUID"].astype(str).tolist()
    train_ids = [sid for sid in train_ids if sid in targets.index and float(weights.loc[sid, FINDINGS].sum()) > 0]
    val_ids = [sid for sid in val_ids if sid in targets.index]
    oof_ids = val_ids[:OOF_SUBSET_N]
    print("train", len(train_ids), "val", len(val_ids), "oof_subset", len(oof_ids), "images", images, flush=True)

    import timm
    backbone = timm.create_model(
        ARCH, pretrained=False, num_classes=0, in_chans=3, global_pool="avg", img_size=TRAIN_RES,
    )
    model = RaptorClassifier(backbone, f_dim=backbone.num_features, n=12).cuda()
    stepper = Stepper(make_optimizer(model), torch.amp.GradScaler("cuda"), accum=ACCUM)
    start_epoch = load_v3_weights(model, ckpt_in)
    print("v5 optimizer fresh, warmup", WARMUP_STEPS, "from", ckpt_in, flush=True)
    swa = None
    out = Path("/kaggle/working")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "fold0_last.pt"
    builder = _fastread.make_builder("study_lut", IMG, SLOTS, hdr_threads=32, px_threads=16)
    ser = series_map(comp)
    rng = np.random.default_rng(SEED + start_epoch)
    remaining = max(EPOCHS - start_epoch, 1)
    total_updates = max(1, (len(train_ids) * remaining) // ACCUM)
    seen_probs: list[torch.Tensor] = []

    def one_study(sid: str):
        vol, mask = builder(sid, ser, str(images))
        if vol.shape != (N_SLICES, IMG, IMG):
            raise RuntimeError(f"{sid} volume {vol.shape}")
        centers = sample_train_centers(mask, N_SLICES, K_TRAIN, rng)
        wins = windows_from_volume(vol, centers, TRAIN_RES)
        x = imagenet_norm(torch.from_numpy(wins)).unsqueeze(0).cuda(non_blocking=True)
        y = torch.tensor(targets.loc[sid, FINDINGS].to_numpy(np.float32), device="cuda").view(1, 12)
        w = torch.tensor(weights.loc[sid, FINDINGS].to_numpy(np.float32), device="cuda").view(1, 12)
        return x, y, w

    def finish(status: str, epoch: int, extra: dict | None = None) -> None:
        _checkpoint(ckpt, model, stepper, epoch, swa)
        held = swa.module if swa is not None else model
        rows = _oof_rows(held, val_ids, builder, ser, images, targets, log_every=200)
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "fold0_oof.csv", index=False)
        payload = {
            "status": status,
            "epoch": epoch,
            "resumed_from_epoch": start_epoch,
            "wall_hours": (time.time() - t0) / 3600.0,
            "oof_rows": len(rows),
            "pred_std": _pred_std(frame) if len(frame) else None,
            "oof_macro": _oof_macro(frame, targets) if len(frame) else None,
            "head_lr": HEAD_LR,
            "backbone_lr": BACKBONE_LR,
            "train_res": TRAIN_RES,
            "warmup_steps": WARMUP_STEPS,
        }
        if extra:
            payload.update(extra)
        (out / "fold0_timing.json").write_text(json.dumps(payload))
        print("EXIT", json.dumps(payload), flush=True)

    def stop_collapsed(probs: torch.Tensor, epoch: int) -> bool:
        if not collapsed(probs):
            return False
        pred_std = probability_std(probs)
        payload = {"status": "collapse", "pred_std": pred_std, "floor": COLLAPSE_STD, "epoch": epoch}
        (out / "collapse.json").write_text(json.dumps(payload))
        print("COLLAPSE", json.dumps(payload), flush=True)
        finish("collapse", epoch, {"pred_std": pred_std, "floor": COLLAPSE_STD})
        return True

    model.train()
    stepper.opt.zero_grad(set_to_none=True)
    study_i = 0
    t_step = time.time()
    last_epoch = start_epoch
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        if (time.time() - t0) > WALL_HOURS * 3600:
            finish("wall", last_epoch)
            return
        last_epoch = epoch
        order = rng.permutation(len(train_ids))
        running = 0.0
        nstep = 0
        for idx in order:
            if (time.time() - t0) > WALL_HOURS * 3600:
                finish("wall", epoch)
                return
            x, y, w = one_study(train_ids[int(idx)])
            study_i += 1
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = weighted_bce(logits, y, w)
            stepper.backward(loss)
            running += float(loss.detach())
            nstep += 1
            seen_probs.append(torch.sigmoid(logits.detach()).float().cpu())
            if len(seen_probs) > 32:
                seen_probs.pop(0)
            if study_i == 50:
                rate = (time.time() - t_step) / 50.0
                projected = epoch_seconds(rate * 50.0, 50, len(train_ids))
                print(f"s/iter {rate:.3f} projected_epoch_hours {projected / 3600:.3f}", flush=True)
            if stepper.ready():
                stepper.update(WARMUP_STEPS, total_updates)
                if len(seen_probs) >= 8 and stop_collapsed(torch.cat(seen_probs, 0), epoch):
                    return
        avg = running / max(nstep, 1)
        if epoch in SWA_EPOCHS:
            if swa is None:
                swa = torch.optim.swa_utils.AveragedModel(model)
            swa.update_parameters(model)
        _checkpoint(ckpt, model, stepper, epoch, swa)
        if (time.time() - t0) >= WALL_HOURS * 3600:
            finish("wall", epoch)
            return
        held = swa.module if swa is not None else model
        rows = _oof_rows(held, oof_ids, builder, ser, images, targets)
        if not rows:
            print(f"epoch {epoch} steps {nstep} loss {avg:.4f} subset empty", flush=True)
            continue
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "fold0_oof_subset.csv", index=False)
        pred_std = _pred_std(frame)
        auc = _oof_macro(frame, targets)
        resumed_ep = epoch - start_epoch
        print(
            f"epoch {epoch} resumed_epoch {resumed_ep} steps {nstep} loss {avg:.4f} "
            f"subset_oof {auc:.4f} std {pred_std:.6f} mean {_mean_text(frame)}",
            flush=True,
        )
        if pred_std < COLLAPSE_STD:
            payload = {"status": "collapse", "epoch": epoch, "pred_std": pred_std, "floor": COLLAPSE_STD}
            (out / "collapse.json").write_text(json.dumps(payload))
            print("COLLAPSE", json.dumps(payload), flush=True)
            finish("collapse", epoch, payload)
            return
        if resumed_ep == OOF_KILL_RESUMED_EPOCH and auc < OOF_KILL_FLOOR:
            print("KILL oof", auc, OOF_KILL_FLOOR, flush=True)
            finish("kill", epoch, {"subset_oof": auc, "floor": OOF_KILL_FLOOR, "pred_std": pred_std})
            return
    finish("ran", last_epoch)


if __name__ == "__main__":
    main()
