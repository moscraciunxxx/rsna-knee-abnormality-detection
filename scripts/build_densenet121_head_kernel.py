"""Private DenseNet121 head on the non-holdout LLM-soft rows. Does not Submit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "kaggle_kernel_densenet121_head"

EMBED = (
    ROOT / "rsna_knee" / "metrics.py",
    ROOT / "rsna_knee" / "v4_split.py",
    ROOT / "rsna_knee" / "soft_labels.py",
    ROOT / "rsna_knee" / "radimagenet_densenet.py",
    ROOT / "rsna_knee" / "densenet_probe.py",
    ROOT / "rsna_knee" / "densenet_train.py",
)

RUNNER = r'''
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
print("nvidia-smi:", subprocess.getoutput("nvidia-smi -L"), flush=True)
print(
    "[dn121-head] PRIVATE frozen RadImageNet-DenseNet121 + new 12-way head. "
    "Gold-58 out of the loss. Mill-870 out of the loss. DO NOT SUBMIT. keep 55779066.",
    flush=True,
)

HARD_LIMIT_S = int(ENCODE_ABORT_S)


def _watchdog() -> None:
    time.sleep(HARD_LIMIT_S)
    print(f"[dn121-head] WATCHDOG {HARD_LIMIT_S}s", flush=True)
    os._exit(99)


threading.Thread(target=_watchdog, daemon=True, name="dn121-head-watchdog").start()


def _comp_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for path in candidates:
        if (path / "train.csv").is_file() and (path / "train_series").is_dir():
            return path
    base = Path("/kaggle/input")
    for path in base.rglob("train.csv"):
        if (path.parent / "train_series").is_dir():
            return path.parent
    raise FileNotFoundError("competition train_series not mounted")


def _find_named(name: str) -> Path | None:
    base = Path("/kaggle/input")
    skip = {"train_series", "test_series", "train_images", "test_images"}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [item for item in dirnames if item not in skip and not item.startswith("1.2.")]
        if name in filenames:
            return Path(dirpath) / name
    return None


def _study_batch(root: Path, uid: str, k: int = IMAGES_PER_STUDY) -> torch.Tensor | None:
    folder = root / "train_series" / uid
    files = sorted(path for path in folder.rglob("*") if path.suffix.lower() == ".dcm")
    if not files:
        return None
    if len(files) > k:
        pick = np.linspace(0, len(files) - 1, k).astype(int)
        files = [files[int(i)] for i in pick]
    images = []
    for path in files:
        array = np.asarray(pydicom.dcmread(str(path), force=True).pixel_array, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        low, high = np.percentile(array, [1, 99])
        scale = max(float(high - low), 1e-6)
        array = np.clip((array - float(low)) / scale, 0.0, 1.0).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        images.append(tensor.repeat(1, 3, 1, 1))
    if not images:
        return None
    return torch.cat(images, dim=0)


def _encode(model, device, comp: Path, ids: list[str]) -> tuple[np.ndarray, int]:
    rows = []
    missing = 0
    started = time.perf_counter()
    n_images = 0
    for n_done, uid in enumerate(ids, start=1):
        batch = _study_batch(comp, uid)
        if batch is None:
            missing += 1
            rows.append(np.full(1024, np.nan, dtype=np.float32))
        else:
            with torch.no_grad():
                feats = model.features(batch.to(device).float())
                pooled = F.adaptive_avg_pool2d(F.relu(feats), (1, 1)).flatten(1).mean(dim=0)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))
            n_images += int(batch.shape[0])
            if device.type == "cuda":
                torch.cuda.synchronize()
        if n_done == PROBE_STUDIES and n_images > 0:
            elapsed = time.perf_counter() - started
            if encode_probe_aborts(elapsed, n_images):
                print(
                    f"[dn121-head] ABORT early projected encode > 2.8h "
                    f"elapsed_s={elapsed:.1f} images={n_images}",
                    flush=True,
                )
                raise SystemExit(2)
        if n_done % 200 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[dn121-head] encoded {n_done}/{len(ids)} images={n_images} "
                f"missing={missing} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(
        f"[dn121-head] encode done n={len(ids)} images={n_images} missing={missing} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return np.stack(rows), missing


def _table(ids, scores) -> pd.DataFrame:
    frame = pd.DataFrame(scores, columns=list(FINDINGS))
    frame.insert(0, "StudyInstanceUID", list(ids))
    return frame


def main() -> None:
    comp = _comp_root()
    v4_path = _find_named("llm_labels_v4_blend.csv")
    if v4_path is None:
        raise SystemExit("[dn121-head] steven v4 labels missing")
    h5_path = _find_named("RadImageNet-DenseNet121_notop.h5")
    if h5_path is None:
        raise SystemExit("[dn121-head] densenet h5 missing")
    train = pd.read_csv(comp / "train.csv", usecols=["StudyInstanceUID", *FINDINGS])
    v4 = pd.read_csv(v4_path, dtype={"StudyInstanceUID": str})
    fit_ids, hold_ids, gold_ids = densenet_id_split(train, v4)
    print(
        f"[dn121-head] fit_n={len(fit_ids)} holdout_n={len(hold_ids)} gold_n={len(gold_ids)} "
        f"gold_in_fit={len(set(fit_ids) & set(gold_ids))} "
        f"holdout_in_fit={len(set(fit_ids) & set(hold_ids))}",
        flush=True,
    )
    model, report = load_radimagenet_densenet121(h5_path)
    print(
        f"[dn121-head] loaded n={report['n_loaded']} unexpected={len(report['unexpected'])} "
        f"frozen={report['features_frozen']} head={report['classifier_out']}",
        flush=True,
    )
    if report["unexpected"] or not report["features_frozen"]:
        raise SystemExit("[dn121-head] weight load failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    order = list(fit_ids) + list(hold_ids) + list(gold_ids)
    encoded, missing = _encode(model, device, comp, order)
    by_id = {uid: encoded[i] for i, uid in enumerate(order)}

    def stack(ids):
        return np.stack([by_id[uid] for uid in ids])

    v4_idx = v4.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    y_fit = v4_idx.loc[list(fit_ids), list(FINDINGS)].to_numpy(dtype=np.float64)
    pw_path = _find_named("report_labels_v2.csv")
    if pw_path is None:
        print("[dn121-head] pilkwang conf missing; weight=1", flush=True)
        w_fit = np.ones_like(y_fit)
    else:
        pw = pd.read_csv(pw_path, dtype={"StudyInstanceUID": str})
        _targets, weights = grade_soft_targets(v4, pw)
        w_idx = weights.set_index("StudyInstanceUID")
        w_fit = w_idx.reindex(list(fit_ids))[list(FINDINGS)].to_numpy(dtype=np.float64)
        w_fit = np.where(np.isfinite(w_fit), w_fit, 1.0)
    x_fit = stack(fit_ids)
    if not np.isfinite(x_fit).all() or not np.isfinite(y_fit).all():
        raise SystemExit("[dn121-head] non-finite fit rows")
    x_hold = stack(hold_ids)
    x_gold = stack(gold_ids)
    y_gold = (
        train.drop_duplicates("StudyInstanceUID")
        .assign(StudyInstanceUID=lambda frame: frame["StudyInstanceUID"].astype(str))
        .set_index("StudyInstanceUID")
        .loc[list(gold_ids), list(FINDINGS)]
        .to_numpy(dtype=np.float64)
    )
    eval_scores = fit_linear_head(x_fit, y_fit, w_fit, np.concatenate([x_gold, x_hold]))
    gold_scores = eval_scores[: len(gold_ids)]
    hold_scores = eval_scores[len(gold_ids) :]
    macro, abort = gold_gate(y_gold, gold_scores)
    print(f"[dn121-head] gold_macro={macro:.4f} abort={abort} threshold=0.94", flush=True)
    work = Path("/kaggle/working")
    work.mkdir(parents=True, exist_ok=True)
    sub = work / "submission.csv"
    if sub.is_file():
        sub.unlink()
    log = (
        f"fit_n={len(fit_ids)}\nholdout_n={len(hold_ids)}\ngold_n={len(gold_ids)}\n"
        f"gold_in_fit=0\nholdout_in_fit=0\nmissing_images={missing}\n"
        f"gold_macro={macro:.6f}\nabort={abort}\nDO NOT SUBMIT\n"
    )
    (work / "protocol_log.txt").write_text(log, encoding="utf-8")
    _table(gold_ids, gold_scores).to_csv(work / "densenet_gold58.csv", index=False)
    if abort:
        print("[dn121-head] ABORT gold macro > 0.94. No holdout license. DO NOT SUBMIT.", flush=True)
        raise SystemExit(0)
    _table(hold_ids, hold_scores).to_csv(work / "densenet_holdout870.csv", index=False)
    (work / "DO_NOT_SUBMIT_HOLDOUT.txt").write_text(
        "densenet_holdout870.csv is the seed-2026 train holdout, not the hidden test.\n"
        "Keep 55779066 until a local mill-870 join licenses a TEST overlay.\n",
        encoding="utf-8",
    )
    print(
        f"[dn121-head] wrote densenet_holdout870.csv rows={len(hold_ids)} missing_images={missing}. "
        "DO NOT SUBMIT. keep 55779066.",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def _embed(path: Path) -> str:
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("from __future__") or line.startswith("from rsna_knee"):
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def build_kernel(dst: Path = DST) -> Path:
    parts = [
        '''"""Private RadImageNet-DenseNet121 head. Do not Submit.

Fit on non-gold LLM-soft rows except the mill-870 holdout. Gold macro above
0.94 aborts before a holdout file is written.
"""
from __future__ import annotations

ID_COLUMN = "StudyInstanceUID"
FINDINGS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
''',
    ]
    parts.extend(_embed(path) for path in EMBED)
    parts.append(RUNNER)
    text = "\n".join(parts)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "main.py").write_text(text, encoding="utf-8")
    compile(text, str(dst / "main.py"), "exec")
    meta = {
        "id": "vitaliecervinschi/rsna-knee-densenet121-head870",
        "title": "RSNA Knee densenet121 head870",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "sjudal/radimagenet-densenet121",
            "stevenleehans/rsna-knee-llm-report-labels",
            "pilkwang/rsna-knee-llm-labels",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
    }
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dst}")
    print("DO NOT SUBMIT. Private head gate only.")
    return dst


if __name__ == "__main__":
    build_kernel()
