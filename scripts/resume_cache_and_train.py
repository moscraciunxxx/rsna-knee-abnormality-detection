"""Resume vols.tar downloads, extract, smoke-test, then train reader fold 3.

One process. Logs to D:\\rsna_cache\\dl.log. No competition submit and no Kaggle GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ROOT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection")
CACHE = Path(r"D:\rsna_cache")
LOG = CACHE / "dl.log"
PY = sys.executable
OWNER = "vitaliecervinschi"
ATTEMPTS = 10
BACKOFF_S = 10

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
STEVEN = ROOT / "data" / "external_labels" / "llm_labels_v4_blend.csv"
PILK = ROOT / ".cache" / "llm-labels" / "report_labels_v2.csv"
LIXIN = ROOT / ".cache" / "llm_lixin" / "labels_llm_gpt56sol.csv"
QWEN = ROOT / ".cache" / "qwen" / "qwen_knee_weak_labels.csv"
SERIES_CSV = ROOT / "data" / "train_series.csv"
TRAIN_CSV = ROOT / "data" / "train.csv"
KNEE_DIR = ROOT / "artifacts" / "provenance" / "reader_weights"
RUN_DIR = ROOT / "artifacts" / "reader_fold3"


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S") + " " + msg
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def fresh_output(slug: str, want: str | None = None):
    """Return (url or '', kernel log). Pages until `want` is found."""
    api = KaggleApi()
    api.authenticate()
    token = None
    found = ""
    log_text = ""
    with api.build_kaggle_client() as kaggle:
        for _ in range(80):
            req = ApiListKernelSessionOutputRequest()
            req.user_name = OWNER
            req.kernel_slug = slug
            req.page_size = 100
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            if resp.log:
                log_text = resp.log
            for item in resp.files or []:
                if want is None or item.file_name == want:
                    found = item.url
                    if want:
                        return found, log_text
            token = resp.next_page_token
            if not token or found:
                break
    return found, log_text


def parse_tar_log(text: str, fold: int) -> tuple[int | None, int | None]:
    match = re.search(rf"fold {fold} files (\d+) bytes (\d+)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def stream_resume(url: str, part: Path) -> tuple[str, int]:
    """Append with Range. Return ('ok', total) or ('ignored', 0)."""
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    resp = requests.get(url, headers=headers, stream=True, timeout=(30, 120))
    try:
        if start and resp.status_code == 200:
            return "ignored", 0
        if resp.status_code == 416:
            probe = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=(30, 60))
            header = probe.headers.get("Content-Range", "")
            probe.close()
            match = re.match(r"bytes (\d+)-(\d+)/(\d+)", header)
            if not match:
                raise RuntimeError(f"range not satisfiable at {start} ({header!r})")
            total = int(match.group(3))
            if start == total:
                return "ok", total
            raise RuntimeError(f"range not satisfiable at {start} of {total}")
        resp.raise_for_status()
        if resp.status_code == 206:
            header = resp.headers.get("Content-Range", "")
            match = re.match(r"bytes (\d+)-(\d+)/(\d+)", header)
            if not match:
                raise RuntimeError(f"missing Content-Range {header!r}")
            begin, _end, total = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if begin != start:
                raise RuntimeError(f"server begin {begin} != local {start}")
            mode = "ab"
        else:
            total = int(resp.headers.get("Content-Length") or 0)
            if total <= 0:
                raise RuntimeError("no Content-Length on full response")
            mode = "wb"
            start = 0
        written = start
        t0 = time.time()
        last = t0
        with part.open(mode) as out:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                out.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last >= 20:
                    mbps = ((written - start) / 1e6) / max(now - t0, 1e-6)
                    log(f"  {part.name} {written}/{total} mbps {mbps:.2f}")
                    last = now
        if written != total:
            raise RuntimeError(f"short read {written} != {total}")
        return "ok", total
    finally:
        resp.close()


def download_fold(fold: int) -> None:
    dest = CACHE / f"f{fold}"
    dest.mkdir(parents=True, exist_ok=True)
    part = dest / "vols.tar.part"
    final = dest / "vols.tar"
    slug = f"rsna-cache-tar-f{fold}"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            url, klog = fresh_output(slug, "vols.tar")
            if not url:
                raise RuntimeError("no signed url")
            n_files, n_bytes = parse_tar_log(klog, fold)
            if n_bytes and part.exists() and part.stat().st_size > n_bytes:
                log(f"f{fold} part longer than tar ({part.stat().st_size} > {n_bytes}), restart from 0")
                part.unlink()
            if final.exists():
                final.unlink()
            if n_bytes and part.exists() and part.stat().st_size == n_bytes:
                log(f"f{fold} part already complete {n_bytes}")
                part.replace(final)
            else:
                have = part.stat().st_size if part.exists() else 0
                log(f"f{fold} attempt {attempt} resume_at {have} log_bytes {n_bytes} log_files {n_files}")
                kind, total = stream_resume(url, part)
                if kind == "ignored":
                    log(f"f{fold} server ignored Range, restart from 0")
                    if part.exists():
                        part.unlink()
                    url, _klog = fresh_output(slug, "vols.tar")
                    kind, total = stream_resume(url, part)
                    if kind != "ok":
                        raise RuntimeError("restart from 0 did not return a full body")
                size = part.stat().st_size
                if size != total:
                    raise RuntimeError(f"size {size} != http {total}")
                if n_bytes and size != n_bytes:
                    raise RuntimeError(f"size {size} != log {n_bytes}")
                log(f"f{fold} size ok {size}")
                part.replace(final)
            n_ext = extract(final, dest)
            manifest = fetch_manifest(fold, dest / "manifest.csv")
            ok, msg = verify(dest, manifest)
            log(f"f{fold} verify {ok} {msg} extracted {n_ext}")
            if not ok:
                raise RuntimeError(msg)
            final.unlink()
            log(f"f{fold} tar deleted")
            return
        except Exception as exc:
            log(f"f{fold} attempt {attempt} FAIL {type(exc).__name__}: {exc}")
            if final.exists():
                log(f"f{fold} extract or verify failed, restart from 0")
                final.unlink()
                if part.exists():
                    part.unlink()
            if attempt == ATTEMPTS:
                raise
            time.sleep(BACKOFF_S)


def extract(tar_path: Path, dest: Path) -> int:
    n = 0
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or not name.endswith(".npz"):
                continue
            target = dest / name
            src = tar.extractfile(member)
            if src is None:
                continue
            with target.open("wb") as out:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            n += 1
            if n % 200 == 0:
                log(f"  extracted {n}")
    return n


def fetch_manifest(fold: int, dest: Path) -> Path:
    slug = f"rsna-knee-cache-f{fold}-no-oai"
    last = "manifest not fetched"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            url, _log = fresh_output(slug, "manifest.csv")
            if not url:
                raise RuntimeError("manifest.csv not in listing")
            resp = requests.get(url, timeout=(30, 60))
            resp.raise_for_status()
            body = resp.content
            if body[:1] == b"<" or b"StudyInstanceUID" not in body[:300]:
                raise RuntimeError(f"manifest not csv {body[:40]!r}")
            dest.write_bytes(body)
            return dest
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            log(f"f{fold} manifest attempt {attempt} {last}")
            time.sleep(BACKOFF_S)
    raise RuntimeError(last)


def verify(dest: Path, manifest: Path) -> tuple[bool, str]:
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = {row["StudyInstanceUID"] for row in rows}
    got = {path.stem for path in dest.glob("*.npz")}
    missing = len(ids - got)
    extra = len(got - ids)
    ok = missing == 0 and extra == 0 and len(got) == len(rows)
    return ok, f"npz {len(got)} manifest {len(rows)} missing {missing} extra {extra}"


def ensure_qwen() -> None:
    if QWEN.is_file() and QWEN.stat().st_size > 1000:
        log(f"qwen present {QWEN.stat().st_size}")
        return
    QWEN.parent.mkdir(parents=True, exist_ok=True)
    kaggle = r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
    log("downloading qwen labels")
    proc = subprocess.run(
        [kaggle, "datasets", "download", "-d", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels",
         "-f", "qwen_knee_weak_labels.csv", "-p", str(QWEN.parent), "--unzip"],
        check=False,
        capture_output=True,
        text=True,
    )
    log(f"qwen exit {proc.returncode} {(proc.stderr or proc.stdout or '')[-400:]}")
    if not QWEN.is_file():
        found = list(QWEN.parent.rglob("qwen_knee_weak_labels.csv"))
        if not found:
            raise RuntimeError("qwen csv missing")
        found[0].replace(QWEN)


def pull() -> None:
    import threading

    log("pull start")
    bad = CACHE / "f0" / "manifest.csv"
    if bad.is_file() and bad.read_bytes()[:1] == b"<":
        bad.unlink()
        log("removed html manifest from the earlier 502")
    qwen_error: list[BaseException] = []

    def _qwen() -> None:
        try:
            ensure_qwen()
        except BaseException as exc:
            qwen_error.append(exc)

    qwen_thread = threading.Thread(target=_qwen, daemon=True)
    qwen_thread.start()
    download_fold(0)
    qwen_thread.join()
    if qwen_error:
        raise qwen_error[0]
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUNBUFFERED"] = "1"
    smoke = subprocess.Popen([PY, str(Path(__file__).resolve()), "--mode", "smoke"], env=env, cwd=str(ROOT))
    log(f"smoke pid {smoke.pid}")
    try:
        for fold in range(1, 5):
            download_fold(fold)
    finally:
        code = smoke.wait()
        log(f"smoke exit {code}")
    if code != 0:
        raise SystemExit(code)
    log("train fold 3")
    train = subprocess.run(
        [PY, str(Path(__file__).resolve()), "--mode", "train", "--fold", "3"],
        env=env,
        cwd=str(ROOT),
        check=False,
    )
    log(f"train exit {train.returncode}")
    raise SystemExit(train.returncode)


def build_labels():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    ids = train["StudyInstanceUID"].astype(str)
    gold = train.set_index("StudyInstanceUID")[LABELS]
    gold.index = gold.index.astype(str)
    is_gold = gold.notna().all(axis=1)
    frames = []
    steven = pd.read_csv(STEVEN, dtype={"StudyInstanceUID": str})
    pilk = pd.read_csv(PILK, dtype={"StudyInstanceUID": str})
    lixin = pd.read_csv(LIXIN, dtype={"StudyInstanceUID": str})
    qwen = pd.read_csv(QWEN, dtype={"StudyInstanceUID": str})
    qwen = qwen[["StudyInstanceUID", *[f"{c}__label" for c in LABELS]]]
    qwen.columns = ["StudyInstanceUID", *LABELS]
    for frame in (steven, pilk, lixin, qwen):
        frame = frame.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        frames.append(frame[LABELS].astype(float).reindex(ids))
    num = sum(frame.fillna(0) for frame in frames)
    den = sum(frame.notna() for frame in frames)
    soft = (num / den).clip(0.01, 0.99)
    soft.index = ids.to_numpy()
    out = soft.copy()
    out.insert(0, "is_gold", is_gold.reindex(soft.index).astype(int).to_numpy())
    out["fold"] = -1
    weak = out["is_gold"].to_numpy() == 0
    groups = train["Report"].map(lambda text: re.sub(r"\W+", "", str(text).lower())).to_numpy()[weak]
    order = np.random.RandomState(0).permutation(int(weak.sum()))
    folds = np.zeros(int(weak.sum()), int)
    for k, (_, va) in enumerate(GroupKFold(5).split(order, groups=groups[order])):
        folds[order[va]] = k
    out.loc[weak, "fold"] = folds
    log("fold sizes " + str(out["fold"].value_counts().sort_index().to_dict()))
    return out


class StudyCache:
    def __init__(self, folders: list[Path]):
        self.study = {}
        for folder in folders:
            for path in folder.glob("*.npz"):
                self.study[path.stem] = path
        self.series_study: dict[str, str] = {}
        self._zip = None
        self._sid = None
        self._counts = None

    def bind(self, series) -> None:
        for study, serie in zip(series.StudyInstanceUID.astype(str), series.SeriesInstanceUID.astype(str)):
            if study in self.study:
                self.series_study[serie] = study

    def n_slices(self) -> dict[str, int]:
        if self._counts is not None:
            return self._counts
        import numpy as np
        counts = {}
        for i, (sid, path) in enumerate(self.study.items(), start=1):
            with np.load(path) as blob:
                for name in blob.files:
                    counts[name] = int(blob[name].shape[0])
            if i % 200 == 0:
                log(f"  slice counts {i}/{len(self.study)}")
        self._counts = counts
        return counts

    def __call__(self, series_uid: str):
        sid = self.series_study[series_uid]
        if self._sid != sid:
            if self._zip is not None:
                self._zip.close()
            import numpy as np
            self._zip = np.load(self.study[sid])
            self._sid = sid
        return self._zip[series_uid]


def make_model(device):
    import copy
    import torch
    sys.path.insert(0, str(KNEE_DIR))
    from knee import KneeNet

    model = KneeNet("convnext_tiny", pretrained=True).to(device)
    ema = copy.deepcopy(model).eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    return model, ema


def fit_micro(model, batch, device) -> int:
    import torch
    import torch.nn.functional as F

    x, mask, pos, y = [t[:1].to(device) for t in batch]
    for micro in (2, 1):
        torch.cuda.empty_cache()
        try:
            xb = torch.cat([x] * micro, 0) if micro == 2 else x
            mb = torch.cat([mask] * micro, 0) if micro == 2 else mask
            pb = torch.cat([pos] * micro, 0) if micro == 2 else pos
            yb = torch.cat([y] * micro, 0) if micro == 2 else y
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xb, mb, pb, res=256, train=True)
            loss = F.binary_cross_entropy_with_logits(logit.float(), yb)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            log(f"micro {micro} fits")
            return micro
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            log(f"micro {micro} oom")
    raise torch.cuda.OutOfMemoryError("one study does not fit")


def run_steps(model, ema, loader, device, steps: int, micro: int, n_train: int) -> None:
    import math
    import time
    import torch
    import torch.nn.functional as F

    accum = 8 // micro
    enc = list(model.enc.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith("enc.")]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": 1e-4}, {"params": head, "lr": 5e-4}],
        weight_decay=0.02,
    )
    steps_per_epoch = max(n_train // 8, 1)
    total_steps = 14 * steps_per_epoch
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))),
    )
    data = iter(loader)
    model.train()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    times = []
    opt_step = 0
    while opt_step < steps:
        step_t = time.time()
        total_loss = 0.0
        for _ in range(accum):
            try:
                x, mask, pos, y = next(data)
            except StopIteration:
                data = iter(loader)
                x, mask, pos, y = next(data)
            x, mask, pos, y = x.to(device), mask.to(device), pos.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, mask, pos, res=256, train=True)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y) / accum
            loss.backward()
            total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        opt_step += 1
        decay = min(0.998, (1 + opt_step) / (10 + opt_step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        torch.cuda.synchronize()
        times.append(time.time() - step_t)
        if opt_step == 50 or opt_step == steps:
            log(f"step {opt_step} s/iter {sum(times) / len(times):.3f} loss {total_loss:.4f}")
    steady = times[5:] if len(times) > 5 else times
    s_iter = sum(steady) / len(steady)
    hours = s_iter * steps_per_epoch * 14 / 3600
    log(
        f"SMOKE s/iter {s_iter:.3f} hours/fold {hours:.2f} micro {micro} accum {accum} "
        f"n_train {n_train} steps_per_epoch {steps_per_epoch} wall {time.time() - t0:.1f}"
    )
    (CACHE / "smoke.json").write_text(
        '{"s_per_iter": %.6f, "hours_per_fold": %.4f, "micro": %d, "accum": %d, "n_train": %d}\n'
        % (s_iter, hours, micro, accum, n_train),
        encoding="utf-8",
    )


def cache_for(folds: list[int]) -> StudyCache:
    import pandas as pd

    folders = [CACHE / f"f{n}" for n in folds]
    missing = [str(p) for p in folders if not any(p.glob("*.npz"))]
    if missing:
        raise RuntimeError(f"cache missing {missing}")
    cache = StudyCache(folders)
    series = pd.read_csv(SERIES_CSV, dtype=str)
    cache.bind(series)
    return cache


def loader_for(labels, cache: StudyCache, fold: int | None, train: bool, micro: int, k: int):
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, str(KNEE_DIR))
    from knee import StudyDataset, build_study_table

    series = pd.read_csv(SERIES_CSV, dtype=str)
    counts = cache.n_slices()
    table = build_study_table(series, counts)
    weak = labels[labels["is_gold"] == 0]
    if fold is None:
        studies = weak[weak.index.astype(str).isin(cache.study)]
    elif train:
        studies = weak[(weak["fold"] != fold) & weak.index.astype(str).isin(cache.study)]
    else:
        studies = weak[(weak["fold"] == fold) & weak.index.astype(str).isin(cache.study)]
    y = studies[LABELS].to_numpy(np.float32) if train or fold is None else None
    ds = StudyDataset(studies.index, table, None, y, k, train=train, loader=cache)
    return torch.utils.data.DataLoader(ds, batch_size=micro, shuffle=train, drop_last=train, num_workers=0), len(studies)


def smoke() -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import torch

    log(f"smoke device {torch.cuda.get_device_name(0)}")
    labels = build_labels()
    n_train = int(((labels["is_gold"] == 0) & (labels["fold"] != 3)).sum())
    cache = cache_for([0])
    loader, n = loader_for(labels, cache, None, True, 1, 8)
    log(f"smoke studies {n} gold_in_loss 0")
    device = torch.device("cuda")
    model, ema = make_model(device)
    batch = next(iter(loader))
    micro = fit_micro(model, batch, device)
    if micro != 1:
        loader, n = loader_for(labels, cache, None, True, micro, 8)
    run_steps(model, ema, loader, device, 50, micro, n_train)


def predict_fold(model, loader, device):
    import numpy as np
    import torch

    model.eval()
    out = []
    ids = []
    with torch.no_grad():
        for x, mask, pos, _y in loader:
            x, mask, pos = x.to(device), mask.to(device), pos.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, mask, pos, res=256, train=False)
            out.append(logit.float().cpu().numpy())
    probs = 1 / (1 + np.exp(-np.concatenate(out)))
    return probs


def train_fold(fold: int) -> None:
    import math
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    log(f"train fold {fold} device {torch.cuda.get_device_name(0)}")
    labels = build_labels()
    cache = cache_for([0, 1, 2, 3, 4])
    micro = 1
    smoke_path = CACHE / "smoke.json"
    if smoke_path.is_file():
        micro = int(re.search(r'"micro": (\d+)', smoke_path.read_text(encoding="utf-8")).group(1))
    train_loader, n_train = loader_for(labels, cache, fold, True, micro, 8)
    # rebuild counts already done inside loader_for twice if we call again; val uses same cache
    val_loader, n_val = loader_for(labels, cache, fold, False, 1, 12)
    log(f"fold {fold} train {n_train} val {n_val} micro {micro} gold_in_loss 0")
    device = torch.device("cuda")
    model, ema = make_model(device)
    accum = 8 // micro
    enc = list(model.enc.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith("enc.")]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": 1e-4}, {"params": head, "lr": 5e-4}],
        weight_decay=0.02,
    )
    steps_per_epoch = max(len(train_loader) // accum, 1)
    total_steps = 14 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / steps_per_epoch)
        * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total_steps, 1))))),
    )
    weak = labels[(labels["is_gold"] == 0) & (labels["fold"] == fold)]
    weak = weak[weak.index.astype(str).isin(cache.study)]
    val_y = (weak[LABELS].to_numpy(float) > 0.5).astype(int)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    opt_step = 0
    for epoch in range(14):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        seen = 0
        pending = 0
        t0 = time.time()
        for x, mask, pos, y in train_loader:
            x, mask, pos, y = x.to(device), mask.to(device), pos.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, mask, pos, res=256, train=True)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y) / accum
            loss.backward()
            total += float(loss.detach()) * accum
            seen += 1
            pending += 1
            if pending == accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                decay = min(0.998, (1 + opt_step) / (10 + opt_step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
                pending = 0
                if opt_step == 50:
                    log(f"fold {fold} step 50 s/iter {(time.time() - t0) / opt_step:.3f}")
        try:
            probs = predict_fold(ema, val_loader, device)
        except torch.cuda.OutOfMemoryError:
            log(f"fold {fold} eval oom at k=12, retry k=8")
            torch.cuda.empty_cache()
            val_loader, n_val = loader_for(labels, cache, fold, False, 1, 8)
            probs = predict_fold(ema, val_loader, device)
        std = float(probs.std())
        aucs = [
            roc_auc_score(val_y[:, i], probs[:, i]) if 0 < val_y[:, i].sum() < len(val_y) else np.nan
            for i in range(12)
        ]
        macro = float(np.nanmean(aucs))
        log(
            f"fold {fold} epoch {epoch} loss {total / max(seen, 1):.4f} oof {macro:.4f} "
            f"std {std:.6f} sec {time.time() - t0:.0f}"
        )
        frame = pd.DataFrame(probs, index=weak.index, columns=LABELS)
        frame.to_csv(RUN_DIR / f"oof_epoch{epoch}.csv")
        torch.save(
            {"model": ema.state_dict(), "args": {"fold": fold, "backbone": "convnext_tiny", "res": 256, "labels": "v0", "epoch": epoch}},
            RUN_DIR / f"fold{fold}.pt",
        )
        if std < 1e-3:
            log(f"KILL fold {fold} pred std {std:.6e}")
            return
    log(f"fold {fold} saved {RUN_DIR / f'fold{fold}.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pull", "smoke", "train"], default="pull")
    parser.add_argument("--fold", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "pull":
        pull()
    elif args.mode == "smoke":
        smoke()
    else:
        train_fold(args.fold)


if __name__ == "__main__":
    main()
