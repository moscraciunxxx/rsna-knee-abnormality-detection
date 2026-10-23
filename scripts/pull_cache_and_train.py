"""Pull cache tars f1-f4 after f0, extract, verify, then local training.

f0 is streamed by scripts/stream_kernel_tar.py. This process waits for that
file, extracts it, and starts the 3060 smoke test while the later folds download.
"""

from __future__ import annotations

import csv
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
PY = r"C:\Users\moscr\anaconda3\python.exe"
STREAM = ROOT / "scripts" / "stream_kernel_tar.py"
TRAIN = ROOT / "scripts" / "reader_local.py"
OWNER = "vitaliecervinschi"


def log(msg: str) -> None:
    print(msg, flush=True)


def list_output(slug: str):
    api = KaggleApi()
    api.authenticate()
    files = []
    log_text = ""
    token = None
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = OWNER
            req.kernel_slug = slug
            req.page_size = 20
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            files.extend(resp.files or [])
            if resp.log:
                log_text = resp.log
            token = resp.next_page_token
            if not token:
                break
    return files, log_text


def fetch_manifest(fold: int, dest: Path) -> Path | None:
    slug = f"rsna-knee-cache-f{fold}-no-oai"
    api = KaggleApi()
    api.authenticate()
    url = None
    token = None
    try:
        with api.build_kaggle_client() as kaggle:
            for _ in range(3):
                req = ApiListKernelSessionOutputRequest()
                req.user_name = OWNER
                req.kernel_slug = slug
                req.page_size = 20
                if token:
                    req.page_token = token
                resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                url = next((item.url for item in (resp.files or []) if item.file_name == "manifest.csv"), None)
                if url or not resp.next_page_token:
                    break
                token = resp.next_page_token
    except requests.RequestException as exc:
        log(f"manifest list failed fold {fold} {exc}")
        return None
    if url is None:
        log(f"manifest missing from listing fold {fold}")
        return None
    try:
        resp = requests.get(url, timeout=(30, 60))
        resp.raise_for_status()
    except requests.RequestException as exc:
        log(f"manifest download failed fold {fold} {exc}")
        return None
    text = resp.content
    if text[:1] == b"<" or b"StudyInstanceUID" not in text[:200]:
        log(f"manifest not csv fold {fold} head {text[:40]!r}")
        return None
    dest.write_bytes(text)
    return dest


def extract(tar_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
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
            with open(target, "wb") as out:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            n += 1
            if n % 100 == 0:
                log(f"extracted {n} -> {dest}")
    return n


def verify(dest: Path, manifest: Path | None, expected_n: int | None) -> tuple[bool, str]:
    npz = sorted(p for p in dest.glob("*.npz"))
    if manifest is not None and manifest.is_file():
        with manifest.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        ids = {row["StudyInstanceUID"] for row in rows}
        got = {p.stem for p in npz}
        missing = ids - got
        extra = got - ids
        ok = not missing and not extra and len(npz) == len(rows)
        return ok, f"npz {len(npz)} manifest {len(rows)} missing {len(missing)} extra {len(extra)}"
    if expected_n is not None:
        ok = len(npz) == expected_n
        return ok, f"npz {len(npz)} timing_n {expected_n} manifest_absent"
    return False, f"npz {len(npz)} no manifest and no expected count"


def files_from_log(text: str, fold: int) -> int | None:
    needle = f"fold {fold} files "
    for line in text.splitlines():
        if needle in line:
            try:
                return int(line.split(needle, 1)[1].split()[0])
            except ValueError:
                return None
    return None


def wait_f0() -> None:
    tar = CACHE / "f0" / "vols.tar"
    part = CACHE / "f0" / "vols.tar.part"
    while not tar.is_file() or part.exists():
        time.sleep(15)
        if part.exists():
            log(f"f0 part {part.stat().st_size}")
    log(f"f0 tar ready {tar.stat().st_size}")


def pull_fold(fold: int) -> None:
    dest_dir = CACHE / f"f{fold}"
    tar = dest_dir / "vols.tar"
    log(f"download f{fold}")
    proc = subprocess.run(
        [PY, str(STREAM), f"{OWNER}/rsna-cache-tar-f{fold}", str(tar)],
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"download f{fold} exit {proc.returncode}")
    _, kernel_log = list_output(f"rsna-cache-tar-f{fold}")
    expected = files_from_log(kernel_log, fold)
    manifest = fetch_manifest(fold, dest_dir / "manifest.csv")
    n = extract(tar, dest_dir)
    log(f"f{fold} extracted {n}")
    ok, msg = verify(dest_dir, manifest, expected)
    log(f"f{fold} verify {ok} {msg}")
    if not ok:
        raise SystemExit(f"f{fold} verify failed; tar kept")
    tar.unlink()
    log(f"f{fold} tar deleted")


def main() -> None:
    wait_f0()
    _, kernel_log = list_output("rsna-cache-tar-f0")
    expected = files_from_log(kernel_log, 0)
    manifest = fetch_manifest(0, CACHE / "f0" / "manifest.csv")
    n = extract(CACHE / "f0" / "vols.tar", CACHE / "f0")
    log(f"f0 extracted {n}")
    ok, msg = verify(CACHE / "f0", manifest, expected)
    log(f"f0 verify {ok} {msg}")
    if not ok:
        raise SystemExit("f0 verify failed; tar kept")
    (CACHE / "f0" / "vols.tar").unlink()
    log("f0 tar deleted")

    smoke = subprocess.Popen(
        [PY, str(TRAIN), "--mode", "smoke", "--steps", "50"],
        cwd=str(ROOT),
    )
    log(f"smoke pid {smoke.pid}")
    try:
        for fold in range(1, 5):
            pull_fold(fold)
    finally:
        code = smoke.wait()
    log(f"smoke exit {code}")
    if code != 0:
        raise SystemExit(code)
    log("train fold 3")
    train = subprocess.run([PY, str(TRAIN), "--mode", "train", "--fold", "3"], check=False)
    raise SystemExit(train.returncode)


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, ConnectionError, TimeoutError, OSError) as exc:
        log(f"RESET {type(exc).__name__} {exc}")
        raise SystemExit(2) from exc
