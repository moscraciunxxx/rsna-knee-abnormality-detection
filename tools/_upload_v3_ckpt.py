"""Upload the v3 checkpoint as a private dataset and wait until it is ready."""

import hashlib
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

FOLDER = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\fold0_v3_out")
PT = FOLDER / "fold0_last.pt"
REF = "vitaliecervinschi/rsna-knee-f0-v3-ckpt"
EXPECT = "eb0e7aa36def97fb"

digest = hashlib.sha256(PT.open("rb").read(1024 * 1024)).hexdigest()[:16]
print("local_first_mb", digest, "bytes", PT.stat().st_size, flush=True)
if digest != EXPECT:
    raise SystemExit(f"refusing upload: hash {digest}")

api = KaggleApi()
api.authenticate()
try:
    existing = api.dataset_status(REF)
except Exception as exc:
    existing = None
    print("status_before", type(exc).__name__, exc, flush=True)
if existing:
    print("already_exists", existing, flush=True)
else:
    result = api.dataset_create_new(str(FOLDER), public=False, quiet=False, convert_to_csv=False)
    print("create_status", result.status, "error", result.error, "url", getattr(result, "url", None), flush=True)
    if str(result.status).lower() != "ok":
        raise SystemExit(1)

deadline = time.time() + 30 * 60
while time.time() < deadline:
    status = api.dataset_status(REF)
    print("dataset_status", status, flush=True)
    if str(status).lower() == "ready":
        raise SystemExit(0)
    if str(status).lower() in {"error", "failed", "deleted"}:
        raise SystemExit(1)
    time.sleep(15)
print("dataset not ready before deadline", flush=True)
raise SystemExit(1)
