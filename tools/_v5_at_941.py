"""One read of the v5 log at 21:41 PT. Does not push or cancel."""

import time
from datetime import datetime

from kaggle.api.kaggle_api_extended import KaggleApi

target = datetime.now().replace(hour=21, minute=41, second=30, microsecond=0)
wait = (target - datetime.now()).total_seconds()
if wait > 0:
    print("sleep_until", target.isoformat(), "seconds", round(wait), flush=True)
    time.sleep(wait)

api = KaggleApi()
api.authenticate()
slug = "vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai"
for attempt in range(4):
    status = api.kernels_status(slug)
    log = api.kernels_logs(slug) or ""
    print("attempt", attempt, "status", status.status, "log_chars", len(log), flush=True)
    lines = []
    for line in log.splitlines():
        low = line.lower()
        if any(k in low for k in ("epoch ", "s/iter", "resumed", "gpu", "cuda", "device", "abort", "collapse", "kill")):
            lines.append(line)
    print("--- matched ---", flush=True)
    print("\n".join(lines[-40:]), flush=True)
    if any("subset_oof" in line or "epoch 4 " in line for line in lines):
        break
    if attempt < 3:
        time.sleep(180)
print("DONE", flush=True)
