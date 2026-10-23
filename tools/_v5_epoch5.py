"""One read of the v5 log at 22:40 PT. Does not push or cancel."""

import threading
import time
from datetime import datetime

target = datetime.now().replace(hour=22, minute=40, second=30, microsecond=0)
wait = (target - datetime.now()).total_seconds()
if wait > 0:
    print("sleep_until", target.isoformat(), "seconds", round(wait), flush=True)
    time.sleep(wait)

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
slug = "vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai"
status = api.kernels_status(slug)
print("status", status.status, flush=True)
buf = []

def run() -> None:
    try:
        for event in api.kernels_logs_stream(slug):
            data = event.get("data") or ""
            if data:
                buf.append(data)
            if sum(len(x) for x in buf) > 400000:
                return
    except Exception as exc:
        buf.append(f"\nSTREAM_ERROR {type(exc).__name__} {exc}\n")

thread = threading.Thread(target=run)
thread.start()
thread.join(30)
text = "".join(buf)
print("chars", len(text), flush=True)
for line in text.splitlines():
    low = line.lower()
    if any(k in low for k in ("epoch ", "kill", "collapse", "abort", "subset_oof", "std ")):
        print(line, flush=True)
print("DONE", flush=True)
