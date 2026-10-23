"""One bounded read of the running v5 log. Does not push or cancel."""

import threading

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
slug = "vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai"
status = api.kernels_status(slug)
print("status", status.status)
buf = []

def run() -> None:
    try:
        for event in api.kernels_logs_stream(slug):
            data = event.get("data") or ""
            if data:
                buf.append(data)
            if sum(len(x) for x in buf) > 250000:
                return
    except Exception as exc:
        buf.append(f"\nSTREAM_ERROR {type(exc).__name__} {exc}\n")

thread = threading.Thread(target=run)
thread.start()
thread.join(25)
text = "".join(buf)
print("chars", len(text))
for line in text.splitlines():
    low = line.lower()
    if any(k in low for k in ("epoch ", "s/iter", "resumed", "gpu", "cuda", "device", "abort", "std ", "subset_oof", "oof")):
        print(line)
