"""Watch the fold0 CPU reader until it writes submission.csv or errors."""

import time

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
slug = "vitaliecervinschi/rsna-knee-clean-blend"
deadline = time.time() + 40 * 60
buf = ""
while time.time() < deadline:
    try:
        for event in api.kernels_logs_stream(slug):
            data = event.get("data") or ""
            if not data:
                continue
            buf += data
            if any(k in data for k in ("loaded", "seconds/study", "wrote", "Traceback", "Error", "device")):
                print(data.rstrip(), flush=True)
            if "seconds/study" in buf and "wrote" in buf and "submission.csv" in buf:
                print("READY", flush=True)
                raise SystemExit(0)
            if "Traceback" in buf:
                print("FAILED", flush=True)
                raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        print("stream_error", type(exc).__name__, exc, flush=True)
    status = api.kernels_status(slug)
    print("status", status.status, status.failure_message or "", flush=True)
    if str(status.status).endswith("ERROR"):
        print(api.kernels_logs(slug) or "", flush=True)
        raise SystemExit(2)
    if str(status.status).endswith("COMPLETE") and "wrote" in (api.kernels_logs(slug) or ""):
        print(api.kernels_logs(slug), flush=True)
        raise SystemExit(0)
    time.sleep(20)
print("TIMEOUT", flush=True)
raise SystemExit(3)
