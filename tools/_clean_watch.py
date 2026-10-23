"""Watch the clean reader run until submission.csv is written or it errors."""

import time

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
slug = "vitaliecervinschi/rsna-knee-clean-blend"
deadline = time.time() + 90 * 60
buf = ""
while time.time() < deadline:
    try:
        for event in api.kernels_logs_stream(slug):
            data = event.get("data") or ""
            if not data:
                continue
            buf += data
            if any(k in data for k in ("loaded", "wrote", "Traceback", "Error", "refusing", "studies ")):
                print(data.rstrip(), flush=True)
            if "wrote" in data and "submission.csv" in data:
                print("CSV_LINE", flush=True)
                raise SystemExit(0)
            if "Traceback" in buf or "refusing forbidden" in buf:
                print("FAILED", flush=True)
                raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        print("stream_error", type(exc).__name__, exc, flush=True)
    status = api.kernels_status(slug)
    print("status", status.status, status.failure_message or "", flush=True)
    if str(status.status).endswith("COMPLETE") or str(status.status).endswith("ERROR"):
        log = api.kernels_logs(slug) or ""
        print(log[-3000:], flush=True)
        raise SystemExit(0 if "COMPLETE" in str(status.status) else 2)
    time.sleep(30)
print("STILL_RUNNING", flush=True)
raise SystemExit(3)
