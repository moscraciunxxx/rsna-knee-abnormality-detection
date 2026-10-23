"""Wait for v5's hash, RESUMED line, and step-50 s/iter. Cancel on a bad hash."""

from __future__ import annotations

import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest

SLUG = "vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai"
V3 = "eb0e7aa36def97fb"
V4 = "1c0e40b7fe4980fc"
NEEDLE_RESUME = "RESUMED from v3 epoch 3"
DEADLINE_S = 45 * 60


def cancel(api: KaggleApi, session_id: int, reason: str) -> None:
    print(f"CANCEL session {session_id}: {reason}", flush=True)
    with api.build_kaggle_client() as kaggle:
        req = ApiCancelKernelSessionRequest()
        req.kernel_session_id = int(session_id)
        resp = kaggle.kernels.kernels_api_client.cancel_kernel_session(req)
        print("cancel_response", getattr(resp, "error_message", None) or resp, flush=True)


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    status = api.kernels_status(SLUG)
    print("status", status.status, getattr(status, "failure_message", "") or "", flush=True)
    t0 = time.time()
    buf = ""
    session_id = None
    while time.time() - t0 < DEADLINE_S:
        try:
            for event in api.kernels_logs_stream(SLUG):
                data = event.get("data") or ""
                if not data:
                    continue
                if "kernelSessionId" in event and event["kernelSessionId"]:
                    session_id = int(event["kernelSessionId"])
                buf += data
                if not data.endswith("\n"):
                    buf += "\n"
                # Keep the buffer bounded but retain the header.
                if len(buf) > 400_000:
                    buf = buf[-200_000:]
                if "first_mb_sha256" in data or "RESUMED" in data or "s/iter" in data or "ABORT" in data:
                    print(data.rstrip(), flush=True)
                if V4 in buf and V3 not in buf:
                    print("BAD HASH v4", flush=True)
                    if session_id:
                        cancel(api, session_id, "v4 checkpoint")
                    raise SystemExit(2)
                if "first_mb_sha256" in buf and V3 not in buf and "ABORT resume" in buf:
                    print("BAD OR MISSING HASH", flush=True)
                    if session_id:
                        cancel(api, session_id, "hash mismatch")
                    raise SystemExit(2)
                if V3 in buf and "RESUMED from v3 epoch" in buf and NEEDLE_RESUME not in buf:
                    print("BAD RESUME LINE", flush=True)
                    if session_id:
                        cancel(api, session_id, "resume line mismatch")
                    raise SystemExit(2)
                resumed_at = buf.find(NEEDLE_RESUME)
                if V3 in buf and resumed_at >= 0 and "s/iter" in buf[resumed_at:]:
                    print("CONFIRMED", flush=True)
                    raise SystemExit(0)
                if "ABORT resume" in buf:
                    print("ABORT SEEN", flush=True)
                    if session_id:
                        cancel(api, session_id, "resume abort")
                    raise SystemExit(2)
        except SystemExit:
            raise
        except Exception as exc:
            print("stream_error", type(exc).__name__, exc, flush=True)
        status = api.kernels_status(SLUG)
        print("status", status.status, f"elapsed {time.time() - t0:.0f}s", flush=True)
        if str(status.status).endswith("ERROR") or str(status.status).endswith("CANCEL"):
            print("KERNEL ENDED", status.status, status.failure_message or "", flush=True)
            raise SystemExit(2)
        time.sleep(20)
    print("DEADLINE without confirmation", flush=True)
    raise SystemExit(3)


if __name__ == "__main__":
    main()
