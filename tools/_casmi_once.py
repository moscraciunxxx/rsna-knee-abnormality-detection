from datetime import datetime, timezone

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

SLUGS = [
    "casmi26-val-c2f0-ours-nomvnf",
    "casmi26-val-c2f0-ours-nomv",
]

api = KaggleApi()
api.authenticate()
now = datetime.now(timezone.utc)
print("now_utc", now.isoformat())
with api.build_kaggle_client() as kaggle:
    for slug in SLUGS:
        ref = "vitaliecervinschi/" + slug
        status = api.kernels_status(ref)
        req = ApiGetKernelRequest()
        req.user_name = "vitaliecervinschi"
        req.kernel_slug = slug
        meta = kaggle.kernels.kernels_api_client.get_kernel(req).metadata
        started = meta.last_run_time
        elapsed = None
        if started is not None:
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            elapsed = (now - started).total_seconds()
        print("SLUG", slug)
        print(" status", status.status)
        print(" failure", status.failure_message or "")
        print(" last_run_time", started)
        print(" elapsed_sec", None if elapsed is None else round(elapsed, 1))
        print(" current_version", meta.current_version_number)
        log = api.kernels_logs(ref) or ""
        lines = [line for line in log.splitlines() if line.strip()]
        print(" log_lines", len(lines))
        print(" log_head")
        for line in lines[:8]:
            print("  ", line[:240])
        print(" log_tail")
        for line in lines[-8:]:
            print("  ", line[:240])
