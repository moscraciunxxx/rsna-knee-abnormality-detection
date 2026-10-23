"""Cancel the v5 session if it is still running after the v4 hash."""

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiCancelKernelSessionRequest,
    ApiGetKernelRequest,
    ApiListKernelSessionOutputRequest,
)

SLUG = "rsna-knee-nartaa-fold0-no-oai"
OWNER = "vitaliecervinschi"

api = KaggleApi()
api.authenticate()
status = api.kernels_status(f"{OWNER}/{SLUG}")
print("status", status.status, status.failure_message or "")
log = api.kernels_logs(f"{OWNER}/{SLUG}") or ""
print("--- log head ---")
print("\n".join(log.splitlines()[:30]))
with api.build_kaggle_client() as kaggle:
    req = ApiGetKernelRequest()
    req.user_name = OWNER
    req.kernel_slug = SLUG
    meta = kaggle.kernels.kernels_api_client.get_kernel(req).metadata
    print("version", meta.current_version_number, "kernel_id", meta.id)
    out = ApiListKernelSessionOutputRequest()
    out.user_name = OWNER
    out.kernel_slug = SLUG
    out.page_size = 5
    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(out)
    print("log_len", len(resp.log or ""))
    for item in resp.files or []:
        print("file", item.file_name, getattr(item, "url", "")[:80])
