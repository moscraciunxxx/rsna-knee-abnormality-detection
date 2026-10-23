"""Stream one kernel vols.tar to disk. Prints bytes and MB/s. Stops on reset."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def signed_url(slug: str) -> str:
    owner, kernel = slug.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = kernel
        req.page_size = 10
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    for item in resp.files or []:
        if item.file_name == "vols.tar":
            return item.url
    raise SystemExit(f"{slug} has no vols.tar")


def stream(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".tar.part")
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink()
    if part.exists() and part.stat().st_size == 0:
        part.unlink()
    t0 = time.time()
    n = 0
    last = t0
    reported_minute = False
    try:
        with requests.get(url, stream=True, timeout=(30, 60)) as resp:
            resp.raise_for_status()
            total = resp.headers.get("Content-Length")
            print(f"content_length {total}", flush=True)
            with open(part, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    out.write(chunk)
                    n += len(chunk)
                    now = time.time()
                    if not reported_minute and now - t0 >= 60:
                        mbps = (n / 1e6) / (now - t0)
                        print(
                            f"MINUTE bytes {n} mb {n / 1e6:.1f} mbps {mbps:.2f}",
                            flush=True,
                        )
                        reported_minute = True
                    elif now - last >= 20:
                        mbps = (n / 1e6) / (now - t0)
                        print(f"bytes {n} mbps {mbps:.2f}", flush=True)
                        last = now
    except (requests.RequestException, ConnectionError, TimeoutError, OSError) as exc:
        print(f"RESET {type(exc).__name__} {exc} bytes {n}", flush=True)
        raise SystemExit(2) from exc
    part.replace(dest)
    elapsed = time.time() - t0
    print(f"DONE bytes {n} seconds {elapsed:.1f} mbps {(n / 1e6) / elapsed:.2f}", flush=True)


if __name__ == "__main__":
    slug = sys.argv[1]
    dest = Path(sys.argv[2])
    stream(signed_url(slug), dest)
