"""First-MB SHA-256 of the reader checkpoints, plus infer.py text."""

import hashlib
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

OWNER = "goodpjw2008"
SLUG = "rsna-knee-2-5d-convnext-reader"
FILES = ["cnxt_v0_fold0.pt", "cnxt_v0_fold1.pt", "cnxt_v0_fold2.pt", "infer.py"]
OUT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\provenance\reader_weights")
OUT.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    for name in FILES:
        req = ApiDownloadDatasetRequest()
        req.owner_slug = OWNER
        req.dataset_slug = SLUG
        req.file_name = name
        resp = kaggle.datasets.dataset_api_client.download_dataset(req)
        if name.endswith(".py"):
            data = resp.content
            (OUT / name).write_bytes(data)
            print(name, "bytes", len(data))
        else:
            blob = resp.content[: 1024 * 1024]
            digest = hashlib.sha256(blob).hexdigest()[:16]
            print(f"{name} first_mb_sha256 {digest} read {len(blob)}")
        resp.close()
