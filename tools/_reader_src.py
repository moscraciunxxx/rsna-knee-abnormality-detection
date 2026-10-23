from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

OUT = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\provenance\reader_weights")
api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    for name in ("knee.py", "preprocess.py"):
        req = ApiDownloadDatasetRequest()
        req.owner_slug = "goodpjw2008"
        req.dataset_slug = "rsna-knee-2-5d-convnext-reader"
        req.file_name = name
        resp = kaggle.datasets.dataset_api_client.download_dataset(req)
        (OUT / name).write_bytes(resp.content)
        print(name, len(resp.content))
        resp.close()
