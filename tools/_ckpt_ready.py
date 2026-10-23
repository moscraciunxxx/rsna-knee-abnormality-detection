from kaggle.api.kaggle_api_extended import KaggleApi

REF = "vitaliecervinschi/rsna-knee-f0-v3-ckpt"
api = KaggleApi()
api.authenticate()
try:
    print("status", api.dataset_status(REF))
except Exception as exc:
    print("status_err", type(exc).__name__, exc)
try:
    files = api.dataset_list_files(REF)
    names = []
    for item in getattr(files, "files", None) or []:
        names.append(getattr(item, "name", None) or getattr(item, "file_name", None) or str(item))
    print("files", names)
    print("raw_type", type(files))
except Exception as exc:
    print("files_err", type(exc).__name__, exc)
