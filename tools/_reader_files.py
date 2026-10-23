from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
ref = "goodpjw2008/rsna-knee-2-5d-convnext-reader"
files = api.dataset_list_files(ref)
print("type", type(files))
items = getattr(files, "files", None) or []
for item in items:
    name = getattr(item, "name", None) or getattr(item, "file_name", None)
    size = getattr(item, "total_bytes", None) or getattr(item, "size", None)
    print(f"{name}\t{size}")
print("n", len(items))
