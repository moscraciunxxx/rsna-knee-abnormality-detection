from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
files = api.competition_list_files("rsna-knee-abnormality-detection")
for item in files or []:
    name = getattr(item, "name", None) or getattr(item, "ref", None)
    size = getattr(item, "total_bytes", None) or getattr(item, "size", None)
    print(name, size)
