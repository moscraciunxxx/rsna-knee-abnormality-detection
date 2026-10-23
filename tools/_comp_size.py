from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
files = api.competition_list_files("rsna-knee-abnormality-detection", page_size=20) or []
total = 0
n = 0
for item in files:
    size = int(getattr(item, "total_bytes", 0) or getattr(item, "size", 0) or 0)
    name = getattr(item, "name", None) or getattr(item, "ref", "")
    total += size
    n += 1
    print(f"{size:15d} {name}")
print("listed", n, "bytes", total, "gib", round(total / (1024 ** 3), 2))
