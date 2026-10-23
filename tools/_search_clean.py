from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
seen = set()
for term in ("rsna knee weights", "rsna knee coatnet", "rsna knee convnext", "rsna-knee"):
    rows = api.dataset_list(search=term, page=1) or []
    print("TERM", term, "n", len(rows))
    for row in rows:
        ref = getattr(row, "ref", None)
        if not ref or ref in seen:
            continue
        seen.add(ref)
        title = getattr(row, "title", "")
        print(ref, "|", title)
