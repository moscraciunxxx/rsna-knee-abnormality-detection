from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
rows = api.kernels_list(mine=True, page_size=20, sort_by="dateRun") or []
for row in rows:
    ref = getattr(row, "ref", None)
    title = getattr(row, "title", None)
    print("REF", ref, "TITLE", title)
