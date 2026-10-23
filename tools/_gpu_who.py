from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
rows = api.kernels_list(mine=True, page_size=15, sort_by="dateRun") or []
for row in rows:
    ref = getattr(row, "ref", None)
    if not ref:
        continue
    try:
        st = api.kernels_status(ref)
    except Exception as exc:
        print(ref, type(exc).__name__)
        continue
    print(ref, st.status)
