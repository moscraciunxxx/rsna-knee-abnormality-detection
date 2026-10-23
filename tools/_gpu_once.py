from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
for slug in (
    "vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai",
    "vitaliecervinschi/rsna-knee-clean-blend",
    "vitaliecervinschi/casmi26-val-c2f0-ours-nomvnf",
    "vitaliecervinschi/casmi26-val-c2f0-ours-nomv",
):
    try:
        st = api.kernels_status(slug)
        print(slug, st.status)
    except Exception as exc:
        print(slug, type(exc).__name__, exc)
