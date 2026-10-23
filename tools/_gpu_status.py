from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
slugs = [
    "vitaliecervinschi/casmi26-pcslice",
    "vitaliecervinschi/casmi26-val-c2f0-ours-nomvnf",
    "vitaliecervinschi/casmi26-val-c2f0-ours-nomv",
    "vitaliecervinschi/rsna-knee-nartaa-fold0-no-oai",
    "vitaliecervinschi/casmi26-rows-fold0",
    "vitaliecervinschi/casmi26-val-c1-np",
    "vitaliecervinschi/casmi26-val-c2-np",
    "vitaliecervinschi/casmi26-val-c2-fold0",
]
for slug in slugs:
    try:
        resp = api.kernels_status(slug)
        print(slug, resp.status, resp.failure_message or "")
    except Exception as exc:
        print(slug, "ERR", type(exc).__name__, exc)
