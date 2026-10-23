from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
rows = api.competition_submissions("rsna-knee-abnormality-detection", page_size=5) or []
for row in rows:
    ref = getattr(row, "ref", None)
    status = getattr(row, "status", None)
    desc = getattr(row, "description", None) or getattr(row, "file_name", None)
    print(ref, status, desc)
