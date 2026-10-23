from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
rows = api.dataset_list(user="dreaddevelopment", search="knee", page=1) or []
for row in rows:
    print(getattr(row, "ref", None), "|", getattr(row, "title", ""))
