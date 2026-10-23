from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
for fold in range(5):
    slug = f"vitaliecervinschi/rsna-knee-cache-f{fold}-no-oai"
    st = api.kernels_status(slug)
    print(slug, st.status, st.failure_message or "")
