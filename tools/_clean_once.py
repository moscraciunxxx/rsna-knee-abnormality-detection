from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
slug = "vitaliecervinschi/rsna-knee-clean-blend"
status = api.kernels_status(slug)
print("status", status.status, status.failure_message or "")
log = api.kernels_logs(slug) or ""
print("--- log ---")
print(log[-4000:])
