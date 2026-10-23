from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    for fold in range(5):
        req = ApiListKernelSessionOutputRequest()
        req.user_name = "vitaliecervinschi"
        req.kernel_slug = f"rsna-cache-tar-f{fold}"
        req.page_size = 10
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        print("FOLD", fold)
        for item in resp.files or []:
            print(" ", item.file_name)
