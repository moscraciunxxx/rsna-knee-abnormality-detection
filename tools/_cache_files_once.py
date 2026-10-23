from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    for fold in range(5):
        req = ApiListKernelSessionOutputRequest()
        req.user_name = "vitaliecervinschi"
        req.kernel_slug = f"rsna-knee-cache-f{fold}-no-oai"
        req.page_size = 5
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        files = resp.files or []
        print("FOLD", fold, "nfiles_page", len(files), "next", bool(resp.next_page_token))
        for item in files[:8]:
            print(" ", item.file_name)
