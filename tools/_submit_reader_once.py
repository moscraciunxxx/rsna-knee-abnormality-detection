"""Confirm the 3-row file, then submit kernel version 2 once. No selection change."""

from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

dest = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\clean_blend_cpu")
dest.mkdir(parents=True, exist_ok=True)
api = KaggleApi()
api.authenticate()
api.kernels_output(
    "vitaliecervinschi/rsna-knee-clean-blend",
    str(dest),
    file_pattern=r"submission\.csv$",
    force=True,
    quiet=True,
)
frame = pd.read_csv(dest / "submission.csv")
print("rows", len(frame))
print("cols", list(frame.columns))
print("nan", int(frame.isna().sum().sum()))
if len(frame) != 3 or int(frame.isna().sum().sum()) != 0:
    raise SystemExit("csv check failed; not submitting")
resp = api.competition_submit_code(
    file_name="submission.csv",
    message="goodpjw2008 ConvNeXt reader fold0 only",
    competition="rsna-knee-abnormality-detection",
    kernel="vitaliecervinschi/rsna-knee-clean-blend",
    kernel_version=2,
)
print("SUBMIT", resp)
if hasattr(resp, "to_dict"):
    print(resp.to_dict())
