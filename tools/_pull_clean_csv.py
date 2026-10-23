from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

dest = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\clean_blend")
dest.mkdir(parents=True, exist_ok=True)
api = KaggleApi()
api.authenticate()
api.kernels_output(
    "vitaliecervinschi/rsna-knee-clean-blend",
    str(dest),
    file_pattern=r"submission\.csv",
    force=True,
    quiet=False,
)
frame = pd.read_csv(dest / "submission.csv")
print("rows", len(frame))
print("cols", list(frame.columns))
print("nan", int(frame.isna().sum().sum()))
print(frame.head().to_string(index=False))
