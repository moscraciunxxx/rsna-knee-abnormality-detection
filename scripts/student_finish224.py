"""Finish the last 224 resume epoch. Does not start the 288 run."""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch

from efficiency_student import OUT, SERIES, Student, build_table, cache_index, ids_and_labels
from student_resume import run_epochs


def main() -> None:
    device = torch.device("cuda")
    index = cache_index()
    table = build_table(index, pd.read_csv(SERIES, dtype=str))
    teacher = pd.read_csv(OUT / "teacher_reader.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    hold_ids, y_hold, hold_set, gold, soft = ids_and_labels()
    train_ids = [sid for sid in index if sid not in hold_set and sid not in gold and sid in soft.index and sid in teacher.index]
    ck = torch.load(OUT / "student_resume224.pt", map_location="cpu", weights_only=False)
    import timm

    model = Student("convnext_nano")
    model.enc = timm.create_model("convnext_nano", pretrained=False, num_classes=0)
    model.head = torch.nn.Linear(model.enc.num_features, 12)
    model.load_state_dict(ck["model"])
    model.to(device)
    run_epochs(model, "resume224", train_ids, teacher, soft, hold_ids, y_hold, index, table, device, 24, 224, 1)


if __name__ == "__main__":
    main()
