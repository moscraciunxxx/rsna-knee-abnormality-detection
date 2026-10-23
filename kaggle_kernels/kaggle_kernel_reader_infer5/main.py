"""Five-fold reader inference. Draft. Do not push until folds 3 and 4 exist.

Folds 0-2 come from goodpjw2008/rsna-knee-2-5d-convnext-reader.
Folds 3-4 come from the fold34 kernel output. The submission is the mean
of the five sigmoids. Internet off.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"


def find_named(names: list[str]) -> list[Path]:
    skip = {"train_series", "test_series", "train_images", "test_images"}
    found = {}
    root = Path("/kaggle/input")
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith("1.2.")]
        for name in files:
            if name in names and name not in found:
                found[name] = Path(dirpath) / name
    missing = [name for name in names if name not in found]
    if missing:
        raise FileNotFoundError(f"missing checkpoints {missing}")
    return [found[name] for name in names]


def main() -> None:
    names = ["cnxt_v0_fold0.pt", "cnxt_v0_fold1.pt", "cnxt_v0_fold2.pt", "fold3.pt", "fold4.pt"]
    ckpts = find_named(names)
    code = ckpts[0].parent
    if not (code / "knee.py").is_file():
        code = next(p.parent for p in ckpts if (p.parent / "knee.py").is_file())
    sys.path.insert(0, str(code))
    from knee import N_SLOTS, KneeNet, build_study_table, window_centres
    from preprocess import MAX_SLICES, SIZE, load_series

    device = torch.device("cuda")
    models = []
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["model"] if "model" in ck else ck
        model = KneeNet("convnext_tiny", pretrained=False)
        model.load_state_dict(state)
        models.append(model.to(device).eval())
        print("loaded", path.name, flush=True)
    comp = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
    if not (comp / "test.csv").is_file():
        comp = Path("/kaggle/input/rsna-knee-abnormality-detection")
    studies = pd.read_csv(comp / "test.csv", dtype={ID_COL: str})[ID_COL].tolist()
    series = pd.read_csv(comp / "test_series.csv", dtype=str)
    root = comp / "test_series"
    n_slices = {}
    paths = {}
    for st, sr in zip(series.StudyInstanceUID, series.SeriesInstanceUID):
        folder = root / st / sr
        paths[sr] = str(folder)
        n_slices[sr] = len(list(folder.glob("*.dcm"))) if folder.is_dir() else 0
    table = build_study_table(series, n_slices)
    rows = []
    k = 12
    t0 = time_start()
    for n, st in enumerate(studies, start=1):
        slots = table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, k, 3, SIZE, SIZE), np.uint8)
        mask = np.zeros(N_SLOTS, bool)
        pos = np.zeros((N_SLOTS, k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            vol, _ = load_series(paths[cands[0]], 0)
            nvol = len(vol)
            if nvol < 3:
                continue
            centres = window_centres(nvol, k, False)
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, nvol - 1)] for j in centres])
            mask[s] = True
            pos[s] = centres / max(nvol - 1, 1)
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        mt = torch.from_numpy(mask).unsqueeze(0).to(device)
        pt = torch.from_numpy(pos).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = [torch.sigmoid(model(xt, mt, pt, res=256).float()) for model in models]
            prob = torch.stack(preds).mean(0)[0].cpu().numpy()
        rows.append({ID_COL: st, **{name: float(prob[i]) for i, name in enumerate(LABELS)}})
        if n == len(studies):
            sec = (time_start() - t0) / max(n, 1)
            print(f"seconds/study {sec:.3f} projected_hours_1300 {sec * 1300 / 3600:.3f}", flush=True)
    frame = pd.DataFrame(rows)[[ID_COL, *LABELS]]
    out = Path("/kaggle/working/submission.csv")
    frame.to_csv(out, index=False)
    print("wrote", out, "rows", len(frame), flush=True)


def time_start() -> float:
    import time
    return time.perf_counter()


if __name__ == "__main__":
    main()
