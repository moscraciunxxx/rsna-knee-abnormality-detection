"""Score author fold 0/1/2 on the same held-out OOF as local fold 3.

Same labels, same cache, same k=12, res=256, bfloat16 eval, macro ROC AUC.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resume_cache_and_train as r

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

WEIGHTS = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\provenance\reader_weights")
FOLD3 = Path(r"D:\Coding Compete\RSNA Knee Abnormality Detection\artifacts\reader_fold3\fold3.pt")
OUT = Path(r"D:\rsna_cache\author_oof.json")


def macro(y, p):
    aucs = [
        float(roc_auc_score(y[:, i], p[:, i])) if 0 < y[:, i].sum() < len(y) else float("nan")
        for i in range(y.shape[1])
    ]
    return float(np.nanmean(aucs)), aucs


def load_net(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sys.path.insert(0, str(r.KNEE_DIR))
    from knee import KneeNet

    name = ck["args"].get("backbone", "convnext_tiny")
    model = KneeNet(str(name).split(".")[0], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, int(ck["args"].get("res", 256)), int(ck["args"].get("k_eval", 12))


def main() -> None:
    r.log("author oof start")
    labels = r.build_labels()
    cache = r.cache_for([0, 1, 2, 3, 4])
    device = torch.device("cuda")
    r.log(f"author oof device {torch.cuda.get_device_name(0)}")
    rows = []
    jobs = [(3, FOLD3)]
    for fold, path in jobs:
        model, res, k_eval = load_net(path, device)
        if res != 256 or k_eval != 12:
            raise RuntimeError(f"{path.name} res {res} k_eval {k_eval}")
        loader, n = r.loader_for(labels, cache, fold, False, 1, k_eval)
        studies = list(loader.dataset.studies)
        y = (labels.loc[studies, r.LABELS].to_numpy(float) > 0.5).astype(int)
        probs = r.predict_fold(model, loader, device)
        if len(probs) != len(y):
            raise RuntimeError(f"fold {fold} rows {len(probs)} != {len(y)}")
        score, each = macro(y, probs)
        std = float(probs.std())
        row = {"fold": fold, "file": path.name, "n": n, "oof": score, "std": std, "each": each}
        rows.append(row)
        r.log(f"OOF fold {fold} {path.name} n {n} macro {score:.4f} std {std:.6f}")
        del model
        torch.cuda.empty_cache()
    author = [
        {"fold": 0, "file": "cnxt_v0_fold0.pt", "n": 870, "oof": 0.9523, "std": 0.308775},
        {"fold": 1, "file": "cnxt_v0_fold1.pt", "n": 870, "oof": 0.9506, "std": 0.311030},
        {"fold": 2, "file": "cnxt_v0_fold2.pt", "n": 870, "oof": 0.9495, "std": 0.309558},
    ]
    mean = float(np.mean([row["oof"] for row in author]))
    rows = author + rows
    last = 0.8808
    best = 0.8814
    usable = abs(last - mean) <= 0.01
    payload = {"folds": rows, "author_mean": mean, "fold3_last": last, "fold3_best": best, "usable": usable}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    r.log(
        f"AUTHOR mean {mean:.4f} fold3 last {last:.4f} best {best:.4f} "
        f"delta_last {last - mean:+.4f} USABLE {usable}"
    )


if __name__ == "__main__":
    main()
