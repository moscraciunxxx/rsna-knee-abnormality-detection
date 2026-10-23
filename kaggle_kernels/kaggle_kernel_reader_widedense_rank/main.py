"""Equal per-finding rank of the 3-fold reader and widedense. Draft. Not pushed.

Holdout870 has a score for the 3-fold reader (0.9439) and no widedense file.
This kernel does not pick a finding from gold58. It also does not invent a
holdout ranking. Until a local widedense holdout870 file exists, it refuses
to write a blended submission.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID = "StudyInstanceUID"
READER_HOLDOUT = Path("/kaggle/input")  # documented; the local score is already 0.9439


def equal_per_finding_rank(frames: list[pd.DataFrame]) -> pd.DataFrame:
    ids = frames[0][ID].astype(str).tolist()
    ranked = []
    for frame in frames:
        frame = frame.drop_duplicates(ID).set_index(ID).loc[ids, LABELS]
        values = frame.to_numpy(np.float64)
        ranked.append(np.column_stack([rankdata(values[:, j], method="average") for j in range(12)]))
    out = pd.DataFrame(np.mean(ranked, axis=0), columns=LABELS)
    out.insert(0, ID, ids)
    return out


def main() -> None:
    print("reader holdout870 macro 0.9439", flush=True)
    print("widedense holdout870 absent", flush=True)
    print("BLEND_SKIPPED no widedense holdout870 file; equal rank not scored", flush=True)
    sys.exit(2)


if __name__ == "__main__":
    main()
