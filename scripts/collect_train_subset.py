"""Collect DICOMs for 64 small train studies. Background, bounded, resumable.

The file API has no search, so page the competition listing (200/page) and
keep files whose path contains a chosen StudyInstanceUID. Stops after
MAX_PAGES or when every study has at least one file. Skips files already on
disk, so reruns resume. Prints a manifest line per study at the end.

Writes: data/train_subset/{StudyUID}/{SeriesUID}/{SOP}.dcm + subset_ids.csv
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_STUDIES = 64
MAX_PAGES = 1500
EXE = r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe"
COMP = "rsna-knee-abnormality-detection"


def pick_studies() -> tuple[list[str], dict[str, set[str]]]:
    import pandas as pd

    folds = pd.read_csv(ROOT / "data" / "folds.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(ROOT / "data" / "train_series.csv", dtype=str)
    gold = set(folds.loc[folds["is_gold"] == 1, "StudyInstanceUID"])
    by_study: dict[str, set[str]] = {}
    for row in series.to_dict("records"):
        by_study.setdefault(row["StudyInstanceUID"], set()).add(row["SeriesInstanceUID"])
    cands = [u for u in by_study if u not in gold and len(by_study[u]) >= 5]
    cands.sort(key=lambda u: (len(by_study[u]), u))
    picked = cands[:N_STUDIES]
    return picked, {u: by_study[u] for u in picked}


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    studies, want_series = pick_studies()
    dest = ROOT / "data" / "train_subset"
    dest.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "subset_ids.csv").write_text(
        "StudyInstanceUID\n" + "\n".join(studies) + "\n", encoding="utf-8"
    )
    have_files = {s: 0 for s in studies}
    have_series: dict[str, set[str]] = {s: set() for s in studies}
    for s in studies:
        for f in (dest / s).rglob("*.dcm"):
            have_files[s] += 1
            have_series[s].add(f.parent.name)
    token = None
    t0 = time.time()
    for page in range(MAX_PAGES):
        resp = api.competition_list_files(COMP, page_token=token, page_size=200)
        files = list(resp.files if hasattr(resp, "files") else resp)
        for f in files:
            name = f.name
            for s in studies:
                if s in name:
                    out = dest / name
                    if not out.is_file():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        api.competition_download_file(COMP, name, path=str(dest), force=False, quiet=True)
                        have_files[s] += 1
                    have_series[s].add(Path(name).parent.name)
                    break
        token = getattr(resp, "nextPageToken", None)
        if page % 50 == 0:
            done = sum(1 for s in studies if want_series[s] <= have_series[s])
            print(f"page {page} complete_studies {done}/{len(studies)}", flush=True)
        if all(want_series[s] <= have_series[s] for s in studies) or not token:
            break
    print(f"pages {page + 1} seconds {time.time() - t0:.0f}", flush=True)
    for s in studies:
        print(f"{s} files={have_files[s]} series={len(have_series[s])}/{len(want_series[s])}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
