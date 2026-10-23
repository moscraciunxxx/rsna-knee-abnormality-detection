"""Drive the shipped inference path: folder discovery + 12-finding CSV."""

from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.infer import discover_study_ids, run_inference, write_inference_csv
from rsna_knee.schema import SUBMISSION_COLUMNS


def _write_volume(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(path, rng.random((10, 12, 12)).astype(np.float32))


def _make_fixture(root: Path) -> list[str]:
    ids = [
        "1.2.826.0.1.3680043.8.498.fixtureA",
        "1.2.826.0.1.3680043.8.498.fixtureB",
        "1.2.826.0.1.3680043.8.498.fixtureC",
        "1.2.826.0.1.3680043.8.498.fixtureD",
    ]
    for i, study in enumerate(ids):
        _write_volume(root / study / "series-1" / "slice_stack.npy", seed=10 + i)
        _write_volume(root / study / "series-2" / "slice_stack.npy", seed=20 + i)
    # A stale test.csv listing only 2 of 4 studies — discovery must still find folders.
    pd.DataFrame({"StudyInstanceUID": ids[:2]}).to_csv(root / "test.csv", index=False)
    return ids


def test_discover_study_ids_from_folders_not_only_test_csv(tmp_path: Path):
    ids = _make_fixture(tmp_path)
    found = discover_study_ids(tmp_path)
    assert set(found) == set(ids)
    assert len(found) >= 3


def test_run_inference_writes_official_header_and_unit_probs(tmp_path: Path):
    ids = _make_fixture(tmp_path)
    table = run_inference(tmp_path)
    assert list(table.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in table.columns
    assert len(table) >= 3
    assert set(table["StudyInstanceUID"]) == set(ids)
    probs = table[list(SUBMISSION_COLUMNS[1:])].to_numpy()
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_write_inference_csv_roundtrip(tmp_path: Path):
    _make_fixture(tmp_path)
    out = tmp_path / "submission.csv"
    write_inference_csv(tmp_path, out)
    text = out.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert header == ",".join(SUBMISSION_COLUMNS)
    assert "Baker's" in header
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == list(SUBMISSION_COLUMNS)
    assert len(reloaded) >= 3
