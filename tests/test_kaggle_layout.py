"""Competition layout: studies live under test_series/, not only repo root."""

from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.infer import (
    competition_input_dir,
    discover_study_ids,
    run_inference,
)
from rsna_knee.schema import SUBMISSION_COLUMNS


def _kaggle_like_fixture(root: Path) -> list[str]:
    ids = [
        "1.2.826.0.1.3680043.8.498.kaggleA",
        "1.2.826.0.1.3680043.8.498.kaggleB",
        "1.2.826.0.1.3680043.8.498.kaggleC",
    ]
    for i, study in enumerate(ids):
        series = root / "test_series" / study / f"series-{i}"
        series.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(30 + i)
        np.save(series / "stack.npy", rng.random((6, 8, 8)).astype(np.float32))
        (series / "slice.dcm").write_bytes(b"not-a-real-dicom")
    pd.DataFrame({"StudyInstanceUID": ids[:2]}).to_csv(root / "test.csv", index=False)
    return ids


def test_discover_studies_under_test_series_not_the_container_name(tmp_path: Path):
    ids = _kaggle_like_fixture(tmp_path)
    found = discover_study_ids(tmp_path)
    assert "test_series" not in found
    assert set(found) == set(ids)
    assert len(found) >= 3


def test_inference_on_kaggle_test_series_layout(tmp_path: Path):
    ids = _kaggle_like_fixture(tmp_path)
    table = run_inference(tmp_path)
    assert list(table.columns) == list(SUBMISSION_COLUMNS)
    assert "Baker's" in table.columns
    assert set(table["StudyInstanceUID"]) == set(ids)
    probs = table[list(SUBMISSION_COLUMNS[1:])].to_numpy()
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_competition_input_dir_prefers_kaggle_input(tmp_path: Path, monkeypatch):
    fake = tmp_path / "kaggle" / "input" / "rsna-knee-abnormality-detection"
    fake.mkdir(parents=True)
    (fake / "test.csv").write_text("StudyInstanceUID\nx\n", encoding="utf-8")
    monkeypatch.setattr(
        "rsna_knee.infer.KAGGLE_INPUT",
        fake,
    )
    assert competition_input_dir() == fake
