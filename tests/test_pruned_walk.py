"""The next fold-0 script must not walk DICOM trees while looking for a CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_kernel_nartaa_fold0"))

import driver


def test_pruned_walk_reads_the_label_csv_and_skips_train_series(tmp_path):
    series = tmp_path / "train_series" / "1.2.3"
    series.mkdir(parents=True)
    (series / "a.dcm").write_bytes(b"x")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "llm_labels_v4_blend.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
    seen = list(driver.pruned_walk(tmp_path))
    names = [Path(dirpath).name for dirpath, _files in seen]
    files = [name for _dirpath, found in seen for name in found]
    assert "llm_labels_v4_blend.csv" in files
    assert "1.2.3" not in names
    assert "a.dcm" not in files
