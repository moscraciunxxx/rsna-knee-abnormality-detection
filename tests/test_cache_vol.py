"""Cache roundtrip on synthetic reader volumes. No DICOM needed."""

import numpy as np

from rsna_knee.cache_vol import IMG, MAX_SLICES, load_study, save_study


def test_save_load_roundtrip(tmp_path):
    series = {
        "1.2.ser.a": np.zeros((4, IMG, IMG), dtype=np.uint8),
        "1.2.ser.b": np.full((MAX_SLICES, IMG, IMG), 9, dtype=np.uint8),
    }
    series["1.2.ser.a"][1, 2, 3] = 17
    save_study(tmp_path, "1.2.3", series)
    back = load_study(tmp_path, "1.2.3")
    assert set(back) == set(series)
    for name, vol in series.items():
        np.testing.assert_array_equal(back[name], vol)
