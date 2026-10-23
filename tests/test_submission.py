"""Drive the shipped submission builder — columns, Baker's, clip to [0, 1]."""

import numpy as np

from rsna_knee.schema import SUBMISSION_COLUMNS
from rsna_knee.submission import build_submission, clip_probabilities


def test_builder_output_columns_equal_official_header():
    ids = ["s1", "s2", "s3"]
    probs = np.full((3, 12), 0.4)
    table = build_submission(ids, probs)
    assert list(table.columns) == list(SUBMISSION_COLUMNS)
    assert list(table.columns) == [
        "StudyInstanceUID",
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]


def test_bakers_column_present_with_apostrophe():
    table = build_submission(["a", "b", "c"], np.zeros((3, 12)))
    assert "Baker's" in table.columns
    assert table["Baker's"].tolist() == [0.0, 0.0, 0.0]


def test_probabilities_clip_to_unit_interval():
    raw = np.array([[-0.2, 0.3, 1.7]], dtype=float)
    clipped = clip_probabilities(raw)
    assert clipped.min() >= 0.0
    assert clipped.max() <= 1.0
    np.testing.assert_allclose(clipped[0, 0], 0.0)
    np.testing.assert_allclose(clipped[0, 2], 1.0)


def test_builder_clips_out_of_range_scores():
    ids = ["u1", "u2", "u3"]
    probs = np.array(
        [
            [-5.0] * 12,
            [0.5] * 12,
            [9.0] * 12,
        ],
        dtype=float,
    )
    table = build_submission(ids, probs)
    values = table[list(SUBMISSION_COLUMNS[1:])].to_numpy()
    assert values.min() >= 0.0
    assert values.max() <= 1.0
    assert table["StudyInstanceUID"].tolist() == ids
