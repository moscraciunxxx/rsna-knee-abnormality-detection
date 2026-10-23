"""Drive the shipped 12-finding schema (including Baker's apostrophe)."""

from rsna_knee.schema import FINDINGS, ID_COLUMN, SUBMISSION_COLUMNS


def test_id_column_is_study_instance_uid():
    assert ID_COLUMN == "StudyInstanceUID"


def test_findings_are_the_official_twelve_in_order():
    assert list(FINDINGS) == [
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


def test_bakers_keeps_the_apostrophe():
    assert "Baker's" in FINDINGS
    assert "Bakers" not in FINDINGS
    assert "Baker’s" not in FINDINGS  # curly apostrophe is wrong


def test_submission_columns_match_sample_submission_header():
    assert list(SUBMISSION_COLUMNS) == [
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
