"""Frozen competition schema: 12 findings including Baker's apostrophe.

Header must match data/sample_submission.csv exactly.
"""

from __future__ import annotations

ID_COLUMN = "StudyInstanceUID"

FINDINGS: tuple[str, ...] = (
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
)

SUBMISSION_COLUMNS: tuple[str, ...] = (ID_COLUMN, *FINDINGS)

N_FINDINGS = len(FINDINGS)
